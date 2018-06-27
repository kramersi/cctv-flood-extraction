import os
import glob
import re
import numpy as np
import cv2

from keras.models import Input, Model, load_model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import to_categorical

from keras_unet.utils import f1_loss, f1_np, iou_np, precision_np, recall_np, error_np


def conv_block(m, dim, acti, bn, res, do=0):
    """ creates convolutional block for creating u-net

    """
    n = Conv2D(dim, 3, activation=acti, padding='same')(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same')(n)
    n = BatchNormalization()(n) if bn else n
    return Concatenate()([m, n]) if res else n


def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
    if depth > 0:
        n = conv_block(m, dim, acti, bn, res)
        m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
        m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
        n = Concatenate()([n, m])
        m = conv_block(n, dim, acti, bn, res)
    else:
        m = conv_block(m, dim, acti, bn, res, do)
    return m


def load_masks(path):
    files = glob.glob(os.path.join(path, '*'))
    first_img = load_img(files[0])

    n = len(files)
    w = first_img.width
    h = first_img.height
    x = np.empty((n, w, h))

    for i, f in enumerate(files):
        im = load_img(f)
        x[i, :, :] = img_to_array(im)[:, :, 0].astype(np.int8)

    return x


def load_images(path, sort=False):
    files = glob.glob(os.path.join(path, '*'))
    if sort is True:
        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    first_img = load_img(files[0])

    n = len(files)
    w = first_img.width
    h = first_img.height
    x = np.empty((n, h, w, 3))

    for i, f in enumerate(files):
        im = load_img(f)
        x[i, :, :, :] = img_to_array(im).astype(np.float32)

    return x


def channel_mean_stdev(img):
    m = np.mean(img, axis=(0, 1, 2))
    s = np.std(img, axis=(0, 1, 2))
    return m, s


def store_prediction(predictions, images, output_dir):
    class_mapping = {0: [0, 0, 0], 1: [0, 0, 255]}
    count = 0
    for pred, img in zip(predictions, images):
        best_pred = np.argmax(pred, axis=-1)  # take label of maximum probability

        # # resize the color map to fit image
        # img_crop = np.uint8(img[0, :, :, :] * 255)

        # overlay cmap with image
        prediction = np.repeat(best_pred[:, :, np.newaxis], 3, axis=-1)
        # fill prediction with right rgb colors
        for label, rgb in class_mapping.items():
            prediction[prediction[:, :, 0] == label] = rgb

        overlay_img = cv2.addWeighted(np.uint8(img), 0.8, np.uint8(prediction), 0.5, 0)
        cv2.imwrite(os.path.join(output_dir, 'pred' + str(count) + '.png'),
                    cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        count +=1


class UNet(object):
    """ Class which create UNet model and trains it and test it

    U-Net: Convolutional Networks for Biomedical Image Segmentation
    (https://arxiv.org/abs/1505.04597)

    Arguments:
        img_shape: (height, width, channels)
        n_class: number of output channels, classes to predict in one-hot coding
        root_features: number of channels of the first conv
        layers: zero indexed depth of the U-structure, number of layers
        inc_rate: rate at which the conv channels will increase
        activation: activation function after convolutions
        dropout: amount of dropout in the contracting part
        batch_norm: adds Batch Normalization if true
        max_pool: use strided conv instead of maxpooling if false
        up_conv: use transposed conv instead of upsamping + conv if false
        residual: add residual connections around each conv block if true
    """
    def __init__(self, img_shape, n_class=2, root_features=64, layers=4, inc_rate=1., activation='relu', dropout=0.5,
                 batch_norm=False, max_pool=True, up_conv=True, residual=False):
        self.img_shape = img_shape
        self.n_class = n_class
        self.root_features = root_features
        self.layers = layers
        self.inc_rate = inc_rate
        self.activation = activation
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.max_pool = max_pool
        self.up_conv = up_conv
        self.residuals = residual

        self.tr_mean = None
        self.tr_std = None

        # define model
        i = Input(shape=img_shape)
        o = level_block(i, root_features, layers, inc_rate, activation, dropout, batch_norm, max_pool, up_conv, residual)
        o = Conv2D(n_class, 1, activation='sigmoid')(o)
        self.model = Model(inputs=i, outputs=o)

    def normalize(self, x):
        self.tr_mean = np.array([76.51, 75.41, 71.02])
        self.tr_std = np.array([76.064, 75.23, 75.03])

        if self.tr_mean is None:
            print('mean and standard deviation of training pictures not calculated yet, calculating...')
            self.tr_mean, self.tr_std = channel_mean_stdev(x)
            print('mean: ', self.tr_mean, 'std: ', self.tr_std)

        x_norm = (x - self.tr_mean) / self.tr_std

        return x_norm

    def train(self, model_dir, train_dir, valid_dir, epochs=20, batch_size=4, augmentation=True, normalisation=True):
        """ trains a unet instance on keras. With on-line data augmentation to diversify training samples in each batch.

            example of defining paths
            train_dir = "E:\\watson_for_trend\\3_select_for_labelling\\train_cityscape\\"
            model_dir = "E:\\watson_for_trend\\5_train\\cityscape_l5f64c3n8e20\\"

        """
        seed = 1234

        x_tr = load_images(os.path.join(train_dir, 'images', '0'))  # load training pictures in numpy array
        shape = x_tr.shape  # pic_nr x width x height x depth
        n_train = shape[0]  # len(image_generator)

        # compile model with optimizer and loss function
        self.model.compile(optimizer=Adam(lr=0.001), loss=f1_loss,
                      metrics=['acc', 'categorical_crossentropy'])

        # define callbacks
        mc = ModelCheckpoint(os.path.join(model_dir, 'model.h5'), save_best_only=True, save_weights_only=False)
        es = EarlyStopping(patience=9)
        tb = TensorBoard(log_dir=model_dir)

        y_tr = load_masks(os.path.join(train_dir, 'masks', '0'))  # load mask arrays
        x_va = load_images(os.path.join(valid_dir, 'images', '0'))
        y_va = load_masks(os.path.join(valid_dir, 'masks', '0'))
        n_valid = x_va.shape[0]

        # data normalisation
        if normalisation is True:
            x_tr = self.normalize(x_tr)
            x_va = self.normalize(x_va)

        # create one-hot
        y_tr = to_categorical(y_tr, self.n_class)
        y_va = to_categorical(y_va, self.n_class)

        if augmentation:

            image_datagen = ImageDataGenerator(featurewise_center=False,
                                               featurewise_std_normalization=False,
                                               width_shift_range=0.1,
                                               height_shift_range=0.1,
                                               horizontal_flip=True,
                                               zoom_range=0.0)

            # calculate mean and stddeviation of training sample for normalisation (if featurwise center is true)
            # image_datagen.fit(x_tr, seed=seed)

            # create image generator for online data augmentation
            train_generator = image_datagen.flow(x_tr, y_tr, batch_size=batch_size, shuffle=True, seed=seed)
            valid_generator = (x_va, y_va)

            # train unet with image_generator
            self.model.fit_generator(train_generator,
                                     validation_data=valid_generator,
                                     steps_per_epoch=n_train / batch_size,
                                     validation_steps=n_valid / batch_size,
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=[mc, es, tb],
                                     use_multiprocessing=False,
                                     workers=4)
        else:
            self.model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs, batch_size=batch_size,
                           shuffle=True, callbacks=[mc, es, tb])

        scores = self.model.evaluate(x_va, y_va, verbose=1)
        print('scores', scores)

    def test(self, model_dir, test_img_dir, output_dir, batch_size=4, train_dir=None):

        x_va = load_images(os.path.join(test_img_dir, 'images', '0'))
        y_va = load_masks(os.path.join(test_img_dir, 'masks', '0'))

        if train_dir is not None and self.tr_mean is None:
            x_tr = load_images(train_dir)
            self.normalize(x_tr)

        x_va_norm = self.normalize(x_va)
        y_va = to_categorical(y_va, self.n_class)

        self.model.compile(optimizer=Adam(lr=0.001), loss=f1_loss, metrics=['acc', 'categorical_crossentropy'])
        self.model.load_weights(os.path.join(model_dir, 'model.h5'))

        # model = load_model(os.path.join(model_dir, 'model.h5'))
        p_va = self.model.predict(x_va_norm, batch_size=batch_size, verbose=1)

        scores = self.model.evaluate(x_va_norm, y_va, verbose=1)
        store_prediction(p_va, x_va, output_dir)

        print('DICE:      ' + str(f1_np(y_va, p_va)))
        print('IoU:       ' + str(iou_np(y_va, p_va)))
        print('Precision: ' + str(precision_np(y_va, p_va)))
        print('Recall:    ' + str(recall_np(y_va, p_va)))
        print('Error:     ' + str(error_np(y_va, p_va)))
        print('Scores:    ', scores)

    def predict(self, model_dir, img_dir, output_dir, batch_size=4, train_dir=None):
        x_va = load_images(os.path.join(img_dir), sort=True)
        # self.tr_mean = np.array([76.51, 75.41, 71.02])
        # self.tr_std = np.array([76.064, 75.23, 75.03])

        if train_dir is not None and self.tr_mean is None:
            x_tr = load_images(os.path.join(train_dir))
            self.normalize(x_tr)

        # pre-process
        if self.tr_mean is not None:
            x_va_norm = self.normalize(x_va)

        # model = load_model(os.path.join(model_dir, 'model.h5'))

        self.model.compile(optimizer=Adam(lr=0.001), loss=f1_loss, metrics=['acc', 'categorical_crossentropy'])
        self.model.load_weights(os.path.join(model_dir, 'model.h5'))

        p_va = self.model.predict(x_va_norm, batch_size=batch_size, verbose=1)
        store_prediction(p_va, x_va, output_dir)


if __name__ == '__main__':
    # for apple
    # file_base = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/03_ImageSegmentation/structure_vidFloodExt/'

    # for windows
    file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\'

    model_name = 'corr_c2l4b4e15f32_noaug_f1'
    model_dir = os.path.join(file_base, 'models', model_name)

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    train_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\train'
    valid_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\validate'
    test_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\test'

    pred_dir_flood = os.path.join(file_base, 'models', model_name, 'test_img')

    test_dir_elliot = os.path.join(file_base, 'frames', 'elliotCityFlood')
    test_dir_athletic = os.path.join(file_base, 'frames', 'ChaskaAthleticPark')
    pred_dir = os.path.join(file_base, 'predictions', model_name)

    if not os.path.isdir(pred_dir):
        os.mkdir(pred_dir)

    pred_dir_elliot = os.path.join(file_base, 'predictions', model_name, 'elliotCityFlood')
    pred_dir_athletic = os.path.join(file_base, 'predictions', model_name, 'ChaskaAthleticPark')

    if not os.path.isdir(pred_dir_flood):
        os.mkdir(pred_dir_flood)

    if not os.path.isdir(pred_dir_athletic):
        os.mkdir(pred_dir_athletic)

    img_shape = (512, 512, 3)
    unet = UNet(img_shape, root_features=64, layers=4, batch_norm=True)

    unet.train(model_dir, train_dir_flood, valid_dir_flood, batch_size=4, epochs=15, augmentation=True)
    unet.test(model_dir, test_dir_flood, pred_dir_flood)
    unet.predict(model_dir, test_dir_athletic, pred_dir_athletic, batch_size=3, train_dir=os.path.join(train_dir_flood, 'images', '0'))

