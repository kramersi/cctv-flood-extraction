import os
import glob
import re
import numpy as np
import cv2
import pandas as pd

from keras.models import Input, Model, load_model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
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


def load_images(path, sort=False, target_size=None):
    files = glob.glob(os.path.join(path, '*'))
    if sort is True:
        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    first_img = load_img(files[0])

    n = len(files)
    if target_size is not None:
        h = target_size[0]
        w = target_size[1]
    else:
        w = first_img.width
        h = first_img.height
    x = np.empty((n, h, w, 3))

    for i, f in enumerate(files):
        im = load_img(f, target_size=target_size)
        x[i, :, :, :] = img_to_array(im).astype(np.float32)

    return x.astype('float32')


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
        self.residual = residual

        self.tr_mean = None
        self.tr_std = None

        # define model
        i = Input(shape=img_shape)
        o = level_block(i, root_features, layers, inc_rate, activation, dropout, batch_norm, max_pool, up_conv, residual)
        o = Conv2D(n_class, 1, activation='sigmoid')(o)
        self.model = Model(inputs=i, outputs=o)

    def normalize(self, x):
        self.tr_mean = np.array([69.7399, 69.8885, 65.1602])
        self.tr_std = np.array([72.9841, 72.3374, 71.6508])

        if self.tr_mean is None:
            print('mean and standard deviation of training pictures not calculated yet, calculating...')
            self.tr_mean, self.tr_std = channel_mean_stdev(x)
            print('mean: ', self.tr_mean, 'std: ', self.tr_std)

        x_norm = (x - self.tr_mean.astype('float32')) / self.tr_std.astype('float32')
        # x_norm = (x - np.amin(x)) / np.amax(x)
        # img_eq = exposure.equalize_hist(x_norm)
        return x_norm

    def train(self, model_dir, train_dir, valid_dir, epochs=20, batch_size=4, augmentation=True, normalisation=True, base_dir=None, learning_rate=0.01):
        """ trains a unet instance on keras. With on-line data augmentation to diversify training samples in each batch.

            example of defining paths
            train_dir = "E:\\watson_for_trend\\3_select_for_labelling\\train_cityscape\\"
            model_dir = "E:\\watson_for_trend\\5_train\\cityscape_l5f64c3n8e20\\"

        """
        seed = 1234  # Provide the same seed and keyword arguments to the fit and flow methods

        x_tr = load_images(os.path.join(train_dir, 'images'))  # load training pictures in numpy array
        shape = x_tr.shape  # pic_nr x width x height x depth
        n_train = shape[0]  # len(image_generator)

        # define callbacks
        mc = ModelCheckpoint(os.path.join(model_dir, 'model.h5'), save_best_only=True, save_weights_only=False)
        es = EarlyStopping(monitor='val_loss', patience=30)
        tb = TensorBoard(log_dir=model_dir, write_graph=True, write_images=True)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_lr=0.000001)

        if base_dir is not None:
            self.model.load_weights(os.path.join(base_dir, 'model.h5'))

            for layer in self.model.layers[:]:
                layer.trainable = False

            # # Check the trainable status of the individual layers
            # for layer in self.model.layers:
            #     print(layer, layer.trainable)

        # compile model with optimizer and loss function
        self.model.compile(optimizer=Adam(lr=learning_rate), loss=f1_loss,
                           metrics=['acc', 'categorical_crossentropy'])

        # summary of parameters in each layer
        self.model.summary()

        y_tr = load_masks(os.path.join(train_dir, 'masks'))  # load mask arrays
        x_va = load_images(os.path.join(valid_dir, 'images'))
        y_va = load_masks(os.path.join(valid_dir, 'masks'))
        n_valid = x_va.shape[0]

        # data normalisation
        if normalisation is True:
            x_tr = self.normalize(x_tr)
            x_va = self.normalize(x_va)

        # create one-hot
        y_tr = to_categorical(y_tr, self.n_class)
        y_va = to_categorical(y_va, self.n_class)

        if augmentation:
            data_gen_args = dict(featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=0,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=(0.5, 1),
                                 horizontal_flip=True,
                                 img_aug=False,
                                 fill_mode='reflect')

            # use affinity transform for masks
            mask_datagen = ImageDataGenerator(**data_gen_args)

            # add picture worsening on images not on masks
            data_gen_args['img_aug'] = True
            data_gen_args['blur_range'] = (0.0, 1.2)
            # data_gen_args['contrast_range'] = (0.9, 1.1)
            # data_gen_args['grayscale_range'] = (0.0, 0.1)

            image_datagen = ImageDataGenerator(**data_gen_args)

            ## fit the augmentation model to the images and masks with the same seed
            image_datagen.fit(x_tr, augment=True, seed=seed)
            mask_datagen.fit(y_tr, augment=True, seed=seed)
            # create image generator for online data augmentation
            aug_path = os.path.join(model_dir, 'augmentations')
            image_generator = image_datagen.flow(
                x_tr,
                batch_size=batch_size,
                shuffle=True,
                seed=seed)
                # save_to_dir=aug_path)
            ## set the parameters for the data to come from (masks)
            mask_generator = mask_datagen.flow(
                y_tr,
                batch_size=batch_size,
                shuffle=True,
                seed=seed)

            # combine generators into one which yields image and masks
            train_generator = zip(image_generator, mask_generator)
            #train_generator = image_datagen.flow(x_tr, y_tr, batch_size=batch_size, shuffle=True, seed=seed, save_to_dir=aug_path)
            valid_generator = (x_va, y_va)

            # train unet with image_generator
            self.model.fit_generator(train_generator,
                                     validation_data=valid_generator,
                                     steps_per_epoch=n_train / batch_size,
                                     validation_steps=n_valid / batch_size,
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=[mc, tb, es, lr],
                                     use_multiprocessing=False,
                                     workers=4)
        else:
            self.model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs, batch_size=batch_size,
                           shuffle=True, callbacks=[mc, es, tb, lr])

        scores = self.model.evaluate(x_va, y_va, batch_size=batch_size, verbose=1)
        print('scores', scores)

    def test(self, model_dir, test_img_dir, output_dir, batch_size=4, train_dir=None):

        x_va = load_images(os.path.join(test_img_dir, 'images'))
        y_va = load_masks(os.path.join(test_img_dir, 'masks'))
        self.tr_mean = np.array([69.739, 69.888, 65.160])
        self.tr_std = np.array([72.98415532, 72.33742881, 71.6508131])

        if train_dir is not None and self.tr_mean is None:
            x_tr = load_images(train_dir)
            self.normalize(x_tr)

        x_va_norm = self.normalize(x_va)
        y_va = to_categorical(y_va, self.n_class)

        self.model.compile(optimizer=Adam(lr=0.001), loss=f1_loss, metrics=['acc', 'categorical_crossentropy'])
        self.model.load_weights(os.path.join(model_dir, 'model.h5'))

        # model = load_model(os.path.join(model_dir, 'model.h5'))
        p_va = self.model.predict(x_va_norm, batch_size=batch_size, verbose=1)

        scores = self.model.evaluate(x_va_norm, y_va, batch_size=batch_size, verbose=1)
        store_prediction(p_va, x_va, output_dir)
        res = {'DICE': [f1_np(y_va, p_va)], 'IoU': [iou_np(y_va, p_va)], 'Precision': [precision_np(y_va, p_va)],
               'Recall': [recall_np(y_va, p_va)], 'Error': [error_np(y_va, p_va)]}

        pd.DataFrame(res).to_csv(os.path.join(model_dir, 'result.csv'))

        print('DICE:      ' + str(f1_np(y_va, p_va)))
        print('IoU:       ' + str(iou_np(y_va, p_va)))
        print('Precision: ' + str(precision_np(y_va, p_va)))
        print('Recall:    ' + str(recall_np(y_va, p_va)))
        print('Error:     ' + str(error_np(y_va, p_va)))
        print('Scores:    ', scores)

    def fine_tune(self, model_dir, img_dir, valid_dir):

        self.model.compile(optimizer=Adam(lr=0.001), loss=f1_loss, metrics=['acc', 'categorical_crossentropy'])
        self.model.load_weights(os.path.join(model_dir, 'model.h5'))

        for layer in self.model.layers[:-4]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in self.model.layers:
            print(layer, layer.trainable)

        self.train(model_dir, img_dir, valid_dir, batch_size=2, epochs=20, augmentation=False)


    def predict(self, model_dir, img_dir, output_dir, batch_size=4, train_dir=None):
        x_va = load_images(os.path.join(img_dir), sort=True, target_size=(512, 512))
        self.tr_mean = np.array([69.739934, 69.88847943, 65.16021837])
        self.tr_std = np.array([72.98415532, 72.33742881, 71.6508131])

        if train_dir is not None and self.tr_mean is None:
            x_tr = load_images(os.path.join(train_dir), sort=True, target=(512, 512))
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
    model_names = ['ft_l5b3e200f16_dr075i2res_lr', 'ft_l4b3e60f64_dr075caugi2', 'cflood_c2l3b4e60f32_dr075caugi2_ext']
    aug = [True, False, True]
    feat = [16, 64, 32]
    ep = [200, 60, 60]
    lay = [5, 4, 3]
    drop = [0.75, 0.75, 0.75]
    bat = [3, 3, 4]
    res = [True, False, False]

    for i, model_name in enumerate(model_names):
        if i == 0:
            model_dir = os.path.join(file_base, 'models', model_name)

            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

            # configs for fine tune
            #base_model_dir = os.path.join(file_base, 'models', 'cflood_c2l5b3e40f16_dr075caugi2res')
            # base_model_dir = os.path.join(file_base, 'models', 'cflood_c2l4b3e60f64_dr075caugi2')
            #
            # train_dir_ft = os.path.join(file_base, 'video_masks', 'CombParkGarage_train')
            # valid_dir_ft = os.path.join(file_base, 'video_masks', 'CombParkGarage_validate')
            # pred_dir_ft = os.path.join(file_base, 'models', model_name, 'test_img_tf')
            # if not os.path.isdir(pred_dir_ft):
            #     os.mkdir(pred_dir_ft)

            # # configs for training from scratch
            train_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class\\train'
            valid_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class\\validate'
            test_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class\\test'

            pred_dir_flood = os.path.join(file_base, 'models', model_name, 'test_img_ext')
            if not os.path.isdir(pred_dir_flood):
                os.mkdir(pred_dir_flood)

            # configs for testing model
            test_dir_elliot = os.path.join(file_base, 'frames', 'elliotCityFlood')
            test_dir_athletic = os.path.join(file_base, 'frames', 'ChaskaAthleticPark')
            test_dir_floodx = os.path.join(file_base, 'frames', 'FloodX')

            pred_dir = os.path.join(file_base, 'predictions', model_name)
            if not os.path.isdir(pred_dir):
                os.mkdir(pred_dir)

            test_dir = os.path.join(file_base, 'frames', '*')

            # pred_dir_elliot = os.path.join(file_base, 'predictions', model_name, 'elliotCityFlood')
            # pred_dir_athletic = os.path.join(file_base, 'predictions', model_name, 'ChaskaAthleticPark')
            # pred_dir_floodx = os.path.join(file_base, 'predictions', model_name, 'FloodX')

            # test_dirs = [test_dir_elliot, test_dir_athletic, test_dir_floodx]
            # pred_dirs = [pred_dir_elliot, pred_dir_athletic, pred_dir_floodx]

            # for pred_dir in pred_dirs:
            #     if not os.path.isdir(pred_dir):
            #         os.mkdir(pred_dir)

            img_shape = (512, 512, 3)
            unet = UNet(img_shape, root_features=feat[i], layers=lay[i], batch_norm=True, dropout=drop[i], inc_rate=2., residual=res[i])
            # unet.model.summary()

            unet.train(model_dir, train_dir_flood, valid_dir_flood, batch_size=bat[i], epochs=ep[i], augmentation=aug[i], base_dir=None, learning_rate=0.1)
            # unet.test(model_dir, test_dir_flood, pred_dir_flood, batch_size=4)

            # for test in glob.glob(test_dir):  # test for all frames in directory
            #     base, tail = os.path.split(test)
            #     pred = os.path.join(pred_dir, tail)
            #
            #     if not os.path.isdir(pred):
            #         os.mkdir(pred)
            #
            #     unet.predict(model_dir, test, pred, batch_size=3, train_dir=os.path.join(train_dir_flood, 'images'))

