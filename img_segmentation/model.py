import os
import numpy as np
import pandas as pd

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization
from keras.models import Input, Model
from keras.optimizers import Adam

from img_segmentation.image_gen import ImageGenerator
from img_segmentation.utils import f1_loss, f1_np, iou_np, precision_np, recall_np, error_np, load_images, channel_mean_stdev, \
    store_prediction, load_img_msk_paths


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
        #self.tr_mean = np.array([69.7399, 69.8885, 65.1602])
        #self.tr_std = np.array([72.9841, 72.3374, 71.6508])

        if self.tr_mean is None:
            print('mean and standard deviation of training pictures not calculated yet, calculating...')
            self.tr_mean, self.tr_std = channel_mean_stdev(x)
            print('mean: ', self.tr_mean, 'std: ', self.tr_std)

        x_norm = (x - self.tr_mean.astype('float32')) / self.tr_std.astype('float32')
        # x_norm = (x - np.amin(x)) / np.amax(x)
        # img_eq = exposure.equalize_hist(x_norm)
        return x_norm

    def train(self, model_dir, train_dir, valid_dir, epochs=20, batch_size=3, augmentation=True, normalisation=True, base_dir=None, trainable_index=14, save_aug=False, learning_rate=0.01):
        """ trains a unet instance on keras. With on-line data augmentation to diversify training samples in each batch.

            example of defining paths
            train_dir = "E:\\watson_for_trend\\3_select_for_labelling\\train_cityscape\\"
            model_dir = "E:\\watson_for_trend\\5_train\\cityscape_l5f64c3n8e20\\"

        """
        # define callbacks
        mc = ModelCheckpoint(os.path.join(model_dir, 'model.h5'), save_best_only=True, save_weights_only=False)
        es = EarlyStopping(monitor='val_loss', patience=30)
        tb = TensorBoard(log_dir=model_dir, write_graph=True)  # write_images=True, write_grads=True, histogram_freq=5
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, verbose=1, min_lr=0.0000001)

        # define weights (not used now, keras does not support it with segmentation)
        class_weights = {0: 0.5, 1: 0.5}

        if base_dir is not None:
            self.model.load_weights(os.path.join(base_dir, 'model.h5'))

            for layer in self.model.layers[:-trainable_index]:
                layer.trainable = False

            # Check the trainable status of the individual layers
            for layer in self.model.layers:
                print(layer.name, layer.trainable)

        # compile model with optimizer and loss function
        self.model.compile(optimizer=Adam(lr=learning_rate), loss=f1_loss,
                           metrics=['acc', 'categorical_crossentropy'])

        # summary of parameters in each layer
        self.model.summary()

        path_tr = load_img_msk_paths(train_dir)
        path_va = load_img_msk_paths(valid_dir)

        if save_aug is True:
            aug_path = os.path.join(model_dir, 'augmentations')
            if not os.path.exists(aug_path):
                print('created augmentation dir', aug_path)
                os.makedirs(aug_path)
        else:
            aug_path = None

        # augmentation are defined here and can be changed
        aug_dict = dict(horizontal_flip=0.5, vertical_flip=0.0, rotation_range=(0.0, 0.0),
                        width_shift_range=(-0.2, 0.2), height_shift_range=(-0.2, 0.2), contrast_range=(0.5, 1.5),
                        zoom_range=(1.0, 1.33), grayscale_range=(0.0, 0.8), brightness_range=(-80, 20),
                        crop_range=(0, 0), blur_range=(0.0, 1.0), shear_range=(0.0, 0.0), prob=0.2)

        train_generator = ImageGenerator(list(path_tr.keys()), masks=path_tr, batch_size=batch_size, dim=(512, 512), shuffle=True,
                                         normalize='std_norm', save_to_dir=aug_path, augmentation=augmentation, aug_dict=aug_dict)

        valid_generator = ImageGenerator(list(path_va.keys()), masks=path_va, batch_size=batch_size, dim=(512, 512), shuffle=True,
                                         normalize='std_norm', augmentation=augmentation, aug_dict=aug_dict)

        # train unet with image_generator
        self.model.fit_generator(train_generator,
                                 validation_data=valid_generator,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[mc, tb, es, lr],
                                 use_multiprocessing=False,
                                 workers=4)

        print('Training completed')

    def test(self, model_dir, test_img_dir, output_dir, csv_path=None):
        path_test = load_img_msk_paths([test_img_dir])

        img_gen_norm = ImageGenerator(list(path_test.keys()), masks=path_test, batch_size=1, shuffle=False, normalize='std_norm', augmentation=False)
        img_gen = ImageGenerator(list(path_test.keys()), masks=path_test, batch_size=1, shuffle=False, normalize=None, augmentation=False)

        n = len(img_gen)
        x_va = np.empty((n, 512, 512, 3))
        y_va = np.empty((n, 512, 512, 2))
        for i in range(n):
            x_va[i, ], y_va[i,] = img_gen[i]

        self.model.compile(optimizer=Adam(lr=0.001), loss=f1_loss, metrics=['acc', 'categorical_crossentropy'])
        self.model.load_weights(os.path.join(model_dir, 'model.h5'))

        p_va = self.model.predict_generator(generator=img_gen_norm, verbose=1)
        scores = self.model.evaluate_generator(img_gen_norm, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)

        store_prediction(p_va, x_va, output_dir)

        res = {'DICE': [f1_np(y_va, p_va)], 'IoU': [iou_np(y_va, p_va)], 'Precision': [precision_np(y_va, p_va)],
               'Recall': [recall_np(y_va, p_va)], 'Error': [error_np(y_va, p_va)]}

        if csv_path is None:
            pd.DataFrame(res).to_csv(os.path.join(model_dir, 'result.csv'))
        else:
            pd.DataFrame(res).to_csv(os.path.join(csv_path))

        print('DICE:      ' + str(f1_np(y_va, p_va)))
        print('IoU:       ' + str(iou_np(y_va, p_va)))
        print('Precision: ' + str(precision_np(y_va, p_va)))
        print('Recall:    ' + str(recall_np(y_va, p_va)))
        print('Error:     ' + str(error_np(y_va, p_va)))
        print('Scores:    ', scores)


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

        self.model.compile(optimizer=Adam(lr=0.001), loss=f1_loss, metrics=['acc', 'categorical_crossentropy'])
        self.model.load_weights(os.path.join(model_dir, 'model.h5'))

        p_va = self.model.predict(x_va_norm, batch_size=batch_size, verbose=1)
        store_prediction(p_va, x_va, output_dir)
