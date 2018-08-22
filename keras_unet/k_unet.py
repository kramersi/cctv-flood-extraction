import os
import glob
import numpy as np
import pandas as pd

from keras.models import Input, Model
from keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose, UpSampling2D, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

from keras_utils import f1_loss, f1_np, iou_np, precision_np, recall_np, error_np, load_masks, load_images, channel_mean_stdev, store_prediction, load_img_msk_paths
from image_generator import ImageGenerator

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

    def train(self, model_dir, train_dir, valid_dir, epochs=20, batch_size=3, augmentation=True, normalisation=True, base_dir=None, save_aug=False, learning_rate=0.01):
        """ trains a unet instance on keras. With on-line data augmentation to diversify training samples in each batch.

            example of defining paths
            train_dir = "E:\\watson_for_trend\\3_select_for_labelling\\train_cityscape\\"
            model_dir = "E:\\watson_for_trend\\5_train\\cityscape_l5f64c3n8e20\\"

        """
        # seed = 1234  # Provide the same seed and keyword arguments to the fit and flow methods

        # x_tr = load_images(os.path.join(train_dir, 'images'))  # load training pictures in numpy array
        # shape = x_tr.shape  # pic_nr x width x height x depth
        # n_train = shape[0]  # len(image_generator)

        # define callbacks
        mc = ModelCheckpoint(os.path.join(model_dir, 'model.h5'), save_best_only=True, save_weights_only=False)
        es = EarlyStopping(monitor='val_loss', patience=30)
        tb = TensorBoard(log_dir=model_dir, write_graph=True)  # write_images=True, write_grads=True, histogram_freq=5
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, verbose=1, min_lr=0.0000001)

        # define weights
        class_weights = {0: 0.5, 1: 0.5}

        if base_dir is not None:
            self.model.load_weights(os.path.join(base_dir, 'model.h5'))

            # for layer in self.model.layers[:-14]:
            #     layer.trainable = False

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


        # aug_dict = dict(horizontal_flip=0.5, vertical_flip=0.0, rotation_range=(0, 0),
        #             width_shift_range=(-0.2, 0.2), height_shift_range=(-0.2, 0.2), contrast_range=(0.5, 1.5),
        #             zoom_range=(1, 1.2), grayscale_range=(0.0, 1.0), brightness_range=(0.05, 1.25), crop_range=(0, 0),
        #             blur_range=(0.0, 1.0), shear_range=(0, 0), prob=0.25)

        aug_dict1 = dict(horizontal_flip=0.5, vertical_flip=0.0, rotation_range=(0.0, 0.0),
                        width_shift_range=(-0.2, 0.2), height_shift_range=(-0.2, 0.2), contrast_range=(0.5, 1.5),
                        zoom_range=(1.0, 1.33), grayscale_range=(0.0, 0.8), brightness_range=(-80, 20),
                        crop_range=(0, 0), blur_range=(0.0, 1.0), shear_range=(0.0, 0.0), prob=0.2)

        # aug_dict = dict(horizontal_flip=0.5, vertical_flip=0.0, rotation_range=(0, 0),
        #                 width_shift_range=(-0.2, 0.2), height_shift_range=(-0.2, 0.2), contrast_range=1.0,
        #                 zoom_range=(1, 1), grayscale_range=(0.0, 0.0), brightness_range=(1.0, 1.0), crop_range=(0, 0),
        #                 blur_range=0, shear_range=(0, 0), prob=0.25)

        train_generator = ImageGenerator(list(path_tr.keys()), masks=path_tr, batch_size=batch_size, dim=(512, 512), shuffle=True,
                                         normalize='std_norm', save_to_dir=aug_path, augmentation=augmentation, aug_dict=aug_dict1)

        valid_generator = ImageGenerator(list(path_va.keys()), masks=path_va, batch_size=batch_size, dim=(512, 512), shuffle=True,
                                         normalize='std_norm', augmentation=True, aug_dict=aug_dict1)

        # y_tr = load_masks(os.path.join(train_dir, 'masks'))  # load mask arrays
        # x_va = load_images(os.path.join(valid_dir, 'images'))
        # y_va = load_masks(os.path.join(valid_dir, 'masks'))
        # n_valid = x_va.shape[0]

        # # data normalisation
        # if normalisation is True:
        #     x_tr = self.normalize(x_tr)
        #     x_va = self.normalize(x_va)

        # # create one-hot
        # y_tr = to_categorical(y_tr, self.n_class)
        # y_va = to_categorical(y_va, self.n_class)

        # if augmentation:
        #     data_gen_args = dict(featurewise_center=False,
        #                          featurewise_std_normalization=False,
        #                          rotation_range=0,
        #                          width_shift_range=0.2,
        #                          height_shift_range=0.2,
        #                          zoom_range=(0.5, 1),
        #                          horizontal_flip=True,
        #                          img_aug=False,
        #                          fill_mode='reflect')
        #
        #     # use affinity transform for masks
        #     mask_datagen = ImageDataGenerator(**data_gen_args)
        #
        #     # add picture worsening on images not on masks
        #     data_gen_args['img_aug'] = True
        #     data_gen_args['blur_range'] = (0.0, 1.2)
        #     # data_gen_args['contrast_range'] = (0.9, 1.1)
        #     # data_gen_args['grayscale_range'] = (0.0, 0.1)
        #
        #     image_datagen = ImageDataGenerator(**data_gen_args)
        #
        #     ## fit the augmentation model to the images and masks with the same seed
        #     image_datagen.fit(x_tr, augment=True, seed=seed)
        #     mask_datagen.fit(y_tr, augment=True, seed=seed)
        #     # create image generator for online data augmentation
        #     aug_path = os.path.join(model_dir, 'augmentations')
        #     image_generator = image_datagen.flow(
        #         x_tr,
        #         batch_size=batch_size,
        #         shuffle=True,
        #         seed=seed)
        #         # save_to_dir=aug_path)
        #     ## set the parameters for the data to come from (masks)
        #     mask_generator = mask_datagen.flow(
        #         y_tr,
        #         batch_size=batch_size,
        #         shuffle=True,
        #         seed=seed)
        #
        #     # combine generators into one which yields image and masks
        #     train_generator = zip(image_generator, mask_generator)
        #     #train_generator = image_datagen.flow(x_tr, y_tr, batch_size=batch_size, shuffle=True, seed=seed, save_to_dir=aug_path)
        #     valid_generator = (x_va, y_va)

        # train unet with image_generator
        self.model.fit_generator(train_generator,
                                 validation_data=valid_generator,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[mc, tb, es, lr],
                                 use_multiprocessing=False,
                                 workers=4)
        # else:
        #     self.model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=epochs, batch_size=batch_size,
        #                    shuffle=True, callbacks=[mc, tb, lr])

        #scores = self.model.evaluate_generator(valid_generator, workers=4, verbose=0)
        #print('scores', scores)

    def test_gen(self, model_dir, test_img_dir, output_dir, batch_size=4, train_dir=None, csv_path=None):
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

    def fine_tune(self, model_dir, img_dir, valid_dir):

        self.model.compile(optimizer=Adam(lr=0.001), loss=f1_loss, metrics=['acc', 'categorical_crossentropy'])
        self.model.load_weights(os.path.join(model_dir, 'model.h5'))

        for layer in self.model.layers[:-15]:
            layer.trainable = False

        # Check the trainable status of the individual layers
        for layer in self.model.layers:
            print(layer, layer.trainable)

        self.train(model_dir, img_dir, valid_dir, batch_size=2, epochs=20, augmentation=False)

    def predict_gen(self, model_dir, img_dir, output_dir, batch_size=4, train_dir=None):

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
    tune_vid = ''
    file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\'
    model_names = ['train_test_l5_refaug', 'train_test_l3f64aug', 'gentrain_l2f128aug', 'gentrain_l4f32aug_res', 'gentrain_l5f16aug', 'gentrain_l6f8aug_res']
    aug = [True, False, True, True, True, True]
    feat = [16, 32, 128, 32, 16, 8]
    ep = [250, 200, 200, 200, 200, 200]
    lay = [5, 3, 2, 4, 5, 6]
    drop = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
    bat = [8, 8, 2, 4, 8, 6]
    res = [True, False, False, True, False, True]
    bd = [None, None, None, None, None, None]  # os.path.join(file_base, 'models', 'train_test_l5_' + tune_vid + 'Top')
    # bd = [os.path.join(file_base, 'models', 'ft_l5b3e200f16_dr075i2res_lr'), None, None]

    for i, model_name in enumerate(model_names):
        if i in [3, 5]:
            model_dir = os.path.join(file_base, 'models', model_name)

            if not os.path.isdir(model_dir):
                os.mkdir(model_dir)

            # configs for fine tune
            # base_model_dir = os.path.join(file_base, 'models', 'cflood_c2l5b3e40f16_dr075caugi2res')
            # base_model_dir = os.path.join(file_base, 'models', 'cflood_c2l4b3e60f64_dr075caugi2')
            #
            # train_dir_ft = os.path.join(file_base, 'video_masks', 'CombParkGarage_train')
            # valid_dir_ft = os.path.join(file_base, 'video_masks', 'CombParkGarage_validate')
            # pred_dir_ft = os.path.join(file_base, 'models', model_name, 'test_img_tf')
            # if not os.path.isdir(pred_dir_ft):
            #     os.mkdir(pred_dir_ft)

            # # configs for training from scratch
            train_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class\\train' # os.path.join(file_base, 'video_masks', 'floodX_cam1', 'train')
            valid_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class\\validate'  #os.path.join(file_base, 'video_masks', 'floodX_cam1', 'validate')
            test_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class\\test'  #os.path.join(file_base, 'video_masks', 'floodX_cam1', 'validate')

            # paths for finetune
            train_dir_further = os.path.join(file_base, 'other_video_masks', 'FurtherYoutube', 'train')
            valid_dir_further = os.path.join(file_base, 'other_video_masks', 'FurtherYoutube', 'validate')

            train_tune_dir = os.path.join(file_base, 'video_masks', tune_vid, 'train')
            valid_tune_dir = os.path.join(file_base, 'video_masks', tune_vid, 'validate')

            pred_dir_flood = os.path.join(file_base, 'models', model_name, 'test_img')
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

            #unet.train(model_dir, [train_tune_dir], [valid_tune_dir], batch_size=bat[i], epochs=ep[i], augmentation=aug[i], base_dir=bd[i], save_aug=True, learning_rate=0.001)
            unet.train(model_dir, [train_dir_flood, train_dir_further], [valid_dir_flood, valid_dir_further], batch_size=bat[i], epochs=ep[i], augmentation=aug[i], base_dir=bd[i], save_aug=False, learning_rate=0.001)
            #unet.test_gen(model_dir, test_dir_flood, pred_dir_flood, batch_size=3)
            # test_dir = os.path.join(file_base, 'video_masks', '*')
            # #
            # for test in glob.glob(test_dir):  # test for all frames in directory
            #     base, tail = os.path.split(test)
            #     pred = os.path.join(model_dir, 'pred_' + tail)
            #     model_dir = os.path.join(file_base, 'models', model_name)  #  + tail
            #     csv_path = os.path.join(model_dir, tail + '.csv')
            #     test_val = os.path.join(test, 'validate')
            #     if not os.path.isdir(pred):
            #         os.mkdir(pred)
            #
            #     unet.test_gen(model_dir, test_val, pred, batch_size=3, csv_path=csv_path)

            # script for storing prediction
            # from keras_utils import overlay_img_mask
            # vid_name = 'FloodXCam1'
            # img_path = os.path.join(file_base, 'video_masks', vid_name, 'validate', 'images')
            # msk_path = os.path.join(file_base, 'video_masks', vid_name, 'validate', 'masks')
            # output = os.path.join(file_base, 'video_masks', vid_name, 'validate', 'human_masks')
            # if not os.path.isdir(output):
            #     os.mkdir(output)
            # im = load_images(img_path)
            # msk = load_masks(msk_path)
            # for nr, (i, m) in enumerate(zip(im, msk)):
            #     name = 'human' + str(nr) + '.png'
            #     overlay_img_mask(m, i, os.path.join(output, name))