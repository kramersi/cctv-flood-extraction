import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from imgaug import augmenters as iaa
import glob
import shutil
from tf_unet import unet, util, image_util
from keras_unet.utils import *
from keras import utils as keras_utils


class CCTVFloodExtraction(object):

    cr_win = dict(top=0, left=0, width=640, height=360)

    def __init__(self, video_file, model_dir, frame_dir=None, pred_dir=None, signal_dir=None, crop_window=cr_win):

        self.video_file = video_file
        self.model_dir = model_dir

        self.model_name = os.path.splitext(os.path.basename(self.model_dir))[0]
        self.video_name = os.path.splitext(os.path.basename(self.video_file))[0]

        self.frame_dir = self.check_create_folder(frame_dir, 'frames', self.video_name)
        self.pred_dir = self.check_create_folder(pred_dir, 'predictions', self.video_name)
        self.signal_dir = self.check_create_folder(signal_dir, 'signal')

        self.predictions = []
        self.crop_window = crop_window

    def check_create_folder(self, output, *folder_names):
        path = self.video_file

        # if none then create diectory on same level as video directory with the folder_name and video name
        if output is None:
            output = os.path.abspath(os.path.join(os.path.dirname(path), os.pardir, *folder_names))
        else:
            output = os.path.join(output, self.video_name)

        # if directory not existing create directory
        if not os.path.exists(output):
            print('created new directory: ', output)
            os.makedirs(output)

        return output

    def import_video(self, url, source='youtube'):
        # video must be of following formats: mp4 | flv | ogg | webm | mkv | avi
        if source == 'youtube':
            from pytube import YouTube
            yt = YouTube(url)
            fp = yt.streams.first().download(os.path.dirname(self.video_file))

            if fp is not None:
                self.video_file = fp
                print('video imported and stored in ', fp)

    def video2frame(self, skip=1, resize_dims=None, mirror=False, keep_aspect=True, max_frames=10, rotate=0):
        """ extract frames out of a video

        Args:
            skip (int): how many frames to skip
            resize_dims (tuple): tuple of pixel width and height. If None then original is kept
            mirror (bool): if frames should be mirrored
            max_frames (int): how much frames in maximumn
            rotate (int): how many degree frames should be rotated. One of 0, 90, 180 or 270

        """

        if len(os.listdir(self.frame_dir)) > 0:
            print('Picture from this movie already extracted in that directory.')
        else:
            video_object = cv2.VideoCapture(self.video_file)  # make video object

            index = 0
            last_mirrored = True

            frame_count = video_object.get(cv2.CAP_PROP_FRAME_COUNT)

            skip_delta = 0
            if max_frames and frame_count > max_frames:
                skip_delta = frame_count / max_frames

            while True:
                success, frame = video_object.read()  # extract frames
                if success:
                    if index % skip == 0:

                        # resize frames
                        if resize_dims is not None:
                            if keep_aspect is True:
                                frame = util.resize_keep_aspect(frame, resize_dims)
                            else:
                                frame = cv2.resize(frame, resize_dims, interpolation=cv2.INTER_CUBIC)

                        # mirror frames
                        if mirror and last_mirrored:
                            frame = np.fliplr(frame)
                        last_mirrored = not last_mirrored

                        # Rotate if needed:
                        if rotate > 0:
                            if rotate == 90:
                                frame = cv2.transpose(frame)
                                frame = cv2.flip(frame, 1)
                            elif rotate == 180:
                                frame = cv2.flip(frame, -1)
                            elif rotate == 270:
                                frame = cv2.transpose(frame)
                                frame = cv2.flip(frame, 0)

                        # write images to output file
                        frame_fp = os.path.join(self.frame_dir, 'frame_' + str(index) + '.png')
                        cv2.imwrite(frame_fp, frame)
                else:
                    break

                index += int(1 + skip_delta)
                video_object.set(cv2.CAP_PROP_POS_FRAMES, index)

            print('frame extracted from video')

    def load_model(self, model_type='tensorflow'):
        """ loads neural network model of different types.

        The pretrained model must have following specifications:
            Input: RGB-Picture as an array of width*height*channel
            Classes: Has to have two classes which are labelled 0=background and 1=water
            Output: array of width*height*channel*class_number

        Args:
            model_type (str): type of model, which should be loaded, has to be one of tensorflow, keras, caffee etc.
            n_class (int): number of classes (defaults to two classes: background and water

        """
        def load_image(path, dtype=np.float32):
            data = np.array(cv2.imread(path), dtype)

            # normalization
            data -= np.amin(data)
            data /= np.amax(data)
            return data

        if os.path.isdir(self.pred_dir):
            print('Pictures already predicted with that model')
        else:
            print('created new directory ', self.pred_dir)
            os.makedirs(self.pred_dir)

            if model_type == 'tensorflow':
                import tensorflow as tf

                tensor_node_op = 'ArgMax:0'  # 'div_1:0'  # 'truediv_21:0'
                tensor_node_x = 'Placeholder:0'
                tensor_node_prob = 'Placeholder_2:0'
                meta_file = '/model.cpkt.meta'

                with tf.Session() as sess:

                    writer = tf.summary.FileWriter("tensorflow")  # setup writer object for tensorboard
                    saver = tf.train.import_meta_graph(self.model_dir + meta_file)
                    saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
                    print("Model restored from file: %s" % self.model_dir)

                    # get the graph in the current thread
                    graph = tf.get_default_graph()
                    # node_names = [tensor.name for tensor in graph.as_graph_def().node]
                    # print(node_names)

                    # access the input key words for feed_dict
                    xk = graph.get_tensor_by_name(tensor_node_x)
                    keep_prob = graph.get_tensor_by_name(tensor_node_prob)

                    # Now, access the operation that you want to run.
                    restored_op = graph.get_tensor_by_name(tensor_node_op)

                    # loop through files and save predictions
                    for image in os.listdir(self.frame_dir):
                        # load image
                        x = load_image(os.path.join(self.frame_dir, image))

                        # run prediction
                        prediction = sess.run(restored_op, feed_dict={xk: [x], keep_prob: 1.})[0]

                        # transform prediction to black and white and store as png
                        pred_processed = (prediction * 255).astype(np.uint8)
                        self.predictions.append(pred_processed)

                        # create image file in prediction folder
                        image_name = self.model_name + '__' + os.path.splitext(image)[0] + '.png'
                        cv2.imwrite(os.path.join(self.pred_dir, image_name), pred_processed)

                    writer.add_graph(graph)  # add graph to tensorboard

                if model_type == 'keras':
                    print('not implemented')

            print('model loaded and images predicted')

    def flood_extraction(self, threshold=200):
        """ extract a flood index out of detected pixels in the frames

        """
        # define file paths for output data
        prefix_name = self.model_name + '__' + self.video_name
        signal_name = prefix_name + '__' + 'signal.csv'
        plot_name = prefix_name + '__' + 'plot.png'
        signal_file_path = os.path.join(self.signal_dir, signal_name)
        plot_file_path = os.path.join(self.signal_dir, plot_name)

        # define crop window
        top = self.crop_window['top']
        left = self.crop_window['left']
        height = self.crop_window['height']
        width = self.crop_window['width']

        # if predictions are stored in variable then directly otherwise load predictions from pngs
        if len(self.predictions) > 0:
            predictions = self.predictions
        else:
            # natural sort the files in directory
            f_names = os.listdir(self.pred_dir)
            f_names.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            predictions = [cv2.imread(os.path.join(self.pred_dir, file)) for file in f_names]
            predprint = [file for file in f_names]

        # iterate over each predicted frame, crop image and calculate flood index
        flood_index = []
        for pred in predictions:
            pred_crop = pred[top:(top + height), left:(left + width)]
            flood_index.append((pred_crop > threshold).sum() / (pred_crop.shape[0] * pred_crop.shape[1]))

        # export flood_index to csv.
        data = pd.DataFrame(dict(flood_index=flood_index))
        data.to_csv(signal_file_path)

        # plot flood_index and store it.
        data.plot()
        plt.xlabel('index (#)')
        plt.ylabel('flood index (-)')
        plt.show()
        plt.savefig(plot_file_path)
        print('flood signal extracted')

    def train_k_unet(self, train_dir, valid_dir, layers=4, features_root=64, batch_size=1, epochs=20, cost="cross_entropy"):
        """ trains a unet instance on keras. With on-line data augmentation to diversify training samples in each batch.

            example of defining paths
            train_dir = "E:\\watson_for_trend\\3_select_for_labelling\\train_cityscape\\"
            model_dir = "E:\\watson_for_trend\\5_train\\cityscape_l5f64c3n8e20\\"

        """
        img_dir_tr = os.path.join(train_dir, 'images')
        mask_dir_tr = os.path.join(train_dir, 'masks')
        img_dir_va = os.path.join(valid_dir, 'images')
        mask_dir_va = os.path.join(valid_dir, 'masks')

        seed = 1

        x_tr = util.load_images(os.path.join(img_dir_tr, '0'))  # load training pictures in numpy array
        shape = x_tr.shape  # pic_nr x width x height x depth
        n_train = shape[0]  # len(image_generator)
        n_class = 2
        augmentation = True
        subtract_pixel_mean = True
        dir_type = 'flow'

        # define u-net
        model = UNet(shape[1:], out_ch=n_class, start_ch=features_root, depth=layers, inc_rate=1, activation='relu',
                     upconv=False, batchnorm=True)
        model.compile(optimizer=Adam(lr=0.0005), loss='categorical_crossentropy', metrics=['acc', 'categorical_crossentropy'])

        mc = ModelCheckpoint(os.path.join(self.model_dir, 'model.h5'), save_best_only=True, save_weights_only=False)
        es = EarlyStopping(patience=9)
        tb = TensorBoard(log_dir=self.model_dir)

        if augmentation:

            image_datagen = ImageDataGenerator(featurewise_center=False,
                                               featurewise_std_normalization=False,
                                               width_shift_range=0.2,
                                               height_shift_range=0.2,
                                               horizontal_flip=True,
                                               zoom_range=0.0)

            mask_datagen = ImageDataGenerator(width_shift_range=0.2,
                                              height_shift_range=0.2,
                                              horizontal_flip=True,
                                              zoom_range=0.0)

            image_datagen.fit(x_tr, seed=seed)  # calculate mean and stddeviation of training sample for normalisation

            if dir_type == 'dir_flow':

                flow_args_i = dict(target_size=shape[1:-1], class_mode=None, seed=seed, batch_size=batch_size, color_mode='rgb')
                flow_args_m = dict(target_size=shape[1:-1], class_mode=None, seed=seed, batch_size=batch_size, color_mode='grayscale')

                image_generator = image_datagen.flow_from_directory(img_dir_tr, **flow_args_i)
                mask_generator = mask_datagen.flow_from_directory(mask_dir_tr, **flow_args_m)

                img_valid_gen = image_datagen.flow_from_directory(img_dir_va, **flow_args_i)
                mask_valid_gen = mask_datagen.flow_from_directory(mask_dir_va, **flow_args_m)

                n_valid = mask_valid_gen.n

                train_generator = zip(image_generator, mask_generator)
                valid_generator = zip(img_valid_gen, mask_valid_gen)

            else:
                print('flowtyp flow')
                y_tr = util.load_masks(os.path.join(mask_dir_tr, '0'))  # load mask arrays
                x_va = util.load_images(os.path.join(valid_dir, 'images', '0'))
                y_va = util.load_masks(os.path.join(valid_dir, 'masks', '0'))
                n_valid = x_va.shape[0]

                # data normalisation
                img_mean, img_stdev = util.calc_mean_stdev2(x_tr)
                x_tr -= img_mean
                x_tr /= img_stdev
                x_va -= img_mean
                x_va /= img_stdev

                # create one-hot
                y_tr = keras_utils.to_categorical(y_tr, n_class)
                y_va = keras_utils.to_categorical(y_va, n_class)

                # create image generator for online data augmentation
                save_dir = os.path.join(self.model_dir, 'augmentations')
                train_generator = image_datagen.flow(x_tr, y_tr, batch_size=batch_size, save_to_dir=save_dir)
                valid_generator = (x_va, y_va)

            # # loading data with resizing it to size keeping aspect ratio
            # # data augmentation
            # seq = iaa.Sequential([
            #     #iaa.Fliplr(1),  # horizontally flip 50% of the images
            #     iaa.GaussianBlur(sigma=(1.0, 2.0))  # blur images with a sigma of 0 to 3.0
            # ])
            # # x_tr = seq.augment_images(x_tr)
            # print('data preprocessing finished')

            # train unet with image_generator
            model.fit_generator(train_generator,
                                validation_data=valid_generator,
                                steps_per_epoch=n_train / batch_size,
                                validation_steps=n_valid / batch_size,
                                epochs=epochs,
                                verbose=1,
                                callbacks=[mc, es, tb],
                                use_multiprocessing=False,
                                workers=4)

        else:
            y_tr = util.load_masks(os.path.join(mask_dir_tr, '0'))  # load mask arrays
            x_va = util.load_images(os.path.join(valid_dir, 'images', '0'))
            y_va = util.load_masks(os.path.join(valid_dir, 'masks', '0'))

            # Normalize data.
            x_tr = x_tr.astype('float32') / 255
            x_va = x_va.astype('float32') / 255
            # img_mean, img_stdev = util.calc_mean_stdev2(x_tr)
            # x_tr -= img_mean
            # x_tr /= img_stdev
            # x_va -= img_mean
            # x_va /= img_stdev

            # If subtract pixel mean is enabled
            if subtract_pixel_mean:
                x_train_mean = np.mean(x_tr, axis=0)
                x_tr -= x_train_mean
                x_va -= x_train_mean

            # Convert class vectors to binary class matrices.
            y_tr = keras_utils.to_categorical(y_tr, n_class)
            y_va = keras_utils.to_categorical(y_va, n_class)

            model.fit(x_tr, y_tr, validation_data=(x_va, y_va),
                       epochs=epochs, batch_size=batch_size, shuffle=True, callbacks=[mc, es, tb])

        scores = model.evaluate(x_va, y_va, verbose=1)

        print('scores', scores)

    def test_k_unet(self, test_img_dir, layers=4, features_root=64, batch_size=8, channels=3, n_class=2):

        x_va = util.load_images(os.path.join(test_img_dir, 'images', '0'))
        y_va = util.load_masks(os.path.join(test_img_dir, 'masks', '0'))
        y_va = keras_utils.to_categorical(y_va, n_class)

        model = UNet(x_va.shape[1:], out_ch=n_class, start_ch=features_root, depth=layers, inc_rate=1, activation='relu', upconv=False, batchnorm=True)
        model.compile(optimizer=Adam(lr=0.001), loss=f1_loss, metrics=['acc', 'categorical_crossentropy'])

        model.load_weights(os.path.join(self.model_dir, 'model.h5'))
        p_va = model.predict(x_va, batch_size=batch_size, verbose=1)

        scores = model.evaluate(x_va, y_va, verbose=1)
        util.store_prediction(p_va, x_va, self.pred_dir)

        print('DICE:      ' + str(f1_np(y_va, p_va)))
        print('IoU:       ' + str(iou_np(y_va, p_va)))
        print('Precision: ' + str(precision_np(y_va, p_va)))
        print('Recall:    ' + str(recall_np(y_va, p_va)))
        print('Error:     ' + str(error_np(y_va, p_va)))
        print('Scores:    ', scores)

    def predict_k_unet(self, test_img_dir, layers=4, features_root=64, batch_size=8, channels=3, n_class=2):
        x_va = util.load_images(os.path.join(test_img_dir), sort=True)
        model = UNet(x_va.shape[1:], out_ch=n_class, start_ch=features_root, depth=layers, inc_rate=1,
                     activation='relu', upconv=False, batchnorm=True)
        model.compile(optimizer=Adam(lr=0.001), loss=f1_loss, metrics=['acc', 'categorical_crossentropy'])

        model.load_weights(os.path.join(self.model_dir, 'model.h5'))
        p_va = model.predict(x_va, batch_size=batch_size, verbose=1)
        util.store_prediction(p_va, x_va, self.pred_dir)

    def train_tf_unet(self, train_dir, n_class=3, layers=4, features_root=64, channels=3, batch_size=1, training_iters=10,
                   epochs=20, cost="cross_entropy", cost_kwargs={}):
        """ trains a unet instance with n_classes. Other hyperparameters have to be tuned in the code.

            example of defining paths
            train_dir = "E:\\watson_for_trend\\3_select_for_labelling\\train_cityscape\\"
            model_dir = "E:\\watson_for_trend\\5_train\\cityscape_l5f64c3n8e20\\"

        """
        # preparing data loading
        files = os.path.join(train_dir, '*.png')
        img_mean, img_stdev = util.calc_mean_stdev(files, mask_suffix='label.png')

        data_provider = image_util.ImageDataProvider(files, data_suffix=".png", mask_suffix='_label.png',
                                                     shuffle_data=True, n_class=n_class, channel_mean=img_mean, channel_stdev=img_stdev)

        # setup & training
        net = unet.Unet(layers=layers, features_root=features_root, channels=channels, n_class=n_class, cost=cost, cost_kwargs=cost_kwargs)
        trainer = unet.Trainer(net, batch_size=batch_size, norm_grads=False, optimizer="adam")
        path = trainer.train(data_provider, self.model_dir, training_iters=training_iters, epochs=epochs)

        return path

    def test_tf_unet(self, test_img_dir, layers=4, features_root=64, channels=3, n_class=3):
        """ makes test prediction after U-Net is trained.

        Examples of paths
        test_img_dir = "E:\\watson_for_trend\\3_select_for_labelling\\test_cityscape\\"
        test_pred_dir = "E:\\watson_for_trend\\6_test\\cityscape_l5f64c3n8e20\\"

        """
        # prediction
        net = unet.Unet(layers=layers, features_root=features_root, channels=channels, n_class=n_class)
        files = os.path.join(test_img_dir, '*.png')
        data_provider = image_util.ImageDataProvider(files, data_suffix=".png",
                                                     mask_suffix='_label.png', shuffle_data=True, n_class=n_class)

        # loop through files
        for file in data_provider.data_files:
            x, [y] = data_provider(1)

            prediction = net.predict(os.path.join(self.model_dir, 'model.ckpt'), x)

            unet.error_rate(prediction, util.crop_to_shape(y, prediction.shape))

            util.create_overlay(prediction, util.crop_to_shape(x, prediction.shape), os.path.basename(file), self.pred_dir)


if __name__ == '__main__':
    # for apple
    #file_base = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/03_ImageSegmentation/structure_vidFloodExt/'

    # for windows
    file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\'

    video_file = os.path.join(file_base, 'videos', 'ChaskaAthleticPark.mp4')
    video_url = 'https://youtu.be/nrGBtQhAvo8'

    model_file = os.path.join(file_base, 'models', 'flood_keras_c2l4b4e50f32_aug')

    train_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\train'
    valid_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\validate'
    test_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\test'

    pred_dir_flood = os.path.join(file_base, 'predictions', 'keras_pred_allnogen')

    test_dir_elliot = os.path.join(file_base, 'frames', 'elliotCityFlood')
    test_dir_athletic = os.path.join(file_base, 'frames', 'ChaskaAthleticPark')

    cfe = CCTVFloodExtraction(video_file, model_file)

    # importing video and model and do flood extraction
    # cfe.import_video(video_url)
    # cfe.video2frame(resize_dims=(512, 512), keep_aspect=True, max_frames=77)
    # cfe.load_model()
    # cfe.flood_extraction(threshold=200)

    cfe.train_k_unet(train_dir_flood, valid_dir_flood, layers=4, features_root=32, batch_size=4, epochs=50,
                  cost='cross_entropy')
    cfe.test_k_unet(test_dir_flood, layers=4, features_root=32, channels=3, n_class=2)

    cfe.predict_k_unet(test_dir_athletic, layers=4, features_root=32, channels=3, n_class=2)

    # # move pictures from supervisely export
    # src_h = "C:\\Users\\kramersi\\Downloads\\all_flood_raw\\Flood*\\masks_human\\*.png"
    # dst_h = "E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\human\\"
    #
    # src_img = "C:\\Users\\kramersi\\Downloads\\all_flood_raw\\Flood*\\img\\*.jpeg"
    # dst_img = "E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\images\\"
    #
    # src_la = "C:\\Users\\kramersi\\Downloads\\all_flood_raw\\Flood*\\masks_machine\\*.png"
    # dst_la = "E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\labels\\"
    #
    # src_an = "C:\\Users\\kramersi\\Downloads\\all_flood_raw\\Flood*\\ann\\*.json"
    # dst_an = "E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\annotations\\"

    # util.move_pics(src_h, dst_h)
    # util.move_pics(src_img, dst_img)
    # util.move_pics(src_la, dst_la)
    # util.move_pics(src_an, dst_an)
    # util.rename_pics(dst_la + '*')
    # util.convert_images(dst_img, src='jpeg', dst='png')

    # src = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\test\\*.png'
    # dst = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\annotations\\Flood_*_*.json'
    # mask_suffix = '_label.png'
    # import glob
    # for file in glob.glob(dst):
    #     #print(file)
    #     os.remove(file)
    #     #shutil.move(file, dst)

    # import glob
    # for file in glob.glob(os.path.join(test_dir_elliot, '*')):
    #     base, tail = os.path.split(file)
    #     im = cv2.imread(file)
    #     im_resize = util.resize_keep_aspect(im, 512)
    #     util.save_image(im_resize, file)

