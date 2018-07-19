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
from keras_unet.k_unet import load_images, store_prediction

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

    def load_model(self, model_type='keras'):
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

            if model_type == 'keras':
                # def predict(self, model_dir, img_dir, output_dir, batch_size=4, train_dir=None):
                imgs = load_images(self.frame_dir, sort=True)

                # normalize
                tr_mean = np.array([76.51, 75.41, 71.02])
                tr_std = np.array([76.064, 75.23, 75.03])
                imgs_norm = imgs - tr_mean
                imgs_norm /= tr_std

                model = load_model(os.path.join(self.model_dir, 'model.h5'))

                p_va = model.predict(imgs_norm, batch_size=4, verbose=1)
                store_prediction(p_va, imgs, self.pred_dir)

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

    def create_prediction_movie(self, img_path, pred_path, video_path, trend_path=None, size=(512, 512), fps=5, margin=30, vid_format='DIVX'):
        """ crate a movie by showing images and prediction as well as the trend compared with groundtruth

        """
        # collect all image paths
        img_paths = glob.glob(os.path.join(img_path, '*'))
        pred_paths = glob.glob(os.path.join(pred_path, '*'))

        # read trend and resize
        if trend_path is not None:
            trend = cv2.imread(trend_path)
            cv2.resize(trend, (size[0], size[1]*2 + margin))

        # characteristics of movie
        width = 2 * size[1] + margin  # width of whole movie
        height = size[0]
        line_y1 = size[0] + margin
        line_y2 = 2 * size[0] + margin
        n_img = len(img_paths)

        # define video instance
        # fourcc = cv2.VideoWriter_fourcc(*vid_format)
        # vid = cv2.VideoWriter(video_path, fourcc, float(fps), (512, width), True)

        vid = cv2.VideoWriter(video_path, -1, 1, (width, height))

        # iterate over each image pair and concatenate toghether and put to video
        for i, (img_path, p_path) in enumerate(zip(img_paths, pred_paths)):
            # read images
            img = cv2.imread(img_path)
            pred = cv2.imread(p_path)

            # concatenate pictures togehter with black margins
            space_h = np.full((margin, size[1], 3), 0).astype('uint8')
            composition = np.concatenate((img, space_h, pred), axis=0)

            if trend_path is not None:
                space_w = np.full((width, margin, 3), 0)
                composition = np.concatenate((composition, space_w, trend), axis=1)

                # draw line on trend graph at position x
                line_x = i / n_img
                cv2.line(composition, (line_x, line_y1), (line_x, line_y2), (0, 0, 255), 5)

            vid.write(composition)  # write to video instance

        cv2.destroyAllWindows()
        vid.release()


if __name__ == '__main__':
    # for apple
    #file_base = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/03_ImageSegmentation/structure_vidFloodExt/'

    # for windows
    file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\'

    video_file = os.path.join(file_base, 'videos', 'ChaskaAthleticPark.mp4')

    video_url = 'https://youtu.be/nrGBtQhAvo8'

    # video urls from matthew
    videos = {
        'url': ['https://www.youtube.com/watch?v=LNVCCrVesgg',  # garden long  12:22
            'https://www.youtube.com/watch?v=ZOpWO7rJbtU',  # garage 19:39
            'https://www.youtube.com/watch?v=EXhE_VEJdMY',  # living room 2:07
            'https://www.youtube.com/watch?v=E10us74vZJI',  # roll stairs 0:55
            'https://www.youtube.com/watch?v=6jOxnUkKP8Q',  # creek flood 0:49
            'https://www.youtube.com/watch?v=h-nZGDJSLuk',  # lockwitz 12:05
            'https://www.youtube.com/watch?v=1T68t_QKsuc',  # spinerstrasse 0:15
            'https://www.youtube.com/watch?v=hxcnMQn5zCA',  # hamburg 14:03
            ],
        'names': ['garden', 'garage', 'living_room', 'roll_stairs', 'creek_flood', 'lockwitz', 'spinerstrasse', 'hamburg'],
        'sec': [12*60+22, 19*60+39, 2*60+7, 55, 49, 12*60+5, 12, 14*60+3]
    }

    frames = {
        'name': ['ChaskaAthleticPark', 'FloodX_cam1', 'FloodX_cam5',
    }

    model_file = os.path.join(file_base, 'models', 'flood_keras_c2l4b4e50f32_aug')

    train_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\train'
    valid_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\validate'
    test_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\test'

    pred_dir_flood = os.path.join(file_base, 'predictions', 'keras_pred_allnogen')

    test_dir_elliot = os.path.join(file_base, 'frames', 'elliotCityFlood')
    test_dir_athletic = os.path.join(file_base, 'frames', 'ChaskaAthleticPark')

    cfe = CCTVFloodExtraction(video_file, model_file)
    # import glob
    # movie_path = os.path.join(file_base, 'videos', '*')
    #
    # for video_file in glob.glob(movie_path):
    #     cfe = CCTVFloodExtraction(video_file, model_file)
    #     cfe.video2frame(resize_dims=512, keep_aspect=True, max_frames=1000)

    # importing video and model and do flood extraction
    # cfe.import_video(video_url)
    # cfe.video2frame(resize_dims=(512, 512), keep_aspect=True, max_frames=77)
    # cfe.load_model()
    # cfe.flood_extraction(threshold=200)

    # cfe.train_k_unet(train_dir_flood, valid_dir_flood, layers=4, features_root=32, batch_size=4, epochs=50,
    #               cost='cross_entropy')
    # cfe.test_k_unet(test_dir_flood, layers=4, features_root=32, channels=3, n_class=2)
    #
    # cfe.predict_k_unet(test_dir_athletic, layers=4, features_root=32, channels=3, n_class=2)
    import glob
    img_dir = os.path.join(file_base, 'frames', 'RollStairsTimeLapse')
    pred_dir = os.path.join(file_base, 'predictions', 'cflood_c2l3b3e40f32_dr075caugi2res', 'RollStairsTimeLapse')
    vid_dir = 'predvid.avi'

    if not os.path.isdir(vid_dir):
        os.mkdir(vid_dir)

    cfe.create_prediction_movie(img_dir, pred_dir, vid_dir)

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
    #util.convert_images(dst_img, src='jpeg', dst='png')
    # src = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\video_masks\\floodXcam5\\masks\\*'

    # src = 'E:\\watson_for_trend\\3_select_for_labelling\\train_cityscape\\*_*_*[0-9].png'
    # dst = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\cityscape\\*'
    # mask_suffix = '_label.png'
    # import glob
    # for file in glob.glob(src):
    #     # print(file)
    #     # os.remove(file)
    #     shutil.move(file, dst)

    # import glob
    # for file in glob.glob(src):
    #     base, tail = os.path.split(file)
    #     im = cv2.imread(file)
    #     im_resize = util.transform_mask(im, class_mapping=[(1, 0), (2, 1)])
    #     os.remove(file)
    #     cv2.imwrite(file, im_resize)
    #     # util.save_image(im_resize, file)
    #     # util.create_zero_mask(file)

