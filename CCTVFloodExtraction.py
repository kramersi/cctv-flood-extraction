import os
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
#from imgaug import augmenters as iaa
import tensorflow as tf

from img_utils import resize_keep_aspect
from keras_utils import load_images, store_prediction, f1_loss, load_img_msk_paths, transform_to_human_mask
from keras.models import load_model
from image_generator import ImageGenerator

class CCTVFloodExtraction(object):

    cr_win = dict(top=0, left=0, width=640, height=360)

    def __init__(self, video_file, model_dir, frame_dir=None, pred_dir=None, signal_dir=None, video_name=None, crop_window=cr_win):

        self.video_file = video_file
        self.model_dir = model_dir

        self.model_name = os.path.splitext(os.path.basename(self.model_dir))[0]
        self.video_name = video_name if video_name is not None else os.path.splitext(os.path.basename(self.video_file))[0]

        self.frame_dir = self.check_create_folder(frame_dir, 'frames', self.video_name)
        self.pred_dir = self.check_create_folder(pred_dir, 'predictions', self.video_name)
        self.signal_dir = self.check_create_folder(signal_dir, 'signal')

        self.predictions = []
        self.model = None
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

    def extract_level_from_name(self):
        images = glob.glob(os.path.join(self.frame_dir, '*'))
        level = []
        for i, im in enumerate(images):
            base, tail = os.path.split(im)
            name = tail.split('.')[-2]
            number = name.split('_')[-1]
            level.append(float(number))
        return np.array(level)

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
                                frame = resize_keep_aspect(frame, resize_dims)
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

    def load_model(self, model_type='keras', create_dir=True):
        """ loads neural network model of different types.

        The pretrained model must have following specifications:
            Input: RGB-Picture as an array of width*height*channel
            Classes: Has to have two classes which are labelled 0=background and 1=water
            Output: array of width*height*channel*class_number

        Args:
            model_type (str): type of model, which should be loaded, has to be one of tensorflow, keras, caffee etc.
            n_class (int): number of classes (defaults to two classes: background and water

        """
        if os.path.isdir(self.pred_dir) and create_dir is True:
            print('Pictures already predicted with that model')
        else:
            if create_dir is True:
                os.makedirs(self.pred_dir)
                print('created new directory ', self.pred_dir)

            if model_type == 'keras':
                # def predict(self, model_dir, img_dir, output_dir, batch_size=4, train_dir=None):

                img_paths = glob.glob(os.path.join(self.frame_dir, '*'))
                img_gen = ImageGenerator(img_paths, batch_size=3, shuffle=False, normalize='std_norm', augmentation=False)

                self.model = load_model(os.path.join(self.model_dir, 'model.h5'), custom_objects={'f1_loss': f1_loss})

            if model_type == 'tensorflow':
                import tensorflow as tf

                def load_image(path, dtype=np.float32):
                    data = np.array(cv2.imread(path), dtype)

                    # normalization
                    data -= np.amin(data)
                    data /= np.amax(data)
                    return data

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

            print('model loaded')

    def predict_images(self, img_paths):
        """ create ImageGenerator and predict images

        Args:
            img_paths (list of str): a batch of all image paths which should be predicted

        """
        img_gen = ImageGenerator(img_paths, batch_size=1, shuffle=False, normalize='std_norm', augmentation=False)

        return self.model.predict_generator(generator=img_gen, verbose=1)

    def flood_extraction(self, threshold=0.5, predictions=None):
        """ extract a flood index out of detected pixels in the frames (prediction)

            Args:
                threshold (float): At which pixel value it is detected as water
                predictions (ndarray):  dimensions: (number * width * height * channel) of the predicted images
        """
        # define crop window
        top = self.crop_window['top']
        left = self.crop_window['left']
        height = self.crop_window['height']
        width = self.crop_window['width']

        # if prediction is not given as parameter then load predictions from stored pngs in the prediction directory
        if predictions is None:
            # natural sort the files in directory
            f_names = os.listdir(self.pred_dir)
            f_names.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            predictions = [cv2.imread(os.path.join(self.pred_dir, file)) for file in f_names]

        # iterate over each predicted frame, crop image and calculate flood index
        flood_index = []
        for pred in predictions:
            pred_crop = pred[top:(top + height), left:(left + width)]
            flood_index.append((pred_crop[:, :, 1] > threshold).sum() / (pred_crop.shape[0] * pred_crop.shape[1]))

        return np.array(flood_index)

    def plot_sofi(self, flood_index, ref_path=None):
        """ extract a flood index out of detected pixels in the frames

            Args:
                flood_index (ndarray): list of all extracted flood indexes (SOFI)
                ref_path (str): three possibilities: None if no reference data availabe, file_name if reference
                                is extracted from file name or path where csv file is laying with two columns: nr, level.
        """
        # define file paths for output data
        prefix_name = self.model_name + '__' + self.video_name
        signal_name = prefix_name + '__' + 'signal.csv'
        plot_name = prefix_name + '__' + 'plot'
        signal_file_path = os.path.join(self.signal_dir, signal_name)
        plot_file_path = os.path.join(self.signal_dir, plot_name)

        # create dataframe for plotting and exporting to csv
        df = pd.DataFrame({'extracted sofi': flood_index})

        # if ref_path defined, then add reference values to dataframe and plot correlation
        if ref_path is not None:
            if ref_path == 'file_name':
                val_ref = self.extract_level_from_name()
                df['reference level'] = val_ref
            else:
                df_ref = pd.read_csv(ref_path, delimiter=';')
                df_ref = df_ref.set_index('nr')
                df_ref = df_ref.interpolate()
                df['reference level'] = df_ref['level'].values

            spe = df.corr(method='spearman').ix[0,1]
            ax = df.plot(kind='scatter', x='reference level', y='extracted sofi')
            ax.text(0.8, 0.1, 'spearman corr.: ' + str(round(spe, 2)), horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, bbox=dict(facecolor='grey', alpha=0.3))
            plt.savefig(plot_file_path + '_corr.png', bbox_inches='tight')
            print('spearman_corr: ', spe)

        # export flood_index to csv.
        df.to_csv(signal_file_path)

        # plot flood_index and store it.
        if 'reference level' in df.columns:
            ax1 = df.plot(figsize=(30, 10), secondary_y=['reference level'])
        else:
            ax1 = df.plot(figsize=(30, 10))
        ax1.set_xlabel('index (#)')
        ax1.set_ylabel('flood index (-)')
        # plt.show()
        plt.savefig(plot_file_path + '_ts.png', bbox_inches='tight')

        return df

    def initialize_movie(self, video_path, size=(512, 512), fps=5, margin=5, vid_format='DIVX'):

        # characteristics of movie
        width = 2 * size[1] + margin  # width of whole movie
        height = size[0] * 2 + margin
        line_y1 = size[0] + margin
        line_y2 = 2 * size[0] + margin

        geometry = {'w': width, 'h': height, 'dim': size[0], 'line_y1': line_y1, 'line_y2': line_y2, 'margin': margin}

        # define video instance
        fourcc = cv2.VideoWriter_fourcc(*vid_format)
        vid = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height), True)
        return vid, geometry

    def write_image_to_vid(self, vid, preds, imgs, trend_path, geometry, n_img, b_ini):

        # iterate over each image pair and concatenate toghether and put to video
        for i, (pred, img) in enumerate(zip(preds, imgs)):
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # concatenate pictures togehter with black margins
            space_h = np.full((geometry['dim'], geometry['margin'], 3), 0).astype('uint8')
            composition = np.concatenate((img, space_h, pred), axis=1)

            space_w = np.full((geometry['margin'], geometry['w'], 3), 0).astype('uint8')
            trend = cv2.imread(trend_path)
            trend = cv2.resize(trend, (geometry['w'], geometry['dim']))
            composition = np.concatenate((composition, space_w, trend), axis=0)

            # draw line on trend graph at position x
            plot_margin = 25
            line_x = plot_margin + int((geometry['w'] - plot_margin - 5) / n_img * (i + b_ini))
            cv2.line(composition, (line_x, geometry['line_y1']), (line_x, geometry['line_y2']), (0, 0, 255), 5)

            vid.write(composition.astype('uint8'))  # write to video instance

        return vid

    def create_prediction_movie(self, video_path, size=(512, 512), fps=5, margin=5, vid_format='DIVX'):
        """ crate a movie by showing images and prediction as well as the trend compared with groundtruth

        """
        # collect all image paths
        img_paths = glob.glob(os.path.join(self.frame_dir, '*'))
        pred_paths = glob.glob(os.path.join(self.pred_dir, '*'))

        img_paths.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        pred_paths.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        plot_name = self.model_name + '__' + self.video_name + '__' + 'plot_ts.png'
        trend_path = os.path.join(self.signal_dir, plot_name)

        # characteristics of movie
        width = 2 * size[1] + margin  # width of whole movie
        height = size[0]
        line_y1 = size[0] + margin
        line_y2 = 2 * size[0] + margin
        n_img = len(img_paths)

        # read trend and resize
        if trend_path is not None:
            trend = cv2.imread(trend_path)
            trend = cv2.resize(trend, (width, height))
            height = height * 2 + margin

        # define video instance
        fourcc = cv2.VideoWriter_fourcc(*vid_format)
        vid = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height), True)
        # vid = cv2.VideoWriter(video_path, -1, 1, (width, height))

        # iterate over each image pair and concatenate toghether and put to video
        for i, (img_path, p_path) in enumerate(zip(img_paths, pred_paths)):
            # read images
            img = cv2.imread(img_path)
            img = cv2.resize(img, size)
            pred = cv2.imread(p_path)

            # concatenate pictures togehter with black margins
            space_h = np.full((size[1], margin, 3), 0).astype('uint8')
            composition = np.concatenate((img, space_h, pred), axis=1)

            if trend_path is not None:
                space_w = np.full((margin, width, 3), 0).astype('uint8')
                composition = np.concatenate((composition, space_w, trend), axis=0)

                # draw line on trend graph at position x
                plot_margin = 25
                line_x = plot_margin + int((width - plot_margin-5) / n_img * i)
                cv2.line(composition, (line_x, line_y1), (line_x, line_y2), (0, 0, 255), 5)

            vid.write(composition)  # write to video instance

        cv2.destroyAllWindows()
        vid.release()

    def run(self, run_types, config, batch_size=300):
        """ runs the extraction with the defined work_types in batches

            Args:
                run_types (list): strings which indicates what should be runned, list should contain one of
                                'import_video', 'extract_frames', 'extract_trend', 'create_prediction_video'
                batch_size (int): number of elements per batch. run extraction of trend in batches because memory
                                to small to load all images at once and store all predictions at once to perform the
                                sofi extraction
        """
        if 'import_video' in run_types:
            self.import_video(config['video_url'])

        if 'extract_frames' in run_types:
            self.video2frame(resize_dims=512, keep_aspect=True, max_frames=1000)

        if 'extract_trend' in run_types:
            # create batches of 100 images and then predict and add to trend
            all_img_paths = glob.glob(os.path.join(self.frame_dir, '*'))
            all_img_paths.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            n_img = len(all_img_paths)
            trend = np.empty(n_img)

            trend_path = os.path.join(self.signal_dir, self.model_name + '__' + self.video_name + '__plot_ts.png')

            output_video = os.path.join(file_base, 'predictions', self.model_name, self.video_name + '_pred.avi')
            output_vid, geometry = self.initialize_movie(output_video, fps=20)

            self.load_model(create_dir=False)

            for b in range(0, n_img, batch_size):
                print('Batch: ' + str(int(b/batch_size)) + '/' + str(int(n_img/batch_size)))
                batch = all_img_paths[b:b + batch_size]
                pred = self.predict_images(batch)
                imgs = load_images(batch)
                trend[b:b + batch_size] = self.flood_extraction(predictions=pred)
                print('images predicted')

                self.plot_sofi(trend, ref_path=ref_path)
                pred_tr = transform_to_human_mask(pred, imgs)

                output_vid = self.write_image_to_vid(output_vid, pred_tr, imgs, trend_path, geometry, n_img, b)

            cv2.destroyAllWindows()
            output_vid.release()


if __name__ == '__main__':
    # for apple
    # file_base = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/03_ImageSegmentation/structure_vidFloodExt/'

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
    model_name = 'ft_l5b3e200f16_dr075i2res_lr'
    config = [dict(
        video_url='https://youtu.be/nrGBtQhAvo8',
        video_file=os.path.join(file_base, 'videos', 'ChaskaAthleticPark.mp4'),
        model_dir=os.path.join(file_base, 'models', model_name),
        pred_dir=os.path.join(file_base, 'predictions', model_name),
        frame_dir=os.path.join(file_base, 'frames'),
        ref_file=os.path.join(file_base, 'frames', 'ChaskaAthleticPark.csv'),
        output_video=os.path.join(file_base, 'predictions', 'ChaskaAtheleticPark_pred.avi'),
        name='ChaskaAtheleticPark',
        roi=[0, 0, 512, 512],
        fps=1
    )]

    frames = {
        'name': ['ChaskaAthleticPark', 'FloodX_cam1', 'FloodX_cam5', 'HoustonHarveyGarage', 'HoustonHarveyGarden',
                 'HamburgFischauktion',],
        'roi': [[0, 0, 512, 512], [0, 0, 512, 512], [0, 0, 512, 512], [0, 0, 512, 512], [0, 0, 512, 512],
                [0, 0, 512, 512]],
        'fps': [1, 1, 15, 15, 15, 15],
        'ref': [os.path.join(file_base, 'frames', 'ChaskaAthleticPark.csv'), 'file_name', 'file_name',
                os.path.join(file_base, 'frames', 'HoustonHarveyGarage.csv'), None, None]
    }
    model_name = 'ft_l5b3e200f16_dr075i2res_lr'
    model_file = os.path.join(file_base, 'models', model_name)

    for i, name in enumerate(frames['name']):
        if i in [1, 2]:
            pred_dir_flood = os.path.join(file_base, 'predictions', model_name)
            frame_dir_flood = os.path.join(file_base, 'frames')
            vid_dir_flood = os.path.join(pred_dir_flood, name + '_pred.avi')
            ref_path = frames['ref'][i]
            cr_win = dict(top=frames['roi'][i][0], left=frames['roi'][i][1], width=frames['roi'][i][2], height=frames['roi'][i][3])
            cfe = CCTVFloodExtraction(video_file, model_file, pred_dir=pred_dir_flood, frame_dir=frame_dir_flood,
                                      video_name=name, crop_window=cr_win)
            cfe.run(['extract_trend'], config)


    # test_dir_elliot = os.path.join(file_base, 'frames', 'elliotCityFlood')
    # test_dir_athletic = os.path.join(file_base, 'frames', 'ChaskaAthleticPark')
    #
    # cfe = CCTVFloodExtraction(video_file, model_file)

    # import glob
    # img_dir = os.path.join(file_base, 'frames', 'RollStairsTimeLapse')
    # pred_dir = os.path.join(file_base, 'predictions', 'cflood_c2l3b3e40f32_dr075caugi2res', 'RollStairsTimeLapse')
    # vid_dir = os.path.join(file_base, 'predictions', 'predvid.avi')
    #
    # # if not os.path.isdir(vid_dir):
    # #     os.mkdir(vid_dir)
    # cfe.load_model()
    # cfe.create_prediction_movie(img_dir, pred_dir, vid_dir)

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

    # train_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\train'
    # valid_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\validate'
    # test_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\test'

    # import glob
    # movie_path = os.path.join(file_base, 'videos', '*')
    #
    # for video_file in glob.glob(movie_path):
    # video_file = os.path.join(file_base, 'videos', 'ChaskaAthleticPark.mp4')
    # cfe = CCTVFloodExtraction(video_file, model_file)
    # cfe.video2frame(resize_dims=512, keep_aspect=True, max_frames=1000)

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
