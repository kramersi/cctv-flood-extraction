import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

class CCTVFloodExtraction(object):

    def __init__(self, video_file, model_dir, frame_dir=None, pred_dir=None, signal_dir=None):
        self.video_file = video_file
        self.model_dir = model_dir
        self.frame_dir = self.check_create_folder(frame_dir, 'frames')
        self.pred_dir = self.check_create_folder(pred_dir, 'predictions')
        self.signal_dir = self.check_create_folder(signal_dir, 'signal')

        self.model_name = os.path.splitext(os.path.basename(self.model_dir))[0]
        self.video_name = os.path.splitext(os.path.basename(self.video_file))[0]

        self.predictions = []

        self.frame_dir = os.path.join(self.frame_dir, self.video_name)
        self.pred_dir = os.path.join(self.pred_dir, self.video_name)

    def check_create_folder(self, output, folder_name):
        path = self.video_file

        # if none then create diectory on same level as video directory with the folder_name
        if output is None:
            output = os.path.abspath(os.path.join(os.path.dirname(path), os.pardir, folder_name))

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

    def video2frame(self, skip=1, resize_dims=None, mirror=False, max_frames=10, rotate=0):
        """ extract frames out of a video

        Args:
            skip (int): how many frames to skip
            resize_dims (tuple): tuple of pixel width and height. If None then original is kept
            mirror (bool): if frames should be mirrored
            max_frames (int): how much frames in maximumn
            rotate (int): how many degree frames should be rotated. One of 0, 90, 180 or 270

        """

        if os.path.isdir(self.frame_dir):
            print('Picture from this movie already extracted in that directory.')
        else:
            print('created new directory ', self.frame_dir)
            os.makedirs(self.frame_dir)

            video_object = cv2.VideoCapture(self.video_file)  # make video object

            index = 0
            last_mirrored = True

            frame_count = video_object.get(cv2.CAP_PROP_FRAME_COUNT)

            skipDelta = 0
            if max_frames and frame_count > max_frames:
                skipDelta = frame_count / max_frames

            while True:
                success, frame = video_object.read()  # extract frames
                if success:
                    if index % skip == 0:

                        # resize frames
                        if resize_dims is not None:
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

                index += int(1 + skipDelta)
                video_object.set(cv2.CAP_PROP_POS_FRAMES, index)

            print('frame extracted from video')

    def load_model(self, model_type='tensorflow', n_class=2):
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
                tensor_node_y = 'Placeholder_1:0'
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

    def flood_extraction(self, top=0, left=0, width=640, height=360, threshold=200):
        """ extract a flood index out of detected pixels in the frames

        """
        # define file paths for output data
        prefix_name = self.model_name + '__' + self.video_name
        signal_name = prefix_name + '__' + 'signal.csv'
        plot_name = prefix_name + '__' + 'plot.png'
        signal_file_path = os.path.join(self.signal_dir, signal_name)
        plot_file_path = os.path.join(self.signal_dir, plot_name)

        # if predictions are stored in variable then directly otherwise load predictions from pngs
        if len(self.predictions) > 0:
            predictions = self.predictions
        else:
            # natural sort the files in directory
            f_names = os.listdir(self.pred_dir)
            f_names.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
            predictions = [cv2.imread(os.path.join(self.pred_dir, file)) for file in f_names]
            predprint = [file for file in f_names]
            print(predprint)

        # iterate over each predicted frame, crop image and calculate flood index
        flood_index = []
        for pred in predictions:
            pred_crop = pred[top:(top + height), left:(left + width)]
            flood_index.append((pred_crop > threshold).sum() / (pred_crop.shape[0] * pred_crop.shape[1]))

        # export flood_index to csv.
        data = pd.DataFrame(dict(flood_index=flood_index))
        data.to_csv(signal_file_path)

        # plot flood_index and store it.
        plt.figure(1)
        data.plot()
        plt.xlabel('index (#)')
        plt.ylabel('flood index (-)')
        plt.show()
        plt.savefig(plot_file_path)
        print('flood signal extracted')


if __name__ == '__main__':
    # for apple
    #file_base = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/ImageSegmentation/structure_vidFloodExt/'

    # for windows
    file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\ImageSegmentation\\structure_vidFloodExt\\'

    video_file = file_base + 'videos\\ChaskaAthleticPark.mp4'
    model_file = file_base + 'models\\unet_2400'
    video_url = 'https://youtu.be/nrGBtQhAvo8'

    cfe = CCTVFloodExtraction(video_file, model_file)

    #cfe.import_video(video_url)
    cfe.video2frame(resize_dims=(640, 360), max_frames=77)
    cfe.load_model()
    cfe.flood_extraction()