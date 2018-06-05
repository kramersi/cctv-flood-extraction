import cv2
import os
import numpy as np


class CCTVFloodExtraction(object):

    def __init__(self, video_dir, model_dir, frame_dir, pred_dir):
        self.video_dir = video_dir
        self.model_dir = model_dir
        self.frame_dir = frame_dir
        self.pred_dir = pred_dir

    def setup_outputfolder(self):
        # setup the output folder
        output = self.frame_dir
        path = self.video_dir
        if output is None:
            output = path[:-4]
        else:
            if not output.endswith('/') and not output.endswith('\\'):
                output += '/'
            output += os.paht.splitext(self.video_dir)[-1].split('.')[-1]
        return output

    def import_video(self):
        # video must be of following formats: mp4 | flv | ogg | webm | mkv | avi
        print('video imported')

    def video2frame(self, skip=1, resize_dims=None, mirror=False, max_frames=10, rotate=0):
        """ extract frames out of a video

        Args:
            skip (int): how many frames to skip
            resize_dims (tuple): tuple of pixel width and height. If None then original is kept
            mirror (bool): if frames should be mirrored
            max_frames (int): how much frames in maximumn
            rotate (int): how many degree frames should be rotated. One of 0, 90, 180 or 270

        """
        video_object = cv2.VideoCapture(self.video_dir)
        output = self.setup_outputfolder()
        index = 0
        last_mirrored = True

        frame_count = video_object.get(cv2.CAP_PROP_FRAME_COUNT)

        skipDelta = 0
        if max_frames and frame_count > max_frames:
            skipDelta = frame_count / max_frames

        while True:
            success, frame = video_object.read()
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
                    cv2.imwrite(output + "_" + str(index) + ".jpg", frame)
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
            # data = np.array(Image.open(path), dtype)
            # normalization
            data -= np.amin(data)
            data /= np.amax(data)
            return data

        if model_type == 'tensorflow':
            import tensorflow as tf

            tensor_node_op = 'Assign_1999:0'
            tensor_node_x = 'Placeholder:0'
            tensor_node_y = 'Placeholder_1:0'
            tensor_node_prob = 'Placeholder_2:0'
            meta_file = '/model.cpkt.meta'

            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(self.model_dir + meta_file)
                saver.restore(sess, tf.train.latest_checkpoint(self.model_dir))
                print("Model restored from file: %s" % self.model_dir)

                #x = tf.placeholder("float", shape=[None, None, None, channels])
                #y = tf.placeholder("float", shape=[None, None, None, n_class])
                #keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

                # get the graph in the current thread
                graph = tf.get_default_graph()

                node_names = [tensor.name for tensor in graph.as_graph_def().node]
                print(node_names)

                # access the input key words for feed_dict
                x = graph.get_tensor_by_name(tensor_node_x)
                y = graph.get_tensor_by_name(tensor_node_y)
                keep_prob = graph.get_tensor_by_name(tensor_node_prob)

                # Now, access the operation that you want to run.
                restored_op = graph.get_tensor_by_name(tensor_node_op)

                # loop through files and save predictions
                for image in os.listdir(self.frame_dir):
                    x = load_image(os.path.join(self.frame_dir, image))
                    y_dummy = np.empty((x.shape[0], x.shape[1], x.shape[2], n_class))
                    print(restored_op)
                    print('x', x)
                    print('y', y_dummy)
                    print('keep_prob', keep_prob)
                    prediction = sess.run(graph, feed_dict={x: x, y: y_dummy, keep_prob: 1.})[0]
                    pred_processed = (prediction * 200).astype(np.uint8)
                    cv2.imwrite(os.paht.join(self.output_dir, os.path.splitext(image)[0] + '.png'), pred_processed)


            if model_type == 'keras':
                print('not implemented')

        print('model loaded and images predicted')

    def flood_extraction(self):
        print('flood signal extracted')


if __name__ == '__main__':
    # for apple
    file_base = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/ImageSegmentation/structure_vidFloodExt/'

    # for windows
    # file_base = "C:\\Users\kramersi\polybox\\4.Semester\\Master_Thesis\\ImageSegmentation\\structure_vidFloodExt\\videos\\

    video_file = file_base + 'videos/180131_A_08.mp4'
    model_file = file_base + 'models/all_flipped2_supervisely__ly4ftr16w2__'
    frames_dir = file_base + 'frames'
    pred_dir = file_base + 'predictions'

    cfe = CCTVFloodExtraction(video_file, model_file, frames_dir, pred_dir)

    #cfe.video2frame()
    cfe.load_model()
    #cfe.flood_extraction()