import cv2
import os
from datetime import datetime
import numpy as np
from PIL import Image


class CCTVFloodExtraction(object):

    def __init__(self, video_dir, model_dir, frame_path):
        self.video_dir = video_dir
        self.model_dir = model_dir
        self.frame_dir = frame_dir

    def setup_outputfolder(self):
        # setup the output folder
        output = self.frame_path
        path = self.video_dir
        if output is None:
            output = path[:-4]
        else:
            if not output.endswith('/') and not output.endswith('\\'):
                output += '/'
            output += 'py_image'
        return output

    def import_video(self):
        # video must be of following formats: mp4 | flv | ogg | webm | mkv | avi
        print('video imported')

    def video2frame(self, path, skip=1, resize_dims=None, mirror=False, max_frames=10, rotate=0):

        def _mirror_image(image):
            return np.fliplr(image)

        video_object = cv2.VideoCapture(path)
        output = self.setup_outputfolder()
        index = 0
        last_mirrored = True
        while True:
            success, frame = video_object.read()
            if success:
                if index % skip == 0:

                    if resize_dims is not None:
                        frame = cv2.resize(frame, resize_dims, interpolation=cv2.INTER_CUBIC)

                    if mirror and last_mirrored:
                        frame = _mirror_image(frame)
                    last_mirrored = not last_mirrored

                    cv2.imwrite(output + "_" + str(datetime.now()) + ".jpg",
                                frame)  # assumes that the extension is three letters long
            else:
                break

            index += 1
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
            data = np.array(Image.open(path), dtype)
            # normalization
            data -= np.amin(data)
            data /= np.amax(data)
            return data

        if model_type == 'tensorflow':
            import tensorflow as tf
            tensor_node_op = 'example:0'
            tensor_node_x = 'x:0'
            tensor_node_y = 'y:0'
            tensor_node_prob = 'z:0'

            node_names = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
            print(node_names)

            with tf.Session() as sess:
                saver = tf.train.import_meta_graph('my-model-1000.meta')
                saver.restore(sess, self.model_dir)
                print("Model restored from file: %s" % self.model_dir)

                #x = tf.placeholder("float", shape=[None, None, None, channels])
                #y = tf.placeholder("float", shape=[None, None, None, n_class])
                #keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

                # get the graph in the current thread
                graph = tf.get_default_graph()

                # access the input key words for feed_dict
                x = graph.get_tensor_by_name(tensor_node_x)
                y = graph.get_tensor_by_name(tensor_node_y)
                keep_prob = graph.get_tensor_by_name(tensor_node_prob)

                # Now, access the operation that you want to run.
                restored_op = graph.get_tensor_by_name(tensor_node_op)

                # loop through files and save predictions
                for image in os.listdir(self.images_dir):
                    x = load_image(os.path.join(self.images_dir, image))
                    y_dummy = np.empty((x.shape[0], x.shape[1], x.shape[2], n_class))
                    prediction = sess.run(restored_op, feed_dict={x: x, y: y_dummy, keep_prob: 1.})[0]

                    img = Image.fromarray((prediction * 200).astype(np.uint8))
                    img.save(os.path.join(self.output_dir, os.path.splitext(image)[0] + '.png'))

            if model_type == 'keras':
                print('not implemented')

        print('model loaded and images predicted')

    def flood_extraction(self):
        print('flood signal extracted')


if __name__ == '__main__':

    cfe = CCTVFloodExtraction('./videos', './models', '/frames')

    cfe.import_video()
    cfe.video2frame()