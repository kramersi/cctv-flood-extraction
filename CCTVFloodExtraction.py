import numpy
import cv2
import os
from datetime import datetime
import numpy as np


class CCTVFloodExtraction(object):

    def __init__(self, video_path, model_path, frame_path):
        self.video_path = video_path
        self.model_path = model_path
        self.frame_path = frame_path

    def setup_outputfolder(self):
        # setup the output folder
        output = self.frame_path
        path = self.video_path
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

    def load_model(self, model_type='unet', channels, classes,):
        if model_type == 'unet':
            from models.tf_unet import unet
            net = unet.Unet(
                channels=self.channels'],
                n_class=self.network['classes'],
                layers=self.network['layers'],
                features_root=self.network['features_root'],
                cost_kwargs=dict(class_weights=s.network['class_weights'])
            )
            # Run prediction
            net.predict_no_label(
                model_path=os.path.join(model_dir, 'model.cpkt'),
                images_dir=sequence_dir, output_dir=output_dir, )

        print('model_loaded')

    def run_segmentation(self):
        print('segmentation started')

    def flood_extraction(self):
        print('flood signal extracted')


if __name__ == '__main__':

    cfe = CCTVFloodExtraction('./videos', './models', '/frames')

    cfe.import_video()
    cfe.video2frame()