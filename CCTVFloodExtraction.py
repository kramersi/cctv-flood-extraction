import numpy
import os


class CCTVFloodExtraction(object):

    def __init__(self, video_path, model_path):
        self.video_path = video_path
        self.model_path = model_path

    def import_video(self):
        print('video imported')

    def video2frame(self):
        print('frame extracted from video')

    def load_model(self):
        print('model_loaded')

    def run_segmentation(self):
        print('segmentation started')

    def flood_extraction(self):
        print('flood signal extracted')


if __name__ == '__main__':

    cfe = CCTVFloodExtraction('./video', './model')

    cfe.import_video()
    cfe.video2frame()