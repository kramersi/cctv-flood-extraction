import numpy as np
import glob
import os
from keras_unet.k_unet import load_images


def combine_pics(img_dir):
    files = glob.glob(img_dir)
    images = []
    img_combs = []
    labels = []
    for f in files:
        base, tail = os.path.split(f)
        images.append(load_images(f))
        labels.append(tail.split('_')[0])
    for img1 in images:
        for img2 in images:
            img_combs.append(np.h_stack(img1, img2))

    return img_combs


def extract_labels(p1, p2):

    level1 = p1.split('_')[-1]
    level2 = p1.split('_')[-1]
    return = np.sign(level2 - level1)

