import numpy as np
import glob
import os
from keras_unet.k_unet import load_images


def combine_pics(img_dir):
    files = glob.glob(img_dir)
    images = []
    img_combs = []
    labels = []
    label_combs = []
    for f in files:
        base, tail = os.path.split(f)
        images.append(load_images(f))
        labels.append(int(tail.split('_')[-1]))

    for img1, lab1 in zip(images, labels):
        for img2, lab2 in zip(images, labels):
            img_combs.append(np.h_stack(img1, img2))
            label_combs.append(np.sign(lab2-lab1))

    return np.array(img_combs), np.array(label_combs)


def extract_labels(p1, p2):

    level1 = p1.split('_')[-1]
    level2 = p1.split('_')[-1]
    return = np.sign(level2 - level1)

