import glob
import os
import numpy as np
import cv2
import re

from keras.preprocessing.image import img_to_array, load_img
from keras.losses import *


def load_img_msk_paths(paths):
    """ load images and masks paths from different folders to one dictonary which will serve for later generator of
    images.

        Args:
            paths (list): list of different paths to folder
        Hint: structure of folder has to be the folder with the subfolders mask and images, which can be ordered the same.
            -- Folder_name
                |---- images
                        |---- img1.png
                        |---- img2.png
                |---- masks
                        |---- img1-mask.png
                        |---- img2-label.png

        Attention:
            names in different foolders should not have the same name.

        Returns:
            dict where keys are all images with the value as the mask path, e.g.
            {'img1.png': 'img1-mask.png', 'img2.png': 'img2-label.png'}
    """
    img_mask_paths = {}
    for path in paths:
        p_img = glob.glob(os.path.join(path, 'images', '*'))
        p_msk = glob.glob(os.path.join(path, 'masks', '*'))
        # p_img.sort()
        # p_msk.sort()
        p_img.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        p_msk.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        for im, msk in zip(p_img, p_msk):
            img_mask_paths[im] = msk

    return img_mask_paths


def load_masks(path):
    files = glob.glob(os.path.join(path, '*'))
    first_img = load_img(files[0])

    n = len(files)
    w = first_img.width
    h = first_img.height
    x = np.empty((n, w, h))

    for i, f in enumerate(files):
        im = load_img(f)
        x[i, :, :] = img_to_array(im)[:, :, 0].astype(np.int8)

    return x


def load_images(path, sort=False, target_size=None, max_files=200):
    if isinstance(path, str):
        files = glob.glob(os.path.join(path, '*'))
    else:
        files = path
    if sort is True:
        files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    first_img = load_img(files[0])

    n = len(files)  # min(len(files), max_files)
    if target_size is not None:
        h = target_size[0]
        w = target_size[1]
    else:
        w = first_img.width
        h = first_img.height
    x = np.empty((n, h, w, 3))

    for i, f in enumerate(files):
        im = load_img(f, target_size=target_size)
        x[i, :, :, :] = img_to_array(im).astype(np.float32)

    return x.astype('float32')


def channel_mean_stdev(img):
    m = np.mean(img, axis=(0, 1, 2))
    s = np.std(img, axis=(0, 1, 2))
    return m, s


def transform_to_human_mask(predictions, images, class_mapping={0: [0, 0, 0], 1: [0, 0, 255]}):
    pred_transformed = []
    for pred, img in zip(predictions, images):
        best_pred = np.argmax(pred, axis=-1)  # take label of maximum probability

        # # resize the color map to fit image
        # img_crop = np.uint8(img[0, :, :, :] * 255)

        # overlay cmap with image
        prediction = np.repeat(best_pred[:, :, np.newaxis], 3, axis=-1)
        # fill prediction with right rgb colors
        for label, rgb in class_mapping.items():
            prediction[prediction[:, :, 0] == label] = rgb

        prediction = cv2.addWeighted(np.uint8(img), 0.8, np.uint8(prediction), 0.5, 0)
        pred_transformed.append(prediction)

    return pred_transformed


def store_prediction(predictions, images, output_dir, overlay=True):
    class_mapping = {0: [0, 0, 0], 1: [0, 0, 255]}
    count = 0
    for pred, img in zip(predictions, images):
        best_pred = np.argmax(pred, axis=-1)  # take label of maximum probability

        # # resize the color map to fit image
        # img_crop = np.uint8(img[0, :, :, :] * 255)

        # overlay cmap with image
        prediction = np.repeat(best_pred[:, :, np.newaxis], 3, axis=-1)
        # fill prediction with right rgb colors
        for label, rgb in class_mapping.items():
            prediction[prediction[:, :, 0] == label] = rgb

        if overlay is True:
            overlay_img = cv2.addWeighted(np.uint8(img), 0.8, np.uint8(prediction), 0.5, 0)
            cv2.imwrite(os.path.join(output_dir, 'pred' + str(count) + '.png'),
                        cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(os.path.join(output_dir, 'pred'+ str(count) + '.png'),
                        cv2.cvtColor(np.uint8(prediction), cv2.COLOR_RGB2BGR))
        count +=1


# Loss Functions

# 2TP / (2TP + FP + FN)
def f1(y_true, y_pred):
    # weight = np.array([1, 10])
    # k_weight = K.variable(weight[None, None, :])
    # y_true = y_true * k_weight
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)


def f1_np(y_true, y_pred):
    return (2. * (y_true * y_pred).sum() + 1.) / (y_true.sum() + y_pred.sum() + 1.)


def f1_loss(y_true, y_pred):
    return 1-f1(y_true, y_pred)

dice = f1
dice_loss = f1_loss

def iou(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1. - intersection)

def iou_np(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1.) / (y_true.sum() + y_pred.sum() + 1. - intersection)

def iou_loss(y_true, y_pred):
    return -iou(y_true, y_pred)

def precision(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_pred_f) + 1.)

def precision_np(y_true, y_pred):
    return ((y_true * y_pred).sum() + 1.) / (y_pred.sum() + 1.)

def recall(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.) / (K.sum(y_true_f) + 1.)

def recall_np(y_true, y_pred):
    return ((y_true * y_pred).sum() + 1.) / (y_true.sum() + 1.)

def mae_img(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return mae(y_true_f, y_pred_f)

def bce_img(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return binary_crossentropy(y_true_f, y_pred_f)

# FP + FN
def error(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return K.sum(K.abs(y_true_f - y_pred_f)) / float(512*512)

def error_np(y_true, y_pred):
    return (abs(y_true - y_pred)).sum() / float(len(y_true.flatten()))