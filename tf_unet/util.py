# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 10, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from PIL import Image
import cv2
import os
import glob
import shutil


def plot_prediction(x_test, y_test, prediction, save=False):
    import matplotlib
    import matplotlib.pyplot as plt
    
    test_size = x_test.shape[0]
    fig, ax = plt.subplots(test_size, 3, figsize=(12,12), sharey=True, sharex=True)
    
    x_test = crop_to_shape(x_test, prediction.shape)
    y_test = crop_to_shape(y_test, prediction.shape)
    
    ax = np.atleast_2d(ax)
    for i in range(test_size):
        cax = ax[i, 0].imshow(x_test[i])
        plt.colorbar(cax, ax=ax[i,0])
        cax = ax[i, 1].imshow(y_test[i, ..., 1])
        plt.colorbar(cax, ax=ax[i,1])
        pred = prediction[i, ..., 1]
        pred -= np.amin(pred)
        pred /= np.amax(pred)
        cax = ax[i, 2].imshow(pred)
        plt.colorbar(cax, ax=ax[i,2])
        if i==0:
            ax[i, 0].set_title("x")
            ax[i, 1].set_title("y")
            ax[i, 2].set_title("pred")
    fig.tight_layout()
    
    if save:
        fig.savefig(save)
    else:
        fig.show()
        plt.show()


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255) 
    
    :param img: the array to convert [nx, ny, channels]
    
    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)
    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img


def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border (expects a tensor of shape [batches, nx, ny, channels].
    
    :param data: the array to crop
    :param shape: the target shape
    """
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]


# function defined myself
def create_overlay(prediction, image, name, output_dir):
    class_mapping = {0: [0, 0, 0], 1: [102, 51, 0], 2: [255, 0, 0], 3: [255, 128, 0], 4: [0, 200, 0],
                     5: [0, 255, 255], 6: [128, 128, 128], 7: [255, 255, 255], 8: [0, 0, 255]}

    best_pred = np.argmax(prediction, axis=3)  # take label of maximum probability

    # cv2 colormaps create BGR, not RGB
    # cmap = cv2.cvtColor(cv2.applyColorMap(best_pred, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    # resize the color map to fit image
    img_crop = np.uint8(image[0, :, :, :] * 255)

    # overlay cmap with image
    prediction = np.repeat(best_pred[0, :, :, np.newaxis], 3, axis=2)
    # fill prediction with right rgb colors
    for label, rgb in class_mapping.items():
        prediction[prediction[:, :, 0] == label] = rgb

    overlay_img = cv2.addWeighted(img_crop, 0.8, np.uint8(prediction), 0.5, 0)
    cv2.imwrite(os.path.join(output_dir, os.path.splitext(name)[0] + '.png'),
                cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, grouth thruth and the prediction into one rgb image
    
    :param data: the data tensor
    :param gt: the ground thruth tensor
    :param pred: the prediction tensor
    
    :returns img: the concatenated rgb image 
    """
    ny = pred.shape[2]
    ch = data.shape[3]
    img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)), 
                          to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)), 
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    return img


def load_img_label(img_path, mask_suffix='lable.png', n_class=2):
    """ load images for keras_unte from folder where labels and images are located. labels must have an suffix to differ from images.

    """
    x, y = [], []
    for file in [f for f in glob.glob(img_path) if not f.endswith(mask_suffix)]:
        print('file', file)
        img = cv2.imread(file, 1)
        x.append(img)

    for file in [f for f in glob.glob(img_path) if f.endswith(mask_suffix)]:
        img = cv2.imread(file, 1)
        # one_hot = np.eye(n_class)[img][:, :, 1, :]
        # y.append(one_hot)
        y.append(img)

    return np.array(x), np.array(y)


def calc_mean_stdev(img_path, mask_suffix='_label.png'):
    """ calculates the chanel wise mean and standard deviation of all pictures in that path ending with a number.

    Args:
        img_path (str): path with glob regex where pictures are located.
        mask_suffix (str): suffix of pictures in the folder who are just mask and shouldn't be considered

    Returns:
        mean (ndarray): vector with mean value for each channel (rgb-order)
        std (ndarray): vector with std dev value for each channel (rgb-order)

    """
    m_all = np.zeros(3)
    s_all = np.zeros(3)

    for file in [f for f in glob.glob(img_path) if not f.endswith(mask_suffix)]:
        print('name', file)
        img = cv2.imread(file, 1)

        m, s = cv2.meanStdDev(img)
        m_all = np.column_stack((m_all, m))
        s_all = np.column_stack((s_all, s))

    return np.mean(m_all, axis=1)[::-1], np.mean(s_all, axis=1)[::-1]


def save_image(img, path):
    """
    Writes the image to disk
    
    :param img: the rgb image to save
    :param path: the target path
    """
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)


# defined myself
def move_pics(source, dest):
    for f in glob.glob(source):
        print(f)
        shutil.move(f, dest)


# defined myself
def copy_pics(source, dest):
    for f in glob.glob(source):
        print(f)
        shutil.copy(f, dest)


# defined myself
def rename_pics(source, suffix='_label.png'):
    for img in glob.glob(source):
        base, tail = os.path.split(img)
        name = os.path.splitext(tail)[0]
        os.rename(img, os.path.join(base, name + suffix))