import numpy as np
import os
import keras
from skimage import exposure
import imgaug as ia
from imgaug import augmenters as iaa
from keras.preprocessing.image import img_to_array, load_img, save_img

from random import randint


class ImageGenerator(keras.utils.Sequence):
    """ Generates batches of images as well as the associated labels to detect classes. This generator is passed to the
    fit generator function.

    Args:
       img_paths (list): paths to the images
       masks (dict): mapping of img_paths to mask_paths, e.g. {img1: mask1, img2: mask2, ...} if None just images are
       processed.
       batch_size (int): number of samples in output array of each interation of the generate method.
       dim (tuple): dimension of pictures
       n_channels (int): number of channels per picture, (1=greyscale, 3=rgb)
       n_classes (int): number of classes
       shuffle (bool): if images should be shuffled after each epoch
       normalize (str): one of (std_norm, minmax_norm, None) if not None images will be normalized by type specified
       augmentation (bool): if augmentation should be conducted. If true a aug_dict has to be defined
       aug_dict (bool): dict with which augmentation should be conducted and in which range following are possible
            horizontal_flip (float): probability with which pictures are flipped horizontally
            vertical_flip (float): probability with which pictures are flipped horizontally
            rotation_range (tuple): tuple of range of roatation in degrees, e.g. (-45, 45). None if no rotation
            width_shift_range (tuple): range of shift in x direction, e.g. (-0.2, 0.2) translate by 20%
            height_shift_range (tuple): range of shift in y direction, e.g. (-0.2, 0.2) translate by 20%
            zoom_range (tuple): range of zoom aspect ratio is perceived, bigger than 1 zooms in lower zooms out
            grayscale_range (tuple): use of color the 0=no change, 1.0 just grayscale, image is still 3 channels
            brightness_range (tuple): 0=black, 1=same brightness, 2=very bright
            contrast_range (tuple):
            crop_range (tuple): range for random crop
            blur_range (tuple): range of gaussian blurring, 0=no blurring, 4=extremly blurred
            shear_range (tuple): range of shear in degrees 0 = no shearing
            prob (float): probability (0-1) on how often the augmenters should be used

    """
    aug_dict = dict(horizontal_flip=0.0, vertical_flip=0.0, rotation_range=0.0,
                    width_shift_range=0.0, height_shift_range=0.0, contrast_range=1.0,
                    zoom_range=(1.0, 1.0), grayscale_range=0.0, brightness_range=1.0, crop_range=(0, 0),
                    blur_range=0.0, shear_range=0.0, prob=0.25)

    def __init__(self, img_paths, masks=None, batch_size=3, dim=(512, 512), n_channels=3, n_classes=2, shuffle=True, normalize=None,
                 augmentation=False, save_to_dir=None, aug_dict=aug_dict):
        self.dim = dim
        self.batch_size = batch_size
        self.masks = masks
        self.img_paths = img_paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.normalize = normalize
        self.augmentation = augmentation
        self.aug_dict = aug_dict
        self.save_to_dir = save_to_dir
        self.on_epoch_end()

        if self.augmentation is True:
            self.seq = self.get_augmentation_sequence()

    def __len__(self):
        """Denotes the number of batches per epoch

        """
        return int(np.floor(len(self.img_paths) / self.batch_size))

    def __getitem__(self, index):
        """"Generate one batch of data

        Args:
            index (int): index of first image in batch of all images

        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        img_paths_temp = [self.img_paths[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(img_paths_temp)

        return x, y

    def get_augmentation_sequence(self):
        a = self.aug_dict
        sometimes = lambda aug: iaa.Sometimes(a['prob'], aug)

        augmenters = [iaa.Fliplr(a['horizontal_flip'], name='fliplr'),  # flip horizontally
                      iaa.Flipud(a['vertical_flip'], name='flipup'),  # flip vertically
                      iaa.CropAndPad(percent=a['crop_range'], sample_independently=False, name='crop'),  # random crops
                      iaa.Affine(
                          scale=a['zoom_range'],  # scale images to % of their size
                          translate_percent={"x": a['width_shift_range'], "y": a['height_shift_range']},  # translate by percent
                          rotate=a['rotation_range'],  # rotate by -45 to +45 degrees
                          shear=a['shear_range'],  # shear by -16 to +16 degrees
                          order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                          mode='wrap',  # use any of scikit-image's warping modes
                          name='affine'),
                      sometimes(iaa.ContrastNormalization(a['contrast_range'], name='contrast')),  # change contrast
                      sometimes(iaa.GaussianBlur(sigma=a['blur_range'], name='blur')),  # Adding blur
                      sometimes(iaa.Grayscale(alpha=a['grayscale_range'], name='grayscale')),  # reduce color of image
                      sometimes(iaa.Multiply(a['brightness_range'], name='brightness'))]  # darker or brighter

        return iaa.Sequential(augmenters, random_order=True)

    def on_epoch_end(self):
        """" Updates indexes after each epoch

        """
        self.indexes = np.arange(len(self.img_paths))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, img_paths_temp):
        """Generates data containing batch_size samples x : (n_samples, *dim, n_channels)

        Args:
            img_paths_temp (list of str): list of image paths in that batch

        """
        # Initialization
        x = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8)
        y = np.zeros((self.batch_size, *self.dim), dtype=np.uint8)

        # Generate data
        for i, img_path in enumerate(img_paths_temp):
            # Load image and mask
            im = load_img(img_path, target_size=self.dim)

            # Store sample
            x[i, ] = img_to_array(im).astype(np.uint8)  # np.load('data/' + ID + '.npy')

            if self.masks is not None:
                msk = load_img(self.masks[img_path])
                y[i, ] = img_to_array(msk)[:, :, 0].astype(np.uint8)

        # Augment image and mask
        if self.augmentation is True:
            x, y = self.__data_augmentation(x, y, img_paths_temp)

        # Normalize batch images
        if self.normalize is not None:
            x = self.__data_normalisation(x)

        y_cat = keras.utils.to_categorical(y, num_classes=self.n_classes)  # transform mask to one-hot encoding

        return x, y_cat

    def __data_normalisation(self, img, hist_eq=False):
        """ normalizing of data, e.g. normalisation or contrast enhancements

        Args:
            img (ndarray): numpy array of image
            type (str): one of std_norm or minmax_norm to clarify which type of normalisation
            hist_eq (bool): whether to do histogram equalisation or not

        """
        # normalize
        if self.normalize == 'std_norm':
            tr_mean = np.array([69.7399, 69.8885, 65.1602])
            tr_std = np.array([72.9841, 72.3374, 71.6508])

            img_norm = (img - tr_mean) / tr_std

        if self.normalize == 'minmax_norm':
            img_norm = (img - np.amin(img)) / np.amax(img)

        # histogram equalisation
        if hist_eq is True:
            img_norm = exposure.equalize_hist(img_norm)

        return img_norm

    def __data_augmentation(self, x, y, im_paths):
        """augments data and labels, if necessary

        """
        def activator(images, augmenter, parents, default):
            return False if augmenter.name in ['blur', 'contrast', 'grayscale', 'brightness'] else default

        seq_det = self.seq.to_deterministic()
        x_aug = seq_det.augment_images(x)
        y_aug = seq_det.augment_images(y, hooks=ia.HooksImages(activator=activator))

        if self.save_to_dir is not None:
            for i, p in enumerate(im_paths):

                base, tail = os.path.split(p)
                root, ext = os.path.splitext(tail)
                save_dir = os.path.join(self.save_to_dir, root + '_' + str(randint(0, 1000)) + ext)
                save_img(save_dir, x_aug[i, ])

        return x_aug, y_aug