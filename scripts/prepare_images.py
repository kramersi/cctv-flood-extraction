"""
script provide some functionalities to rearange images and prepare them for training.

"""
import os
import glob
import cv2
from sofi_extraction.img_utils import transform_mask, resize_keep_aspect, copy_pics, match_label

# # move pictures from supervisely export
# src_h = "C:\\Users\\kramersi\\Downloads\\all_flood_raw\\Flood*\\masks_human\\*.png"
# dst_h = "E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\human\\"
#
# src_img = "C:\\Users\\kramersi\\Downloads\\all_flood_raw\\Flood*\\img\\*.jpeg"
# dst_img = "E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\images\\"
#
# src_la = "C:\\Users\\kramersi\\Downloads\\all_flood_raw\\Flood*\\masks_machine\\*.png"
# dst_la = "E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\labels\\"
#
# src_an = "C:\\Users\\kramersi\\Downloads\\all_flood_raw\\Flood*\\ann\\*.json"
# dst_an = "E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_raw\\annotations\\"

# util.move_pics(src_h, dst_h)
# util.move_pics(src_img, dst_img)
# util.move_pics(src_la, dst_la)
# util.move_pics(src_an, dst_an)
# util.rename_pics(dst_la + '*')
#util.convert_images(dst_img, src='jpeg', dst='png')
# src = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\video_masks\\floodXcam5\\masks\\*'



def match_labels(src_img, src_mask, dst):
    """ util function for matching labels with images and copy them"""
    img_names = glob.glob(src_img)
    mask_names = glob.glob(src_mask)
    msk = [os.path.split(m)[1].split('_')[5] for m in mask_names]
    match = match_label(img_names, msk)
    for m in match:
        copy_pics(m, dst)

# paths
file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\'
src_img = os.path.join(file_base, 'frames', 'FloodXCam5', '*')
src_mask = os.path.join(file_base, 'video_masks', 'FloodXCam1', 'masks', '*')
dst = os.path.join(file_base, 'video_masks', 'FloodXCam5', 'images')

# changes pictures in directory, outcomment steps, which are not necessary
for file in glob.glob(src_mask):
    base, tail = os.path.split(file)
    name = os.path.splitext(tail)[0]
    file_path = os.path.join(base, name)  # path without extension
    ext = 'png'

    im = cv2.imread(file)
    im = transform_mask(im, class_mapping=[(1, 0), (2, 1)])
    im = resize_keep_aspect(im, 512)
    os.remove(file)
    match_labels(src_img, src_mask, dst)
    cv2.imwrite(file_path + '.' + ext, im)  # renamin or changing extension
