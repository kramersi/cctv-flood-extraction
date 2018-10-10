import os
import glob
import cv2
from sofi_extraction.engine import CCTVFloodExtraction

file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\'

vid_dir = os.path.join(file_base, 'video_masks')

frames = {
    'name': ['AthleticPark',
             'FloodXCam1',
             'FloodXCam5',
             'HoustonGarage',
             'HarveyParking',
             'BayouBridge'
             ],
    'roi_old': [
        [100, 160, 310, 200],
        [115, 140, 397, 142],
        [115, 235, 210, 37],
        [40, 115, 472, 285],
        [0, 0, 512, 512],
        [0, 0, 512, 512]
    ],
    'roi': [
        [102, 171, 327, 236],
        [275, 136, 174, 62],
        [8, 239, 101, 43],
        [185, 114, 205, 217],
        [127, 267, 95, 105],
        [58, 124, 401, 259]
    ]
}

models = [
    'train_test_l5_AthleticPark',
    'train_test_l5_FloodXCam1',
    'train_test_l5_HoustonGarage',
    'train_test_l5_HarveyParking',
    'train_test_l5_BayouBridge',
    'train_test_l5',
    'train_test'
]

# adding red box for roi to images
file_location = 'Q:\\Abteilungsprojekte\\eng\\SWWData\\SimonKramer\\data_for_matthew\\images_with_roi'

for i, vid in enumerate(frames['name']):
    for sc in ['groundtruth', 'basic', 'augmented', 'finetuned']:
        for img_dir in glob.glob(os.path.join(file_location, vid, sc, '*')):

            # read image
            img = cv2.imread(img_dir)

            # add roi rectangle to prediction
            left_roi = frames['roi'][i][0]
            top_roi = frames['roi'][i][1]
            w_roi = frames['roi'][i][2]
            h_roi = frames['roi'][i][3]

            cv2.rectangle(img, (left_roi, top_roi), (left_roi + w_roi, top_roi + h_roi), (82, 78, 220), 3)

            # extract direction of new image added with roi and save it
            base, tail = os.path.split(img_dir)
            name = os.path.splitext(tail)[0]
            img_new_name = os.path.join(base, name + '_roi.png')
            cv2.imwrite(img_new_name, img)



# # extracting IOU
# for vid_dir in frames['name']:
#     for sc in ['groundtruth', 'basic', 'augmented', 'finetuned']
#     vid_name = os.path.split(vid_dir)[0]
#     val_dir = os.path.join(vid_dir, 'validate')
#     video_file = os.path.join(file_base, 'videos', vid_name + '.mp4')