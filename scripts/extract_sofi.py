import os
from sofi_extraction import CCTVFloodExtraction

# for apple
# file_base = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/03_ImageSegmentation/structure_vidFloodExt/'

# for windows
file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\'

video_file = os.path.join(file_base, 'videos', 'ChaskaAthleticPark.mp4')

video_url = 'https://youtu.be/nrGBtQhAvo8'

# video urls from matthew
videos = {
    'url': ['https://www.youtube.com/watch?v=LNVCCrVesgg',  # garden long  12:22
        'https://www.youtube.com/watch?v=ZOpWO7rJbtU',  # garage 19:39
        'https://www.youtube.com/watch?v=EXhE_VEJdMY',  # living room 2:07
        'https://www.youtube.com/watch?v=E10us74vZJI',  # roll stairs 0:55
        'https://www.youtube.com/watch?v=6jOxnUkKP8Q',  # creek flood 0:49
        'https://www.youtube.com/watch?v=h-nZGDJSLuk',  # lockwitz 12:05
        'https://www.youtube.com/watch?v=1T68t_QKsuc',  # spinerstrasse 0:15
        'https://www.youtube.com/watch?v=hxcnMQn5zCA',  # hamburg 14:03

        'https://www.youtube.com/watch?v=GhczhkuOEiU',  # different floods to show image segmentation
        'https://www.youtube.com/watch?v=9nZaT8r6qYM',  # harvey parking flood
        'https://www.youtube.com/watch?v=y6jByqVX7PE' # under bridge
        ],
    'names': ['garden', 'garage', 'living_room', 'roll_stairs', 'creek_flood', 'lockwitz', 'spinerstrasse', 'hamburg', 'streetFlood', 'HarveyParking', 'BayouBridge'],
    'sec': [12*60+22, 19*60+39, 2*60+7, 55, 49, 12*60+5, 12, 14*60+3]
}
model_name = 'ft_l5b3e200f16_dr075i2res_lr'
config = [dict(
    video_url='https://youtu.be/nrGBtQhAvo8',
    video_file=os.path.join(file_base, 'videos', 'ChaskaAthleticPark.mp4'),
    model_dir=os.path.join(file_base, 'models', model_name),
    pred_dir=os.path.join(file_base, 'predictions', model_name),
    frame_dir=os.path.join(file_base, 'frames'),
    ref_file=os.path.join(file_base, 'frames', 'ChaskaAthleticPark.csv'),
    output_video=os.path.join(file_base, 'predictions', 'ChaskaAtheleticPark_pred.avi'),
    name='ChaskaAtheleticPark',
    roi=[0, 0, 512, 512],
    fps=1
)]

frames = {
    'name': ['AthleticPark', 'FloodXCam1', 'FloodXCam5', 'HoustonGarage', 'HoustonHarveyGarden',
             'HamburgFischauktion', 'HarveyParking', 'BayouBridge', 'StreetFlood', 'LockwitzbachDresden'],
    'roi': [[100, 160, 310, 200], [115, 140, 397, 142], [0, 130, 512, 270], [40, 115, 472, 285], [0, 0, 512, 512],
            [0, 0, 512, 512], [20, 250, 492, 150], [5, 250, 500, 180], [0, 0, 512, 512], [0, 0, 512, 512]],
    'fps': [1, 1, 15, 15, 15, 15, 15, 15, 15, 10],
    'ref': [os.path.join(file_base, 'frames', 'AthleticPark.csv'), 'file_name', 'file_name',
            os.path.join(file_base, 'frames', 'HoustonGarage.csv'), None, None,
            os.path.join(file_base, 'frames', 'HarveyParking.csv'), os.path.join(file_base, 'frames', 'BayouBridge.csv'),
            None, None],
    'model': ['train_test_l5_AthleticPark', 'train_test_l5_FloodXCam1', 'train_test_l5_HoustonGarage', 'train_test_l5',
              'train_test_l5', 'train_test_l5_HarveyParking', 'train_test_l5_BayouBridge', 'train_test_l5', 'train_test']
}
model_name = 'train_test_l5_aug_reduced'  #'ft_l5b3e200f16_dr075i2res_lr'  # 'ft_l5b3e200f16_dr075i2res_lr'
model_file = os.path.join(file_base, 'models', model_name)

for i, name in enumerate(frames['name']):
    if i in [2]:  # [0, 1, 2, 3, 6, 7]
        #trained_model = model_name + name
        #model_file = os.path.join(file_base, 'models', trained_model)
        pred_dir_flood = os.path.join(file_base, 'predictions', model_name)
        frame_dir_flood = os.path.join(file_base, 'frames')
        vid_dir_flood = os.path.join(pred_dir_flood, name + '_pred.avi')
        ref_path = frames['ref'][i]
        cr_win = dict(left=frames['roi'][i][0], top=frames['roi'][i][1], width=frames['roi'][i][2], height=frames['roi'][i][3])
        cfe = CCTVFloodExtraction(video_file, model_file, pred_dir=pred_dir_flood, frame_dir=frame_dir_flood,
                                  video_name=name, crop_window=cr_win)
        cfe.run(['extract_trend'], config, ref_path=ref_path)


# # iterate over movies
# for i, url in enumerate(videos['url']):
#     if i in [10]:
#         video_file = os.path.join(file_base, 'videos', videos['names'][i] + '.mp4')
#         cfe = CCTVFloodExtraction(video_file, model_file)
#         cfe.video2frame(resize_dims=512, keep_aspect=True, max_frames=4000)

# test_dir_elliot = os.path.join(file_base, 'frames', 'elliotCityFlood')
# test_dir_athletic = os.path.join(file_base, 'frames', 'ChaskaAthleticPark')
#
# cfe = CCTVFloodExtraction(video_file, model_file)

# import glob
# img_dir = os.path.join(file_base, 'frames', 'RollStairsTimeLapse')
# pred_dir = os.path.join(file_base, 'predictions', 'cflood_c2l3b3e40f32_dr075caugi2res', 'RollStairsTimeLapse')
# vid_dir = os.path.join(file_base, 'predictions', 'predvid.avi')
#
# # if not os.path.isdir(vid_dir):
# #     os.mkdir(vid_dir)
# cfe.load_model()
# cfe.create_prediction_movie(img_dir, pred_dir, vid_dir)

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

# src = 'E:\\watson_for_trend\\3_select_for_labelling\\train_cityscape\\*_*_*[0-9].png'
# dst = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\cityscape\\*'
# mask_suffix = '_label.png'
# import glob
# for file in glob.glob(src):
#     # print(file)
#     # os.remove(file)
#     shutil.move(file, dst)

# import glob
# for file in glob.glob(src):
#     base, tail = os.path.split(file)
#     im = cv2.imread(file)
#     im_resize = util.transform_mask(im, class_mapping=[(1, 0), (2, 1)])
#     os.remove(file)
#     cv2.imwrite(file, im_resize)
#     # util.save_image(im_resize, file)
#     # util.create_zero_mask(file)

# train_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\train'
# valid_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\validate'
# test_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class_resized\\test'

# import glob
# movie_path = os.path.join(file_base, 'videos', '*')
#
# for video_file in glob.glob(movie_path):
# video_file = os.path.join(file_base, 'videos', 'ChaskaAthleticPark.mp4')
# cfe = CCTVFloodExtraction(video_file, model_file)
# cfe.video2frame(resize_dims=512, keep_aspect=True, max_frames=1000)

# importing video and model and do flood extraction
# cfe.import_video(video_url)
# cfe.video2frame(resize_dims=(512, 512), keep_aspect=True, max_frames=77)
# cfe.load_model()
# cfe.flood_extraction(threshold=200)

# cfe.train_k_unet(train_dir_flood, valid_dir_flood, layers=4, features_root=32, batch_size=4, epochs=50,
#               cost='cross_entropy')
# cfe.test_k_unet(test_dir_flood, layers=4, features_root=32, channels=3, n_class=2)
#
# cfe.predict_k_unet(test_dir_athletic, layers=4, features_root=32, channels=3, n_class=2)

# parking_path = os.path.join(file_base, 'video_masks', 'HarveyParking', 'validate', 'masks', '*')
#
# flood_index = {}
# threshold = 0.5
# for p in glob.glob(parking_path):
#     print('path ', p)
#     pred = cv2.imread(p)
#     p1, p2 = os.path.split(p)
#     flood_index[p2] = (pred[:, :, 1] > threshold).sum() / (pred.shape[0] * pred.shape[1])
# print(flood_index)
