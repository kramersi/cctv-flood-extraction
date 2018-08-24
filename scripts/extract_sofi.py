import os
from sofi_extraction.engine import CCTVFloodExtraction

# for apple
# file_base = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/03_ImageSegmentation/structure_vidFloodExt/'
# for windows
file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\'

video_file = os.path.join(file_base, 'videos', 'ChaskaAthleticPark.mp4')

# video urls with interesting floodings
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
    'roi': [[100, 160, 310, 200], [115, 140, 397, 142], [115, 235, 210, 37], [40, 115, 472, 285], [0, 0, 512, 512],
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
        # trained_model = model_name + name
        # model_file = os.path.join(file_base, 'models', trained_model)
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


# importing video and model and do flood extraction
# cfe.import_video(video_url)
# cfe.video2frame(resize_dims=(512, 512), keep_aspect=True, max_frames=77)
# cfe.load_model()
# cfe.flood_extraction(threshold=200)


# def extract_sofi_from_img(path):#
#     flood_index = {}
#     threshold = 0.5
#     for p in glob.glob(path):
#         print('path ', p)
#         pred = cv2.imread(p)
#         p1, p2 = os.path.split(p)
#         flood_index[p2] = (pred[:, :, 1] > threshold).sum() / (pred.shape[0] * pred.shape[1])
#     print(flood_index)
