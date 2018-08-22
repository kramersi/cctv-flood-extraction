import os
from img_segmentation.model import UNet

# for apple
# file_base = '/Users/simonkramer/Documents/Polybox/4.Semester/Master_Thesis/03_ImageSegmentation/structure_vidFloodExt/'

# for windows
tune_vid = ''
file_base = 'C:\\Users\\kramersi\\polybox\\4.Semester\\Master_Thesis\\03_ImageSegmentation\\structure_vidFloodExt\\'
model_names = ['train_test_l5_refaug', 'train_test_l3f64aug', 'gentrain_l2f128aug', 'gentrain_l4f32aug_res', 'gentrain_l5f16aug', 'gentrain_l6f8aug_restest']
aug = [True, False, True, True, True, True]
feat = [16, 32, 128, 32, 16, 8]
ep = [250, 200, 200, 200, 200, 200]
lay = [5, 3, 2, 4, 5, 6]
drop = [0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
bat = [8, 8, 2, 4, 8, 6]
res = [True, False, False, True, False, True]
bd = [None, None, None, None, None, None]  # os.path.join(file_base, 'models', 'train_test_l5_' + tune_vid + 'Top')
# bd = [os.path.join(file_base, 'models', 'ft_l5b3e200f16_dr075i2res_lr'), None, None]

for i, model_name in enumerate(model_names):
    if i in [3, 5]:
        model_dir = os.path.join(file_base, 'models', model_name)

        if not os.path.isdir(model_dir):
            os.mkdir(model_dir)

        # configs for fine tune
        # base_model_dir = os.path.join(file_base, 'models', 'cflood_c2l5b3e40f16_dr075caugi2res')
        # base_model_dir = os.path.join(file_base, 'models', 'cflood_c2l4b3e60f64_dr075caugi2')
        #
        # train_dir_ft = os.path.join(file_base, 'video_masks', 'CombParkGarage_train')
        # valid_dir_ft = os.path.join(file_base, 'video_masks', 'CombParkGarage_validate')
        # pred_dir_ft = os.path.join(file_base, 'models', model_name, 'test_img_tf')
        # if not os.path.isdir(pred_dir_ft):
        #     os.mkdir(pred_dir_ft)

        # # configs for training from scratch
        train_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class\\train' # os.path.join(file_base, 'video_masks', 'floodX_cam1', 'train')
        valid_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class\\validate'  #os.path.join(file_base, 'video_masks', 'floodX_cam1', 'validate')
        test_dir_flood = 'E:\\watson_for_trend\\3_select_for_labelling\\dataset__flood_2class\\test'  #os.path.join(file_base, 'video_masks', 'floodX_cam1', 'validate')

        # paths for finetune
        train_dir_further = os.path.join(file_base, 'other_video_masks', 'FurtherYoutube', 'train')
        valid_dir_further = os.path.join(file_base, 'other_video_masks', 'FurtherYoutube', 'validate')

        train_tune_dir = os.path.join(file_base, 'video_masks', tune_vid, 'train')
        valid_tune_dir = os.path.join(file_base, 'video_masks', tune_vid, 'validate')

        pred_dir_flood = os.path.join(file_base, 'models', model_name, 'test_img')
        if not os.path.isdir(pred_dir_flood):
            os.mkdir(pred_dir_flood)

        # configs for testing model
        test_dir_elliot = os.path.join(file_base, 'frames', 'elliotCityFlood')
        test_dir_athletic = os.path.join(file_base, 'frames', 'ChaskaAthleticPark')
        test_dir_floodx = os.path.join(file_base, 'frames', 'FloodX')

        pred_dir = os.path.join(file_base, 'predictions', model_name)
        if not os.path.isdir(pred_dir):
            os.mkdir(pred_dir)

        test_dir = os.path.join(file_base, 'frames', '*')

        # pred_dir_elliot = os.path.join(file_base, 'predictions', model_name, 'elliotCityFlood')
        # pred_dir_athletic = os.path.join(file_base, 'predictions', model_name, 'ChaskaAthleticPark')
        # pred_dir_floodx = os.path.join(file_base, 'predictions', model_name, 'FloodX')

        # test_dirs = [test_dir_elliot, test_dir_athletic, test_dir_floodx]
        # pred_dirs = [pred_dir_elliot, pred_dir_athletic, pred_dir_floodx]

        # for pred_dir in pred_dirs:
        #     if not os.path.isdir(pred_dir):
        #         os.mkdir(pred_dir)

        img_shape = (512, 512, 3)
        unet = UNet(img_shape, root_features=feat[i], layers=lay[i], batch_norm=True, dropout=drop[i], inc_rate=2., residual=res[i])
        # unet.model.summary()

        #unet.train(model_dir, [train_tune_dir], [valid_tune_dir], batch_size=bat[i], epochs=ep[i], augmentation=aug[i], base_dir=bd[i], save_aug=True, learning_rate=0.001)
        unet.train(model_dir, [train_dir_flood, train_dir_further], [valid_dir_flood, valid_dir_further], batch_size=bat[i], epochs=ep[i], augmentation=aug[i], base_dir=bd[i], save_aug=False, learning_rate=0.001)
        #unet.test_gen(model_dir, test_dir_flood, pred_dir_flood, batch_size=3)
        # test_dir = os.path.join(file_base, 'video_masks', '*')
        # #
        # for test in glob.glob(test_dir):  # test for all frames in directory
        #     base, tail = os.path.split(test)
        #     pred = os.path.join(model_dir, 'pred_' + tail)
        #     model_dir = os.path.join(file_base, 'models', model_name)  #  + tail
        #     csv_path = os.path.join(model_dir, tail + '.csv')
        #     test_val = os.path.join(test, 'validate')
        #     if not os.path.isdir(pred):
        #         os.mkdir(pred)
        #
        #     unet.test_gen(model_dir, test_val, pred, batch_size=3, csv_path=csv_path)

        # script for storing prediction
        # from keras_utils import overlay_img_mask
        # vid_name = 'FloodXCam1'
        # img_path = os.path.join(file_base, 'video_masks', vid_name, 'validate', 'images')
        # msk_path = os.path.join(file_base, 'video_masks', vid_name, 'validate', 'masks')
        # output = os.path.join(file_base, 'video_masks', vid_name, 'validate', 'human_masks')
        # if not os.path.isdir(output):
        #     os.mkdir(output)
        # im = load_images(img_path)
        # msk = load_masks(msk_path)
        # for nr, (i, m) in enumerate(zip(im, msk)):
        #     name = 'human' + str(nr) + '.png'
        #     overlay_img_mask(m, i, os.path.join(output, name))