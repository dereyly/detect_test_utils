from easydict import EasyDict as edict

# init easydict
__AUG = edict()
aug_cfg = __AUG

# project name
__AUG.project_name = 'dairy'

# csv path
__AUG.csv_path = '/media/disk200/lots/dairy.csv'

# previous_sku_space_package_type_txt for making correct order
__AUG.previous_sku_space_package_type_txt = '/media/disk200/lots/sku_space_package_type_empty.txt'

# db names
#__AUG.db_foreground_images = "markup_dairy_full_20171120"

# augmentation parameters
__AUG.IMG_MAX_SIZE = 1200
__AUG.num_out_files_for_one_img_in_trainval_using_augmentation = 1
__AUG.background_percent = 0
__AUG.jpg_quality = 60
__AUG.visualization = False
__AUG.part_for_save = 0.35
__AUG.separate_parts_bigger_side = 4
__AUG.separate_parts_smaller_side_default = [3]

# lot_group_name for background
__AUG.lot_group_name_background = 'background_with_pricetags_and_shelfs'

# folder with source images
# if the string is empty, it will be filled from csv
__AUG.origin_storage_pass = '/media/disk200/lots/'

# # check data correctness
# assert (aug_cfg.trainval_ratio + aug_cfg.test_ratio == 1 and aug_cfg.trainval_ratio >= 0 and aug_cfg.trainval_ratio <= 1 and aug_cfg.test_ratio >= 0 and aug_cfg.test_ratio <= 1)
