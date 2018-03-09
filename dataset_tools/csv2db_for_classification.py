#!/usr/bin/python
# -*- coding: utf-8
import os
import shutil
import operator
import cv2
import numpy as np
from math import floor
import sys
import csv
import pandas as pd
from multiprocessing import Pool
import random
import time

from aug_func import distort_img_for_classification, make_just_resize, draw_rect
from augmentation_config_classification import aug_cfg

DEBUG = False
multiproc_flag = True

# output files paths
path_to_trainval = '/media/disk200/classification_dataset/train.txt'
path_to_val = '/media/disk200/classification_dataset/val.txt'
path_to_imgs = '/media/disk200/classification_dataset/imgs/'
path_to_sku_full_statistics_txt = '/media/disk200/classification_dataset/sku_statistics_list.txt'
path_to_sku_train_statistics_txt = '/media/disk200/classification_dataset/sku_train_statistics_list.txt'
path_to_sku_test_statistics_txt = '/media/disk200/classification_dataset/sku_test_statistics_list.txt'
path_to_amount_imgs = '/media/disk200/classification_dataset/amount_imgs.txt'
path_to_csv_with_non_existent_imgs = '/media/disk200/classification_dataset/non_existent_imgs.csv'
path_to_sku_order = '/media/disk200/classification_dataset/sku_order_' + time.strftime("%Y%m%d") + '.txt'


# functions
def replace_klass(src_string):
    def replace_string(src, condition, out):
        dst = out if src == condition else src
        return dst

    def replace_upak_po_6_to_upak(src_str):
        dst_str = src_str[:]
        dst_str = replace_string(dst_str, 'actimel_chernika_ezhevika_im_upak_po_6_bo',
                                 'actimel_chernika_ezhevika_im_upak_bo')
        dst_str = replace_string(dst_str, 'actimel_granat_im_upak_po_6_bo',
                                 'actimel_granat_im_upak_bo')
        dst_str = replace_string(dst_str, 'actimel_klubnika_upak_po_6_bo',
                                 'actimel_klubnika_upak_bo')
        dst_str = replace_string(dst_str, 'actimel_naturalnyy_im_upak_po_6_bo',
                                 'actimel_naturalnyy_im_upak_bo')
        return dst_str

    def replace_upak_po_8_to_upak(src_str):
        dst_str = src_str[:]
        dst_str = replace_string(dst_str, 'actimel_chernika_ezhevika_im_upak_po_8_bo',
                                 'actimel_chernika_ezhevika_im_upak_bo')
        dst_str = replace_string(dst_str, 'actimel_granat_im_upak_po_8_bo',
                                 'actimel_granat_im_upak_bo')
        dst_str = replace_string(dst_str, 'actimel_klubnika_upak_po_8_bo',
                                 'actimel_klubnika_upak_bo')
        dst_str = replace_string(dst_str, 'actimel_naturalnyy_im_upak_po_8_bo',
                                 'actimel_naturalnyy_im_upak_bo')
        dst_str = replace_string(dst_str, 'actimel_kivi_klubnika__upak_po_8_bo',
                                 'actimel_kivi_klubnika_upak_bo')
        dst_str = replace_string(dst_str, 'actimel_smorodina_malina_upak_po_8_bo',
                                 'actimel_smorodina_malina_upak_bo')
        dst_str = replace_string(dst_str, 'actimel_vishnya_chereshnya_imbir_upak_po_8_bo',
                                 'actimel_vishnya_chereshnya_imbir__upak_bo')
        dst_str = replace_string(dst_str, 'actimel_zemlyanika_shipovnik_upak_po_8_bo',
                                 'actimel_zemlyanika_shipovnik_upak_bo')
        return dst_str

    dst_string = src_string[:]
    dst_string = replace_upak_po_6_to_upak(dst_string)
    dst_string = replace_upak_po_8_to_upak(dst_string)
    return dst_string

def delete_folder_content(path_to_folder):
    if not os.path.isdir(path_to_folder):
        os.mkdir(path_to_folder)
    else:
        shutil.rmtree(path_to_folder)
        os.mkdir(path_to_folder)

# def query_yes_no(question, default="yes"):
#     """Ask a yes/no question via raw_input() and return their answer.
#
#     "question" is a string that is presented to the user.
#     "default" is the presumed answer if the user just hits <Enter>.
#         It must be "yes" (the default), "no" or None (meaning
#         an answer is required of the user).
#
#     The "answer" return value is True for "yes" or False for "no".
#     """
#     valid = {"yes": True, "y": True, "ye": True,
#              "no": False, "n": False}
#     if default is None:
#         prompt = " [y/n] "
#     elif default == "yes":
#         prompt = " [Y/n] "
#     elif default == "no":
#         prompt = " [y/N] "
#     else:
#         raise ValueError("invalid default answer: '%s'" % default)
#
#     while True:
#         sys.stdout.write(question + prompt)
#         choice = raw_input().lower()
#         if default is not None and choice == '':
#             return valid[default]
#         elif choice in valid:
#             return valid[choice]
#         else:
#             sys.stdout.write("Please respond with 'yes' or 'no' "
#                              "(or 'y' or 'n').\n")

print('\n')
print('num_out_files_for_one_img_in_trainval_using_augmentation:', aug_cfg.num_out_files_for_one_img_in_trainval_using_augmentation)
print('jpg_quality:', aug_cfg.jpg_quality)
print('visualization:', aug_cfg.visualization)
print('\n')

if aug_cfg.num_out_files_for_one_img_in_trainval_using_augmentation == 1:
    print('TURN ATTENTION TO "num_out_files_for_one_img_in_trainval_using_augmentation"', '=', aug_cfg.num_out_files_for_one_img_in_trainval_using_augmentation, '\n')
if aug_cfg.jpg_quality != 60:
    print('TURN ATTENTION TO "jpg_quality"', '=', aug_cfg.jpg_quality, '\n')
if aug_cfg.visualization:
    print('TURN ATTENTION TO "visualization"', '=', aug_cfg.visualization, '\n')

# answer_continue = query_yes_no('Are you sure you really want to continue?')
# if not answer_continue: sys.exit('Script execution has been stopped because of your desire')

# answer_removing_dataset = query_yes_no('Do you want to remove old dataset?')
answer_removing_dataset = True

# define test folders
# folder for train
train_folder_name = 'Train_Store/'
# folder for test
val_folder_name = 'Val_Store/'


if answer_removing_dataset:
    # delete old content
    delete_folder_content(path_to_imgs)
else:
    assert(False), 'This functional is not ready now' # ToDo add func


if answer_removing_dataset:
    # open files for writing
    f_trainval = open(path_to_trainval, 'w')
    f_val = open(path_to_val, 'w')
    f_sku_statistics = open(path_to_sku_full_statistics_txt, 'w')
    f_sku_train_statistics = open(path_to_sku_train_statistics_txt, 'w')
    f_sku_test_statistics = open(path_to_sku_test_statistics_txt, 'w')
    f_amount_imgs = open(path_to_amount_imgs, 'w')
    f_csv_non_existent_imgs = open(path_to_csv_with_non_existent_imgs, 'w')
    f_sku_order = open(path_to_sku_order, 'w')
    csv_writer_non_existent_imgs = csv.writer(f_csv_non_existent_imgs)

    sku_dict_statistics = {}
    sku_dict_statistics['__background__'] = 0
    sku_dict_train_statistics = {}
    sku_dict_train_statistics['__background__'] = 0
    sku_dict_test_statistics = {}
    sku_dict_test_statistics['__background__'] = 0
    non_existent_imgs_dict = {}
    imgs_amount_train_and_val_dict = {}
    imgs_amount_train_dict = {}
    imgs_amount_val_dict = {}

else:
    assert(False), 'To Do this functional'

with open(aug_cfg.csv_path, 'r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
    csv_rows = [row for row in csv_reader]
    if '->' in csv_rows[0][0]:
        col2ind = {column_name : id_column_name for id_column_name, column_name in enumerate(csv_rows[0])}
        del csv_rows[0]
    else:
        assert(False), 'To Do this functional'
        col2ind = {} #ToDo

def make_folder_if_not_exist(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

dst_folder_path_imgs_val = os.path.join(path_to_imgs, val_folder_name)
make_folder_if_not_exist(dst_folder_path_imgs_val)

dst_folder_path_imgs_train = os.path.join(path_to_imgs, train_folder_name)
make_folder_if_not_exist(dst_folder_path_imgs_train)

imgs_markup_dict = {}
background_imgs_markup_dict = {}
classes = []

for csv_row in csv_rows:
    project_name = csv_row[col2ind['project->name']].replace('"', '')
    lots_root_path = csv_row[col2ind['settings->name->origin_storage_pass']].replace('"', '')
    lot_storagepath = csv_row[col2ind['lot->storagepath']].replace('"', '')
    lot_name = csv_row[col2ind['lot->name']].replace('"', '')
    if 'backround' in lot_name:
        continue
    lot_types_id = csv_row[col2ind['lot->lot_types_id']].replace('"', '')
    lot_group_name = csv_row[col2ind['lot_group->name']].replace('"', '')
    photo_original_id = csv_row[col2ind['photo_original->id']].replace('"', '')
    photo_original_name = csv_row[col2ind['photo_original->name']].replace('"', '')
    task_image_status_is_treated = csv_row[col2ind['task_image_status->is_treated']].replace('"', '')
    klass_id = csv_row[col2ind['klass->id']].replace('"', '')
    klass_code = csv_row[col2ind['klass->code']].replace('"', '')
    klass_is_disabled = csv_row[col2ind['klass->is_disabled']].replace('"', '')
    if klass_is_disabled == '1':
        continue
    product_category_code = csv_row[col2ind['product_category->code']].replace('"', '')
    object_x1 = int(csv_row[col2ind['object_markup->x1']]) if csv_row[col2ind['object_markup->x1']] != '' else ''
    object_x2 = int(csv_row[col2ind['object_markup->x2']]) if csv_row[col2ind['object_markup->x1']] != '' else ''
    object_y1 = int(csv_row[col2ind['object_markup->y1']]) if csv_row[col2ind['object_markup->x1']] != '' else ''
    object_y2 = int(csv_row[col2ind['object_markup->y2']]) if csv_row[col2ind['object_markup->x1']] != '' else ''
    object_embedding_status = csv_row[col2ind['object_markup->embedding_status']]
    if object_embedding_status != '1':
        continue
    photo_original_is_collage = csv_row[col2ind['photo_original->is_collage']]
    src_img_path = os.path.join(lots_root_path, project_name, lot_storagepath, photo_original_name)
    if not os.path.isfile(src_img_path):
        non_existent_imgs_dict[photo_original_id] = src_img_path
        continue
    if not photo_original_id in imgs_markup_dict:
        imgs_markup_dict[photo_original_id] = {}
        imgs_markup_dict[photo_original_id]['project_name'] = project_name
        imgs_markup_dict[photo_original_id]['lots_root_path'] = lots_root_path
        imgs_markup_dict[photo_original_id]['lot_storagepath'] = lot_storagepath
        imgs_markup_dict[photo_original_id]['lot_name'] = lot_name
        imgs_markup_dict[photo_original_id]['lot_type'] = lot_types_id
        imgs_markup_dict[photo_original_id]['lot_group'] = lot_group_name
        imgs_markup_dict[photo_original_id]['photo_original_name'] = photo_original_name
        imgs_markup_dict[photo_original_id]['task_image_status_is_treated'] = task_image_status_is_treated
        imgs_markup_dict[photo_original_id]['src_img_path'] = src_img_path
        imgs_markup_dict[photo_original_id]['photo_original_is_collage'] = photo_original_is_collage
        imgs_markup_dict[photo_original_id]['facings'] = []
        cur_facing = {}
        cur_facing['klass_id'] = klass_id
        cur_facing['klass_code'] = klass_code
        cur_facing['klass_is_disabled'] = klass_is_disabled
        cur_facing['product_category_code'] = product_category_code
        cur_facing['x1'] = object_x1
        cur_facing['x2'] = object_x2
        cur_facing['y1'] = object_y1
        cur_facing['y2'] = object_y2
        cur_facing['embedding_status'] = object_embedding_status
        imgs_markup_dict[photo_original_id]['facings'].append(cur_facing)
        print('from csv:', src_img_path)
    else:
        cur_facing = {}
        cur_facing['klass_id'] = klass_id
        cur_facing['klass_code'] = klass_code
        cur_facing['klass_is_disabled'] = klass_is_disabled
        cur_facing['product_category_code'] = product_category_code
        cur_facing['x1'] = object_x1
        cur_facing['x2'] = object_x2
        cur_facing['y1'] = object_y1
        cur_facing['y2'] = object_y2
        cur_facing['embedding_status'] = object_embedding_status
        if not cur_facing in imgs_markup_dict[photo_original_id]['facings']:
            imgs_markup_dict[photo_original_id]['facings'].append(cur_facing)

    if 'dairy' in aug_cfg.csv_path:
        klass_code = replace_klass(klass_code)

    if lot_types_id != 'verif':
        rel_folder = train_folder_name
    else:
        rel_folder = val_folder_name

    if klass_code not in sku_dict_statistics:
        sku_dict_statistics[klass_code] = 0
        sku_dict_train_statistics[klass_code] = 0
        sku_dict_test_statistics[klass_code] = 0
    if object_embedding_status == 1 or object_embedding_status == '1':
        sku_dict_statistics[klass_code] += 1
        if lot_types_id == 'mark' or 'valid':
            sku_dict_train_statistics[klass_code] += 1
        if lot_types_id == 'verif':
            sku_dict_test_statistics[klass_code] += 1
    if photo_original_id not in imgs_amount_train_and_val_dict:
        imgs_amount_train_and_val_dict[photo_original_id] = True
    if photo_original_id not in imgs_amount_train_dict and lot_types_id == 'mark' or 'valid':
        imgs_amount_train_dict[photo_original_id] = True
    if photo_original_id not in imgs_amount_val_dict and lot_types_id == 'verif':
        imgs_amount_val_dict[photo_original_id] = True

# save statistics
sku_list_statistics = sorted(sku_dict_statistics.items(), key=operator.itemgetter(1))
for key, value in sku_list_statistics:
    f_sku_statistics.write(key + ':' + str(value) + '\n')

sku_list_train_statistics = sorted(sku_dict_train_statistics.items(), key=operator.itemgetter(1))
for key, value in sku_list_train_statistics:
    f_sku_train_statistics.write(key + ':' + str(value) + '\n')

sku_list_test_statistics = sorted(sku_dict_test_statistics.items(), key=operator.itemgetter(1))
for key, value in sku_list_test_statistics:
    f_sku_test_statistics.write(key + ':' + str(value) + '\n')


# save current order

sku_order_list = []
for key, value in sku_dict_statistics.items():
    sku_order_list.append(key)

random.seed(10)
random.shuffle(sku_order_list)

sku_order_list = [[id_sku, sku] for id_sku, sku in enumerate(sku_order_list)]
sku_to_id = {}


for id_sku, sku in sku_order_list:
    sku_to_id[sku] = id_sku


sku_order_list_lines = [str(id_sku) + ':' + sku + '\n' for id_sku, sku in sku_order_list]
f_sku_order.writelines(sku_order_list_lines)

f_amount_imgs.write('src_train_imgs_amount:' + str(len(imgs_amount_train_dict)) + '\n')
f_amount_imgs.write('src_val_imgs_amount:' + str(len(imgs_amount_val_dict)) + '\n')
f_amount_imgs.write('src_imgs_amount:' + str(len(imgs_amount_train_and_val_dict)) + '\n')

for key, value in non_existent_imgs_dict.items():
    csv_writer_non_existent_imgs.writerow([key, value])

print('Source train images amount is', str(len(imgs_amount_train_dict)) )
print('Source test images amount is', str(len(imgs_amount_val_dict)) )
print('Source train&test images amount is', str(len(imgs_amount_train_and_val_dict)) )

# close files
f_sku_statistics.close()
f_sku_train_statistics.close()
f_sku_test_statistics.close()
f_amount_imgs.close()
f_csv_non_existent_imgs.close()
f_sku_order.close()

imgs_markup_list = []
for photo_original_id, imgs_markup_dict_item in imgs_markup_dict.items():
    cur_dict = imgs_markup_dict_item.copy()
    cur_dict['photo_original_id'] = photo_original_id
    imgs_markup_list.append(cur_dict)

imgs_markup_list = sorted(imgs_markup_list, key=lambda item: item['src_img_path'])


def crop_and_augment_img(imgs_markup_list_item):
    def save_img(src_cropped_img, src_facing_dict, src_id_aug=0, cur_coors=None):
        class_code = src_facing_dict['klass_code']
        img_name_without_ext = src_facing_dict['img_name_without_ext']
        rel_folder = src_facing_dict['rel_folder_for_saving']
        id_facing_dict = src_facing_dict['id_facing_dict_for_saving']
        # if cur_coors == None:
        dst_img_name = img_name_without_ext + '_' + str(id_facing_dict) + '_' + str(src_id_aug) + '.jpg'
        # else:
        #     x1, y1, x2, y2 = cur_coors
        #     x1, y1, x2, y2 = '_' + str(x1) + '_', '_' + str(y1) + '_', '_' + str(x2) + '_', '_' + str(y2),
        #     dst_img_name = img_name_without_ext + '_' + str(id_facing_dict) + '_' + str(src_id_aug) + '_' + x1 + y1 + x2 + y2+ '.jpg'
        folder_id_class_code_and_class_code = str(sku_to_id[class_code]) + '_' + class_code
        dst_folder = os.path.join(path_to_imgs, rel_folder, folder_id_class_code_and_class_code)
        dst_path_cropped_img = os.path.join(dst_folder, dst_img_name)
        height, width = src_cropped_img.shape[:2]
        if height == 0 or width == 0:
            print('HERE IS BAD MARK')
            return
        try:
            os.makedirs(dst_folder)
        except:
            pass
        cv2.imwrite(dst_path_cropped_img, src_cropped_img, [int(cv2.IMWRITE_JPEG_QUALITY), aug_cfg.jpg_quality])

    #read info about img and markup
    project_name = imgs_markup_list_item['project_name']
    lots_root_path = imgs_markup_list_item['lots_root_path']
    lot_storagepath = imgs_markup_list_item['lot_storagepath']
    lot_name = imgs_markup_list_item['lot_name']
    lot_type = imgs_markup_list_item['lot_type']
    lot_group = imgs_markup_list_item['lot_group']
    photo_original_name = imgs_markup_list_item['photo_original_name']
    photo_original_id = imgs_markup_list_item['photo_original_id']
    photo_original_is_collage = imgs_markup_list_item['photo_original_is_collage']
    task_image_status_is_treated = imgs_markup_list_item['task_image_status_is_treated']
    src_img_path = imgs_markup_list_item['src_img_path']
    print(src_img_path, 'is being processed...')
    img_name_without_ext = src_img_path[src_img_path.rfind('/')+1:src_img_path.rfind('.')]
    facings = imgs_markup_list_item['facings']

    # read img
    src_img_full = cv2.imread(src_img_path)
    src_img_full_height, src_img_full_width = src_img_full.shape[:2]

    if lot_type != 'verif':
        rel_folder = train_folder_name
    else:
        rel_folder = val_folder_name

    for id_facing_dict, facing in enumerate(facings):
        src_x1 = max(0, int(facing['x1']))
        src_y1 = max(0, int(facing['y1']))
        src_x2 = min(src_img_full_width - 1, int(facing['x2']))
        src_y2 = min(src_img_full_height - 1, int(facing['y2']))
        if aug_cfg.visualization:
            cv2.rectangle(src_img_full, (src_x1, src_y1), (src_x2, src_y2), (0, 255, 0), 2)
        src_width = src_x2 - src_x1
        src_height = src_y2 - src_y1
        if src_height == 0 or src_width == 0:
            continue
        src_max_side = max(src_width, src_height)
        border_x = int(round(aug_cfg.border_in_percents * 0.01 * src_max_side) + (src_max_side - src_width) / 2.0)
        border_y = int(round(aug_cfg.border_in_percents * 0.01 * src_max_side) + (src_max_side - src_height) / 2.0)
        bordered_x1 = src_x1 - border_x
        bordered_y1 = src_y1 - border_y
        bordered_x2 = src_x2 + border_x
        bordered_y2 = src_y2 + border_y
        if aug_cfg.border_from_src_img:
            cur_x1 = max(0, bordered_x1)
            cur_y1 = max(0, bordered_y1)
            cur_x2 = min(src_img_full_width - 1, bordered_x2)
            cur_y2 = min(src_img_full_height - 1, bordered_y2)
            cur_coors = [cur_x1, cur_y1, cur_x2, cur_y2]
            cur_img = src_img_full[cur_y1:cur_y2, cur_x1:cur_x2]
            left_padding = 0
            right_padding = 0
            top_padding = 0
            bottom_padding = 0
            if bordered_x1 < 0:
                left_padding = abs(bordered_x1)
            if bordered_y1 < 0:
                top_padding = abs(bordered_y1)
            if bordered_x2 > src_img_full_width - 1:
                right_padding = bordered_x2 - (src_img_full_width - 1)
            if bordered_y2 > src_img_full_height - 1:
                bottom_padding = bordered_y2 - (src_img_full_height - 1)
            if left_padding != 0 or right_padding != 0 or top_padding != 0 or bottom_padding != 0:
                cur_img = cv2.copyMakeBorder(cur_img, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_REPLICATE)
        else:
            left_padding, right_padding, top_padding, bottom_padding = border_x, border_x, border_y, border_y
            cur_img = src_img_full[src_y1:src_y2, src_x1:src_x2]
            if left_padding != 0 or right_padding != 0 or top_padding != 0 or bottom_padding != 0:
                cur_img = cv2.copyMakeBorder(cur_img, top_padding, bottom_padding, left_padding, right_padding,
                                             cv2.BORDER_REPLICATE)

        bordered_height, bordered_width = cur_img.shape[:2]
        if bordered_height > bordered_width:
            diff_shape = bordered_height - bordered_width
            if np.random.choice([0,1]):
                cur_img = cv2.copyMakeBorder(cur_img, 0, 0, diff_shape, 0, cv2.BORDER_REPLICATE)
            else:
                cur_img = cv2.copyMakeBorder(cur_img, 0, 0, 0, diff_shape, cv2.BORDER_REPLICATE)
        if bordered_height < bordered_width:
            diff_shape = bordered_width - bordered_height
            if np.random.choice([0,1]):
                cur_img = cv2.copyMakeBorder(cur_img, diff_shape, 0, 0, 0, cv2.BORDER_REPLICATE)
            else:
                cur_img = cv2.copyMakeBorder(cur_img, 0, diff_shape, 0, 0, cv2.BORDER_REPLICATE)

        if aug_cfg.resize:
            cur_img = cv2.resize(cur_img, (aug_cfg.resize_side, aug_cfg.resize_side))

        facing['img_name_without_ext'] = img_name_without_ext
        facing['rel_folder_for_saving'] = rel_folder
        facing['id_facing_dict_for_saving'] = id_facing_dict
        if lot_type != 'verif':
            for id_aug in range(aug_cfg.num_out_files_for_one_img_in_trainval_using_augmentation):
                if id_aug != 0:
                    cur_img = distort_img_for_classification(cur_img)
                save_img(cur_img, facing, id_aug, cur_coors)
        else:
            save_img(cur_img, facing, 0, cur_coors)


import time
start_time = time.time()
if multiproc_flag:
    pool = Pool()
    pool.map(crop_and_augment_img, imgs_markup_list)
    pool.close()
    pool.join()
else:
#     map(crop_and_augment_img, imgs_markup_list)
    for lst in imgs_markup_list:
        crop_and_augment_img(lst)
end_time = time.time()
print(end_time-start_time, 'seconds')

def make_dataset_list(rel_folder):
    def get_ind_from_file_name(file_name):
        abs_dst_folder_name = file_name[:file_name.rfind('/')]
        rel_dst_folder_name = abs_dst_folder_name[abs_dst_folder_name.rfind('/')+1:]
        ind = rel_dst_folder_name[:rel_dst_folder_name.find('_')]
        return ind

    root_folder = os.path.join(path_to_imgs, rel_folder)
    file_list = []
    file_size = 0
    folder_count = 0
    for root, subFolders, files in os.walk(root_folder):
        folder_count += len(subFolders)
        for file in files:
            f = os.path.join(root, file)
            file_size = file_size + os.path.getsize(f)
            file_list.append(f)
    fileList = [file_name+' '+get_ind_from_file_name(file_name)+'\n' for file_name in file_list]
    return fileList

train_file_list = make_dataset_list('Train_Store/')
random.shuffle(train_file_list)
val_file_list = make_dataset_list('Val_Store/')
random.shuffle(train_file_list)
f_trainval.writelines(make_dataset_list(train_folder_name))
f_val.writelines(make_dataset_list(val_folder_name))

f_trainval.close()
f_val.close()