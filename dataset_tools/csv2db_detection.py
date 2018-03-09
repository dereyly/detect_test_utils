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

from aug_func import distort_img, make_just_resize, draw_rect
from augmentation_config import aug_cfg
os.chdir('/home/dereyly/progs/pva-faster-rcnn/data')
DEBUG = False
multiproc_flag = True

# output files paths
path_to_trainval = '../data/VOCdevkit5180/VOC5180/ImageSets/Main/trainval.txt'
path_to_test = '../data/VOCdevkit5180/VOC5180/ImageSets/Main/test.txt'
path_to_xmls = '../data/VOCdevkit5180/VOC5180/Annotations/'
path_to_imgs = '../data/VOCdevkit5180/VOC5180/JPEGImages/'
path_to_sku_full_statistics_txt = '../data/VOCdevkit5180/VOC5180/sku_statistics_list.txt'
path_to_sku_train_statistics_txt = '../data/VOCdevkit5180/VOC5180/sku_train_statistics_list.txt'
path_to_sku_test_statistics_txt = '../data/VOCdevkit5180/VOC5180/sku_test_statistics_list.txt'
path_to_category_statistics_txt = '../data/VOCdevkit5180/VOC5180/category_statistics_list.txt'
path_to_category_list_inds_txt = '../data/VOCdevkit5180/VOC5180/category_list_inds.txt'
path_to_amount_imgs = '../data/VOCdevkit5180/VOC5180/amount_imgs.txt'
path_to_sku_space_package_type_list = '../data/VOCdevkit5180/VOC5180/sku_space_package_type_list.txt' # one of the main files for training and testings
path_to_csv_with_non_existent_imgs = '../data/VOCdevkit5180/VOC5180/non_existent_imgs.csv'
path_to_csv_with_lots_names = '../data/VOCdevkit5180/VOC5180/lots_names.csv'

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

def replace_package_type(src_string):
    def replace_string(src, condition, out):
        dst = out if src == condition else src
        return dst

    def replace_to_bo(src_str):
        dst_str = src_str[:]
        dst_str = replace_string(dst_str, 'pb', 'bo')
        return dst_str

    def replace_to_cbl(src_str):
        dst_str = src_str[:]
        dst_str = replace_string(dst_str, 'cfi', 'cbl')
        dst_str = replace_string(dst_str, 'tb', 'cbl')
        dst_str = replace_string(dst_str, 'tbc', 'cbl')
        dst_str = replace_string(dst_str, 'tbq', 'cbl')
        dst_str = replace_string(dst_str, 'tbs', 'cbl')
        dst_str = replace_string(dst_str, 'tc', 'cbl')
        dst_str = replace_string(dst_str, 'tt', 'cbl')
        return dst_str

    def replace_to_gl(src_str):
        dst_str = src_str[:]
        dst_str = replace_string(dst_str, 'mg', 'gl')
        dst_str = replace_string(dst_str, 'mgm', 'gl')
        return dst_str

    def replace_to_mc(src_str):
        dst_str = src_str[:]
        dst_str = replace_string(dst_str, 'mci', 'mc')
        return dst_str

    def replace_to_vac(src_str):
        dst_str = src_str[:]
        dst_str = replace_string(dst_str, 'mp', 'vac')
        return dst_str

    dst_string = src_string[:]
    dst_string = replace_to_bo(dst_string)
    dst_string = replace_to_cbl(dst_string)
    dst_string = replace_to_gl(dst_string)
    dst_string = replace_to_mc(dst_string)
    dst_string = replace_to_vac(dst_string)
    return dst_string

def divide_img(src_img, object_markup_data): #ToDo
    separate_parts_smaller_side = np.random.choice(aug_cfg.separate_parts_smaller_side_default)

    def filter_ext_coors(h_up, h_down, w_left, w_right):
        object_markup_data_filtered = []
        for category_name, sku_name, x1, x2, y1, y2, object_id, embedding_status in object_markup_data:
            if y1 >= h_up and y2 <= h_down and x1 >= w_left and x2 <= w_right:
                object_markup_data_filtered += [
                    [category_name, sku_name, x1 - w_left, x2 - w_left, y1 - h_up, y2 - h_up, object_id, embedding_status]]
            elif y2 < h_up or y1 > h_down or x1 < w_left or x2 > w_right:
                continue
            else:
                img_area = abs(h_down - h_up) * abs(w_right - w_left)
                facing_area = abs(x2 - x1) * abs(y2 - y1)

                if facing_area == 0:
                    print 'Facing area is 0.'
                    continue

                overlap_x1 = max(x1, w_left)
                overlap_x2 = min(x2, w_right)

                overlap_y1 = max(y1, h_up)
                overlap_y2 = min(y2, h_down)

                overlap_area = abs(overlap_x2 - overlap_x1) * abs(overlap_y2 - overlap_y1)
                # print 'Area ', overlap_area, img_area
                if overlap_area >= img_area:
                    print "Here facing covers whole image. Consider changing number of image parts."

                elif (float(overlap_area) / facing_area) >= aug_cfg.part_for_save:
                    object_markup_data_filtered += [[category_name, sku_name, overlap_x1 - w_left, overlap_x2 - w_left,
                                                     overlap_y1 - h_up, overlap_y2 - h_up, object_id, embedding_status]]

        return object_markup_data_filtered

    src_h, src_w = src_img.shape[:2]

    # Check if height, width and separations are valid values
    if src_h <= 0 or src_w <= 0 or aug_cfg.separate_parts_bigger_side <= 0 or separate_parts_smaller_side <= 0:
        raise ValueError

    # Assign bigger and smaller sides' partitions
    if src_h > src_w:
        h_separate_parts = aug_cfg.separate_parts_bigger_side
        w_separate_parts = separate_parts_smaller_side
    elif src_h == src_w:
        h_separate_parts = aug_cfg.separate_parts_bigger_side
        w_separate_parts = aug_cfg.separate_parts_bigger_side
    else:
        h_separate_parts = separate_parts_smaller_side
        w_separate_parts = aug_cfg.separate_parts_bigger_side

    if DEBUG:
        print h_separate_parts, w_separate_parts, w_separate_parts*h_separate_parts
    # Creating horizontal and vertical cutting lines
    h_lines = [int(floor(src_h * i / (float(h_separate_parts) + 1))) for i in xrange(h_separate_parts + 1)]
    h_lines += [src_h]
    w_lines = [int(floor(src_w * i / (float(w_separate_parts) + 1))) for i in xrange(w_separate_parts + 1)]
    w_lines += [src_w]

    src_imgs = []
    object_markup_data_for_imgs = []
    for h in xrange(h_separate_parts):
        for w in range(w_separate_parts):
            temp_coors = filter_ext_coors(h_lines[h], h_lines[h + 2], w_lines[w], w_lines[w + 2])
            if len(temp_coors) > 0:
                temp_img = src_img[h_lines[h]:h_lines[h + 2], w_lines[w]:w_lines[w + 2]]
                if temp_img.shape[0] == 0 or temp_img.shape[1] == 0:
                    print 'Invalid image dimension: ', temp_img.shape
                    continue
                src_imgs.append(temp_img)
                object_markup_data_for_imgs.append(temp_coors)
    if DEBUG:
        if len(object_markup_data_for_imgs) == 0:
            print object_markup_data
            cv2.imshow('some', src_img)
            cv2.waitKey()
    return src_imgs, object_markup_data_for_imgs

def delete_folder_content(path_to_folder):
    if not os.path.isdir(path_to_folder):
        os.mkdir(path_to_folder)
    else:
        shutil.rmtree(path_to_folder)
        os.mkdir(path_to_folder)

def add_background(img, bg_list_names, chance_percent=aug_cfg.background_percent):
    percent_for_choice = np.random.random() * 99
    if chance_percent <= percent_for_choice:
        return img
    else:
        h1, w1 = img.shape[:2]
        bg_im_path = np.random.choice(bg_list_names)
        bg_im = cv2.imread(bg_im_path)
        h2, w2 = bg_im.shape[:2]
        new_w2 = int(1.0 * h1 / h2 * w2)
        bg_im_resized = cv2.resize(bg_im, (new_w2, h1), interpolation=cv2.INTER_AREA)
        im_with_bg_im = np.concatenate((img, bg_im_resized), axis=1)
        return im_with_bg_im

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

print '\n'
# print 'trainval_ratio:', aug_cfg.trainval_ratio
# print 'test_ratio:', aug_cfg.test_ratio
print 'IMG_MAX_SIZE:', aug_cfg.IMG_MAX_SIZE
print 'num_out_files_for_one_img_in_trainval_using_augmentation:', aug_cfg.num_out_files_for_one_img_in_trainval_using_augmentation
print 'background_percent:', aug_cfg.background_percent
print 'jpg_quality:', aug_cfg.jpg_quality
print 'visualization:', aug_cfg.visualization
print 'part_for_save:', aug_cfg.part_for_save
print 'separate_parts_bigger_side', aug_cfg.separate_parts_bigger_side
print 'separate_parts_smaller_side_default', aug_cfg.separate_parts_smaller_side_default
print '\n'

# if aug_cfg.trainval_ratio != 0.8:
#     print 'TURN ATTENTION TO "trainval_ratio"', '=', aug_cfg.trainval_ratio, '\n'
# if aug_cfg.test_ratio != 0.2:
#     print 'TURN ATTENTION TO "test_ratio"', '=', aug_cfg.test_ratio, '\n'
if aug_cfg.IMG_MAX_SIZE != 2432:
    print 'TURN ATTENTION TO "test_ratio"', '=', aug_cfg.IMG_MAX_SIZE, '\n'
if aug_cfg.num_out_files_for_one_img_in_trainval_using_augmentation == 1:
    print 'TURN ATTENTION TO "num_out_files_for_one_img_in_trainval_using_augmentation"', '=', aug_cfg.num_out_files_for_one_img_in_trainval_using_augmentation, '\n'
if aug_cfg.background_percent == 0:
    print 'TURN ATTENTION TO "background_percent"', '=', aug_cfg.background_percent, '\n'
if aug_cfg.jpg_quality != 60:
    print 'TURN ATTENTION TO "jpg_quality"', '=', aug_cfg.jpg_quality, '\n'
if aug_cfg.visualization:
    print 'TURN ATTENTION TO "visualization"', '=', aug_cfg.visualization, '\n'
if aug_cfg.part_for_save != 0.7:
    print 'TURN ATTENTION TO "part_for_save"', '=', aug_cfg.part_for_save, '\n'
if aug_cfg.separate_parts_bigger_side == 1:
    print 'TURN ATTENTION TO "separate_parts_bigger_side"', aug_cfg.separate_parts_bigger_side, '\n'
if aug_cfg.separate_parts_smaller_side_default == 1:
    print 'TURN ATTENTION TO "separate_parts_smaller_side_default"', aug_cfg.separate_parts_smaller_side_default, '\n'

answer_continue = query_yes_no('Are you sure you really want to continue?')
if not answer_continue: sys.exit('Script execution has been stopped because of your desire')

answer_removing_dataset = query_yes_no('Do you want to remove old dataset?')

# define test folders
# folder for train
train_folder_name = 'Train_Store/'
# folder for test
val_folder_name = 'Val_Store/'


if answer_removing_dataset:
    # delete old content
    delete_folder_content(path_to_imgs)
    delete_folder_content(path_to_xmls)

    delete_folder_content(path_to_xmls + val_folder_name)
    delete_folder_content(path_to_imgs + val_folder_name)
else:
    assert(False), 'This functional is not ready now' # ToDo add func


if answer_removing_dataset:
    # open files for writing
    f_trainval = open(path_to_trainval, 'w')
    f_test = open(path_to_test, 'w')
    f_sku_statistics = open(path_to_sku_full_statistics_txt, 'w')
    f_sku_train_statistics = open(path_to_sku_train_statistics_txt, 'w')
    f_sku_test_statistics = open(path_to_sku_test_statistics_txt, 'w')
    f_category_statistics = open(path_to_category_statistics_txt, 'w')
    f_sku_space_package_type = open(path_to_sku_space_package_type_list, 'w')
    f_amount_imgs = open(path_to_amount_imgs, 'w')
    f_csv_non_existent_imgs = open(path_to_csv_with_non_existent_imgs, 'wb')
    csv_writer_non_existent_imgs = csv.writer(f_csv_non_existent_imgs)

    sku_dict_statistics = {}
    sku_dict_statistics['__background__'] = 0
    sku_dict_train_statistics = {}
    sku_dict_train_statistics['__background__'] = 0
    sku_dict_test_statistics = {}
    sku_dict_test_statistics['__background__'] = 0
    package_type_dict_statistics = {}
    package_type_dict_statistics['__background__'] = 0
    sku_space_package_type_dict = {}
    sku_space_package_type_dict['__background__'] = '__background__'
    non_existent_imgs_dict = {}
    imgs_amount_train_and_val_dict = {}
    imgs_amount_train_dict = {}
    imgs_amount_val_dict = {}
else:
    assert(False), 'To Do this functional'

with open(aug_cfg.csv_path, 'rb') as csv_file:
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
dst_folder_path_xmls_val = os.path.join(path_to_xmls, val_folder_name)
make_folder_if_not_exist(dst_folder_path_imgs_val)
make_folder_if_not_exist(dst_folder_path_xmls_val)

dst_folder_path_imgs_train = os.path.join(path_to_imgs, train_folder_name)
dst_folder_path_xmls_train = os.path.join(path_to_xmls, train_folder_name)
make_folder_if_not_exist(dst_folder_path_imgs_train)
make_folder_if_not_exist(dst_folder_path_xmls_train)

imgs_markup_dict = {}
background_imgs_markup_dict = {}
lots_names_dict = {}

for csv_row in csv_rows:
    project_name = csv_row[col2ind['project->name']].replace('"', '')
    lots_root_path = csv_row[col2ind['settings->name->origin_storage_pass']].replace('"', '')
    if aug_cfg.origin_storage_pass:
        lots_root_path = aug_cfg.origin_storage_pass
    lot_storagepath = csv_row[col2ind['lot->storagepath']].replace('"', '')
    lot_name = csv_row[col2ind['lot->name']].replace('"', '')
    if lot_name not in lots_names_dict:
        lots_names_dict[lot_name] = True
    lot_types_id = csv_row[col2ind['lot->lot_types_id']].replace('"', '')
    lot_group_name = csv_row[col2ind['lot_group->name']].replace('"', '')
    photo_original_id = csv_row[col2ind['photo_original->id']].replace('"', '')
    photo_original_name = csv_row[col2ind['photo_original->name']].replace('"', '')
    task_image_status_is_treated = csv_row[col2ind['task_image_status->is_treated']].replace('"', '')
    klass_id = csv_row[col2ind['klass->id']].replace('"', '')
    klass_package_type_code = csv_row[col2ind['klass->package_type']].replace('"', '')
    klass_code = csv_row[col2ind['klass->code']].replace('"', '')
    klass_is_disabled = csv_row[col2ind['klass->is_disabled']].replace('"', '')
    product_category_code = csv_row[col2ind['product_category->code']].replace('"', '')
    object_x1 = int(csv_row[col2ind['object_markup->x1']]) if csv_row[col2ind['object_markup->x1']] != '' else ''
    object_x2 = int(csv_row[col2ind['object_markup->x2']]) if csv_row[col2ind['object_markup->x1']] != '' else ''
    object_y1 = int(csv_row[col2ind['object_markup->y1']]) if csv_row[col2ind['object_markup->x1']] != '' else ''
    object_y2 = int(csv_row[col2ind['object_markup->y2']]) if csv_row[col2ind['object_markup->x1']] != '' else ''
    object_embedding_status = csv_row[col2ind['object_markup->embedding_status']]
    photo_original_is_collage = csv_row[col2ind['photo_original->is_collage']]
    src_img_path = os.path.join(lots_root_path, project_name, lot_storagepath, photo_original_name)
    if 'dairy' in aug_cfg.project_name:
        klass_code = replace_klass(klass_code)
        klass_package_type_code = replace_package_type(klass_package_type_code)
    if not os.path.isfile(src_img_path):
        non_existent_imgs_dict[photo_original_id] = src_img_path
        print src_img_path, 'IS NOT EXIST !!!'
        continue
    if lot_group_name != aug_cfg.lot_group_name_background:
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
            cur_facing['klass_package_type_code'] = klass_package_type_code
            cur_facing['klass_code'] = klass_code
            cur_facing['klass_is_disabled'] = klass_is_disabled
            cur_facing['product_category_code'] = product_category_code
            cur_facing['x1'] = object_x1
            cur_facing['x2'] = object_x2
            cur_facing['y1'] = object_y1
            cur_facing['y2'] = object_y2
            cur_facing['embedding_status'] = object_embedding_status
            imgs_markup_dict[photo_original_id]['facings'].append(cur_facing)
            print 'from csv:', src_img_path
        else:
            cur_facing = {}
            cur_facing['klass_id'] = klass_id
            cur_facing['klass_package_type_code'] = klass_package_type_code
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

        if lot_types_id != 'verif':
            rel_folder = train_folder_name
        else:
            rel_folder = val_folder_name

        dst_folder_path_xml = os.path.join(path_to_xmls, rel_folder, project_name, lot_storagepath)
        dst_folder_path_img = os.path.join(path_to_imgs, rel_folder, project_name, lot_storagepath)
        make_folder_if_not_exist(dst_folder_path_xml)
        make_folder_if_not_exist(dst_folder_path_img)

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
        if klass_package_type_code not in package_type_dict_statistics:
            package_type_dict_statistics[klass_package_type_code] = 0
        if object_embedding_status == 1 or object_embedding_status == '1':
            package_type_dict_statistics[klass_package_type_code] += 1
        if klass_code not in sku_space_package_type_dict:
            sku_space_package_type_dict[klass_code] = klass_package_type_code
        if photo_original_id not in imgs_amount_train_and_val_dict:
            imgs_amount_train_and_val_dict[photo_original_id] = True
        if photo_original_id not in imgs_amount_train_dict and lot_types_id == 'mark' or 'valid':
            imgs_amount_train_dict[photo_original_id] = True
        if photo_original_id not in imgs_amount_val_dict and lot_types_id == 'verif':
            imgs_amount_val_dict[photo_original_id] = True
    else:
        if not src_img_path in background_imgs_markup_dict:
            background_imgs_markup_dict[src_img_path] = True
            print 'bg from csv:', src_img_path

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

category_list_statistics = sorted(package_type_dict_statistics.items(), key=operator.itemgetter(1))
for key, value in category_list_statistics:
    f_category_statistics.write(key + ':' + str(value) + '\n')

f_amount_imgs.write('src_train_imgs_amount:' + str(len(imgs_amount_train_dict)) + '\n')
f_amount_imgs.write('src_val_imgs_amount:' + str(len(imgs_amount_val_dict)) + '\n')
f_amount_imgs.write('src_imgs_amount:' + str(len(imgs_amount_train_and_val_dict)) + '\n')

# load previous order and save current order in sku_space_package_type_txt
with open(aug_cfg.previous_sku_space_package_type_txt, 'r') as f:
    previous_sku_space_package_type_lines = f.readlines()

transit_sku_space_package_type_lines = []
for key, value in sku_space_package_type_dict.items():
    transit_sku_space_package_type_lines.append(str(key) + ' ' + str(value) + '\n')

import random
random.seed(10)
random.shuffle(transit_sku_space_package_type_lines)

def find_common_strings(strings_1, strings_2):
    return [element for element in strings_1 if element in strings_2]

def find_deleted_strings(strings_old_src, strings_curent_src):
    return [element for element in strings_old_src if element not in strings_curent_src]

def find_added_strings(strings_old_src, strings_curent_src):
    return [element for element in strings_curent_src if element not in strings_old_src]

common_strings = find_common_strings(previous_sku_space_package_type_lines, transit_sku_space_package_type_lines)
deleted_strings = find_deleted_strings(previous_sku_space_package_type_lines, transit_sku_space_package_type_lines)
added_strings = find_added_strings(previous_sku_space_package_type_lines, transit_sku_space_package_type_lines)

print 'common_strings'
print common_strings
print 'deleted_strings'
print sorted(deleted_strings)
print 'added_strings'
print sorted(added_strings)

assert(len(deleted_strings) <= len(added_strings))

sku_space_package_type_lines = []
for line in previous_sku_space_package_type_lines:
    if line not in deleted_strings:
        sku_space_package_type_lines.append(line)
    else:
        sku_space_package_type_lines.append(added_strings[0])
        del added_strings[0]
sku_space_package_type_lines += added_strings

print '---'
print(len(previous_sku_space_package_type_lines))
print previous_sku_space_package_type_lines
print(len(transit_sku_space_package_type_lines))
print transit_sku_space_package_type_lines
print(len(sku_space_package_type_lines))
print sku_space_package_type_lines

f_sku_space_package_type.writelines(sku_space_package_type_lines)

with open(path_to_csv_with_lots_names, 'wb') as csv_file_lots_name:
    csv_writer_lots_name = csv.writer(csv_file_lots_name, delimiter=',', quotechar='|')
    for key, value in lots_names_dict.iteritems():
        csv_writer_lots_name.writerow([key])

for key, value in non_existent_imgs_dict.iteritems():
    csv_writer_non_existent_imgs.writerow([key, value])


print 'Source train images amount is', str(len(imgs_amount_train_dict))
print 'Source test images amount is', str(len(imgs_amount_val_dict))
print 'Source train&test images amount is', str(len(imgs_amount_train_and_val_dict))

# close files
f_sku_statistics.close()
f_category_statistics.close()
f_sku_space_package_type.close()
f_amount_imgs.close()

imgs_markup_list = []
for photo_original_id, imgs_markup_dict_item in imgs_markup_dict.iteritems():
    cur_dict = imgs_markup_dict_item.copy()
    cur_dict['photo_original_id'] = photo_original_id
    imgs_markup_list.append(cur_dict)

imgs_markup_list = sorted(imgs_markup_list, key=lambda item: item['src_img_path'])

background_imgs_markup_list = []
for key, value in background_imgs_markup_dict.iteritems():
    background_imgs_markup_list.append(key)

def augment_and_save_img_and_save_xml(imgs_markup_list_item):
    def save_img_and_xml(dst_img, trans_tl_and_br_coors, objects_markup_data_src, dst_img_and_xml_info):
        photo_original_name_without_ext = dst_img_and_xml_info['photo_original_name_without_ext']
        rel_dst_img_name_without_ext = dst_img_and_xml_info['rel_dst_img_name_without_ext']
        full_dst_img_name_without_ext = dst_img_and_xml_info['full_dst_img_name_without_ext']
        full_dst_img_name_with_ext = dst_img_and_xml_info['full_dst_img_name_with_ext']
        full_dst_xml_name_with_ext = dst_img_and_xml_info['full_dst_xml_name_with_ext']
        boxes_list = np.array(
            [np.array(list(coors[0]) + list(coors[1])) for coors in trans_tl_and_br_coors])
        dst_img_width = dst_img.shape[1]
        dst_img_height = dst_img.shape[0]
        src_categ_and_sku = [[object_markup_rec[0], object_markup_rec[1]] for
                             object_markup_rec in objects_markup_data_src]
        photo_original_id = dst_img_and_xml_info['photo_original_id']
        photo_original_is_collage = dst_img_and_xml_info['photo_original_is_collage']
        photo_original_is_collage = photo_original_is_collage if photo_original_is_collage is not None and photo_original_is_collage != '' else 'none'

        # create xml file
        f_xml = open(full_dst_xml_name_with_ext, 'w')
        f_xml.write('<annotation>\n')
        f_xml.write('<folder>' + rel_folder[:-1] + '</folder>\n')
        f_xml.write('<filename>' + photo_original_name_without_ext + '</filename>\n')
        f_xml.write('<source>\n')
        f_xml.write('<database>Deep Food</database>\n')
        f_xml.write('<annotation>' + aug_cfg.project_name + '</annotation>\n')
        f_xml.write('<photo_id>' + photo_original_id.encode('utf8') + '</photo_id>\n')
        f_xml.write('<photo_original_is_collage>' + photo_original_is_collage.encode('utf8') + '</photo_original_is_collage>\n')
        f_xml.write('</source>\n')
        f_xml.write('<size>')
        f_xml.write('<width>' + str(dst_img_width) + '</width>')  # not used directly
        f_xml.write('<height>' + str(dst_img_height) + '</height>')  # not used directly
        f_xml.write('<depth>3</depth>')
        f_xml.write('</size>')
        f_xml.write('<segmented>0</segmented>\n')

        for idx, object_markup_rec in enumerate(objects_markup_data_src):
            if not object_markup_rec[0]:
                continue
            package_type, sku_name, x1_old, x2_old, y1_old, y2_old, object_id, embedding_status = object_markup_rec
            embedding_status = embedding_status if embedding_status is not None else '0'
            embedding_status = embedding_status if embedding_status else '0'
            x1, y1, x2, y2 = boxes_list[idx]
            package_type = package_type if package_type is not None else u'none'
            sku_name = replace_klass(sku_name) if sku_name is not None else u'none'
            f_xml.write('<object>\n')
            f_xml.write('<name>' + sku_name.encode('utf8') + '</name>\n')
            f_xml.write('<pose>Centre</pose>\n')
            f_xml.write('<truncated>0</truncated>\n')
            f_xml.write('<difficult>0</difficult>\n')
            f_xml.write('<bndbox>\n')
            f_xml.write('<xmin>' + str(float(x1)) + '</xmin>\n')
            f_xml.write('<ymin>' + str(float(y1)) + '</ymin>\n')
            f_xml.write('<xmax>' + str(float(x2)) + '</xmax>\n')
            f_xml.write('<ymax>' + str(float(y2)) + '</ymax>\n')
            f_xml.write('</bndbox>\n')
            f_xml.write('<object_id>' + object_id.encode('utf8') + '</object_id>\n')
            f_xml.write('<embedding_status>' + str(embedding_status) + '</embedding_status>\n')
            f_xml.write('</object>\n')
        f_xml.write('</annotation>\n')
        # close xml file
        f_xml.close()
        # visualization
        if aug_cfg.visualization:
            for id_viz, (tl_corner, br_corner) in enumerate(trans_tl_and_br_coors):
                sku_viz_name = src_categ_and_sku[id_viz][1]
                dst_img = draw_rect(dst_img, tl_corner, br_corner)
                h = br_corner[1] - tl_corner[1]
                try:
                    cv2.putText(dst_img, sku_viz_name, (tl_corner[0], tl_corner[1] + np.random.randint(h)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
                except:
                    pass
        # write img
        cv2.imwrite(full_dst_img_name_with_ext, dst_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), aug_cfg.jpg_quality])
        print full_dst_img_name_with_ext

    def make_objects_markup_data(src_facings):
        objects_markup_data_dst = []
        for cur_facing in src_facings:
            klass_id = cur_facing['klass_id']
            klass_package_type_code = cur_facing['klass_package_type_code']
            klass_code = cur_facing['klass_code']
            klass_is_disabled = cur_facing['klass_is_disabled']
            product_category_code = cur_facing['product_category_code']
            object_x1 = cur_facing['x1']
            object_x2 = cur_facing['x2']
            object_y1 = cur_facing['y1']
            object_y2 = cur_facing['y2']
            object_embedding_status = cur_facing['embedding_status']
            objects_markup_data_dst.append([klass_package_type_code, klass_code, object_x1, object_x2, object_y1, object_y2, klass_id, object_embedding_status])
        return objects_markup_data_dst

    # read info about img and markup
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
    facings = imgs_markup_list_item['facings']

    # read img
    src_img_full = cv2.imread(src_img_path)

    if lot_type != 'verif':
        rel_folder = os.path.join(train_folder_name, project_name, lot_storagepath)
    else:
        rel_folder = os.path.join(val_folder_name, project_name, lot_storagepath)

    if lot_type == 'mark' or lot_type == 'valid':
        for i in range(aug_cfg.num_out_files_for_one_img_in_trainval_using_augmentation):
            objects_markup_data = make_objects_markup_data(facings)
            src_imgs, object_markup_data_for_imgs = divide_img(src_img_full, objects_markup_data)
            for id_zip, (src_img, objects_markup_data) in enumerate(zip(src_imgs, object_markup_data_for_imgs)):
                # define image name and xml name
                photo_original_name_without_ext = photo_original_name[:photo_original_name.rfind('.')] + '_' + str(
                    i) + '_' + str(id_zip)
                rel_dst_img_name_without_ext = os.path.join(rel_folder, photo_original_name_without_ext)
                full_dst_img_name_without_ext = os.path.join(path_to_imgs, rel_dst_img_name_without_ext)
                full_dst_img_name_with_ext = full_dst_img_name_without_ext + photo_original_name[photo_original_name.rfind('.'):]
                full_dst_xml_name_with_ext = os.path.join(path_to_xmls, rel_dst_img_name_without_ext + '.xml')
                dst_img_and_xml_info = {}
                dst_img_and_xml_info['photo_original_name_without_ext'] = photo_original_name_without_ext
                dst_img_and_xml_info['rel_dst_img_name_without_ext'] = rel_dst_img_name_without_ext
                dst_img_and_xml_info['full_dst_img_name_without_ext'] = full_dst_img_name_without_ext
                dst_img_and_xml_info['full_dst_img_name_with_ext'] = full_dst_img_name_with_ext
                dst_img_and_xml_info['full_dst_xml_name_with_ext'] = full_dst_xml_name_with_ext
                dst_img_and_xml_info['photo_original_id'] = photo_original_id
                dst_img_and_xml_info['photo_original_is_collage'] = photo_original_is_collage

                # read coors
                src_tl_and_br_coors = [
                    [[object_markup_rec[2], object_markup_rec[4]], [object_markup_rec[3], object_markup_rec[5]]] for
                    object_markup_rec in objects_markup_data]

                src_img = add_background(src_img, background_imgs_markup_list) # ToDo
                if aug_cfg.IMG_MAX_SIZE == 'MAX':
                    IMG_MAX_SIZE = max(src_img.shape[0], src_img.shape[1])
                else:
                    IMG_MAX_SIZE = aug_cfg.IMG_MAX_SIZE
                if i == 0:
                    dst_img, trans_tl_and_br_coors = make_just_resize(src_img, src_tl_and_br_coors,
                                                                      max_size=IMG_MAX_SIZE)
                else:
                    dst_img, trans_tl_and_br_coors = distort_img(src_img, src_tl_and_br_coors,
                                                                 max_size=IMG_MAX_SIZE)

                save_img_and_xml(dst_img, trans_tl_and_br_coors, objects_markup_data, dst_img_and_xml_info)
    if lot_type == 'verif':
        objects_markup_data = make_objects_markup_data(facings)
        photo_original_name_without_ext = photo_original_name[:photo_original_name.rfind('.')] + '_' + str(0)
        rel_dst_img_name_without_ext = os.path.join(rel_folder, photo_original_name_without_ext)
        full_dst_img_name_without_ext = os.path.join(path_to_imgs, rel_dst_img_name_without_ext)
        full_dst_img_name_with_ext = full_dst_img_name_without_ext + photo_original_name[photo_original_name.rfind('.'):]
        full_dst_xml_name_with_ext = os.path.join(path_to_xmls, rel_dst_img_name_without_ext + '.xml')
        dst_img_and_xml_info = {}
        dst_img_and_xml_info['photo_original_name_without_ext'] = photo_original_name_without_ext
        dst_img_and_xml_info['rel_dst_img_name_without_ext'] = rel_dst_img_name_without_ext
        dst_img_and_xml_info['full_dst_img_name_without_ext'] = full_dst_img_name_without_ext
        dst_img_and_xml_info['full_dst_img_name_with_ext'] = full_dst_img_name_with_ext
        dst_img_and_xml_info['full_dst_xml_name_with_ext'] = full_dst_xml_name_with_ext
        dst_img_and_xml_info['photo_original_id'] = photo_original_id
        dst_img_and_xml_info['photo_original_is_collage'] = photo_original_is_collage

        # read coors
        src_tl_and_br_coors = [
            [[object_markup_rec[2], object_markup_rec[4]], [object_markup_rec[3], object_markup_rec[5]]] for
            object_markup_rec in objects_markup_data]

        dst_img, trans_tl_and_br_coors = src_img_full[:], src_tl_and_br_coors[:]
        save_img_and_xml(dst_img, trans_tl_and_br_coors, objects_markup_data, dst_img_and_xml_info)

import time
start_time = time.time()
if multiproc_flag:
    pool = Pool()
    pool.map(augment_and_save_img_and_save_xml, imgs_markup_list) # 810.64363718 seconds
    pool.close()
    pool.join()
else:
    map(augment_and_save_img_and_save_xml, imgs_markup_list) # 3417.19446993 seconds
end_time = time.time()
print end_time-start_time, 'seconds'

def make_dataset_list(rel_folder):
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
    fileList = [file_name[file_name.rfind(rel_folder):file_name.rfind('.')]+'\n' for file_name in file_list]
    return fileList

f_trainval.writelines(make_dataset_list(train_folder_name))
f_test.writelines(make_dataset_list(val_folder_name))

f_trainval.close()
f_test.close()