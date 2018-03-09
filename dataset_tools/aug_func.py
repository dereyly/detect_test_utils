import os
import random
import math
import cv2
import numpy as np

''' Import test libs'''
import matplotlib.pyplot as plt
import time


def make_color_noise(src_img, chance_percent, noise_percent, random_seed):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img
    else:
        width = src_img.shape[1]
        height = src_img.shape[0]
        for j in xrange(height):
            for i in xrange(width):
                percent_for_noise = random.random() * 99
                if noise_percent > percent_for_noise:
                    img_copy[j][i] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        return img_copy


def make_salt_and_pepper_noise(src_img, chance_percent, noise_percent, random_seed):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img
    else:
        width = src_img.shape[1]
        height = src_img.shape[0]
        for j in xrange(height):
            for i in xrange(width):
                percent_for_noise = random.random() * 99
                if noise_percent > percent_for_noise:
                    salt_or_pepper_choice = random.randint(0, 100)
                    if salt_or_pepper_choice < 50:
                        img_copy[j][i] = [0, 0, 0]
                    else:
                        img_copy[j][i] = [255, 255, 255]
        return img_copy


def make_gaussian_blur(src_img, chance_percent, min_kernel, max_kernel, random_seed):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img
    else:
        if min_kernel == max_kernel:
            return cv2.GaussianBlur(img_copy, min_kernel, 0)
        else:
            if min_kernel[0] == min_kernel[1] and max_kernel[0] == max_kernel[1]:
                kernel_width = random.randint(min_kernel[0], max_kernel[0])
                kernel_width = int(kernel_width / 2)
                kernel_width = max(1, kernel_width)
                kernel_width = kernel_width * 2 + 1
                kernel_height = kernel_width
                return cv2.GaussianBlur(img_copy, (kernel_width, kernel_height), 0)
            else:
                kernel_width = random.randint(min(min_kernel[0], max_kernel[0]), max(min_kernel[0], max_kernel[0]))
                kernel_height = random.randint(min(min_kernel[1], max_kernel[1]), max(min_kernel[1], max_kernel[1]))
                kernel_width = int(kernel_width / 2)
                kernel_height = int(kernel_height / 2)
                kernel_width = max(1, kernel_width)
                kernel_width = kernel_width * 2 + 1
                kernel_height = max(1, kernel_height)
                kernel_height = kernel_height * 2 + 1
                return cv2.GaussianBlur(img_copy, (kernel_width, kernel_height), 0)


def make_box_blur(src_img, chance_percent, min_kernel, max_kernel, random_seed):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img
    else:
        if min_kernel == max_kernel:
            return cv2.blur(img_copy, min_kernel)
        else:
            if min_kernel[0] == min_kernel[1] and max_kernel[0] == max_kernel[1]:
                kernel_width = random.randint(min_kernel[0], max_kernel[0])
                kernel_width = int(kernel_width / 2)
                kernel_width = max(1, kernel_width)
                kernel_width = kernel_width * 2 + 1
                kernel_height = kernel_width
                return cv2.blur(img_copy, (kernel_width, kernel_height))
            else:
                kernel_width = random.randint(min(min_kernel[0], max_kernel[0]), max(min_kernel[0], max_kernel[0]))
                kernel_height = random.randint(min(min_kernel[1], max_kernel[1]), max(min_kernel[1], max_kernel[1]))
                kernel_width = int(kernel_width / 2)
                kernel_height = int(kernel_height / 2)
                kernel_width = max(1, kernel_width)
                kernel_width = kernel_width * 2 + 1
                kernel_height = max(1, kernel_height)
                kernel_height = kernel_height * 2 + 1
                return cv2.blur(img_copy, (kernel_width, kernel_height))


def make_sharpness(src_img, chance_percent, random_seed):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img
    else:
        kernel_sharpen = np.array([[-1, -1, -1, -1, -1],
                                     [-1, 2, 2, 2, -1],
                                     [-1, 2, 8, 2, -1],
                                     [-1, 2, 2, 2, -1],
                                     [-1, -1, -1, -1, -1]]) / 8.0
        return cv2.filter2D(img_copy, -1, kernel_sharpen)


def make_motion_blur(src_img, chance_percent, square_kernel_size, blur_direction='random', random_seed=None):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img
    else:
        if blur_direction == 'horizontal':
            kernel = np.zeros(square_kernel_size)
            kernel[int((square_kernel_size[0]-1)/2), :] = np.ones(square_kernel_size[0])
            kernel = kernel / square_kernel_size[0]
            return cv2.filter2D(img_copy, -1, kernel)
        elif blur_direction == 'vertical':
            kernel = np.zeros(square_kernel_size)
            kernel[:, int((square_kernel_size[0] - 1) / 2)] = np.ones(square_kernel_size[0])
            kernel = kernel / square_kernel_size[0]
            return cv2.filter2D(img_copy, -1, kernel)
        elif blur_direction == 'diagonal_lt_to_rb':
            kernel = np.zeros(square_kernel_size)
            for j in xrange(square_kernel_size[0]):
                for i in xrange(square_kernel_size[0]):
                    if i == j:
                        kernel[j][i] = 1.0 / square_kernel_size[0]
            return cv2.filter2D(img_copy, -1, kernel)
        elif blur_direction == 'diagonal_lb_to_rt':
            kernel = np.zeros(square_kernel_size)
            for j in xrange(square_kernel_size[0]):
                for i in xrange(square_kernel_size[0]):
                    if square_kernel_size[0] - 1 - i == j:
                        kernel[j][i] = 1.0 / square_kernel_size[0]
            return cv2.filter2D(img_copy, -1, kernel)
        else:
            direction_list = ['horizontal', 'vertical', 'diagonal_lt_to_rb', 'diagonal_lb_to_rt']
            direction = random.choice(direction_list)
            return make_motion_blur(src_img, chance_percent, square_kernel_size, direction)

def affine_transform_points(points_pairs, M):
    # add ones
    points = []
    for pair in points_pairs:
        x1, y1 = pair[0]
        x2, y2 = pair[1]
        points.append(pair[0])
        points.append([x2, y1])
        points.append([x1, y2])
        points.append(pair[1])
    ones = np.ones(shape=(len(points), 1))

    points_ones = np.hstack([points, ones])

    # transform points
    transformed_points = M.dot(points_ones.T).T

    transformed_points_quads = transformed_points.reshape((transformed_points.shape[0]/4, 4, 2))
    transformed_points_pairs = []
    for quad in transformed_points_quads:
        x1 = np.min(quad[:, 0])
        y1 = np.min(quad[:, 1])
        x2 = np.max(quad[:, 0])
        y2 = np.max(quad[:, 1])
        transformed_points_pairs.append([[x1, y1], [x2, y2]])

    return transformed_points_pairs

def perspective_transform_points(points_pairs, M):
    # add ones
    points = []
    for pair in points_pairs:
        x1, y1 = pair[0]
        x2, y2 = pair[1]
        points.append(pair[0])
        points.append([x2, y1])
        points.append([x1, y2])
        points.append(pair[1])
    ones = np.ones(shape=(len(points), 1))

    points_ones = np.hstack([points, ones])

    # transform points
    transformed_points = M.dot(points_ones.T).T
    transformed_points = np.array([[triplet[0]/triplet[2], triplet[1]/triplet[2]] for triplet in transformed_points])

    transformed_points_quads = transformed_points.reshape((transformed_points.shape[0]/4, 4, 2))
    transformed_points_pairs = []
    for quad in transformed_points_quads:
        x1 = np.min(quad[:, 0])
        y1 = np.min(quad[:, 1])
        x2 = np.max(quad[:, 0])
        y2 = np.max(quad[:, 1])
        transformed_points_pairs.append([[x1, y1], [x2, y2]])

    return transformed_points_pairs

def make_rotate(src_img, tl_and_br_coors, chance_percent, max_angle, random_seed):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img, tl_and_br_coors
    else:
        angle = random.randint(-max_angle, max_angle)
        #angle = random.choice([-max_angle, max_angle])
        width_src = src_img.shape[1]
        height_src = src_img.shape[0]
        M_help = cv2.getRotationMatrix2D((width_src/2.0, height_src/2.0), -angle, 1.0)
        cos = np.abs(M_help[0, 0])
        sin = np.abs(M_help[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((height_src * sin) + (width_src * cos))
        nH = int((height_src * cos) + (width_src * sin))
        padding_y = np.int(np.ceil((nH - height_src) / 2.0))
        padding_x = np.int(np.ceil((nW - width_src) / 2.0))
        if padding_x != 0 or padding_y != 0:
            img_copy = cv2.copyMakeBorder(img_copy, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_REPLICATE)
            #img_copy = cv2.copyMakeBorder(img_copy, padding_y, padding_y, padding_x, padding_x, cv2.BORDER_CONSTANT, value=(128,128,128))
        height, width = img_copy.shape[:2]
        rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        tl_and_br_coors = np.array(tl_and_br_coors)
        tl_and_br_coors[:, :, 0] = tl_and_br_coors[:, :, 0] + padding_x
        tl_and_br_coors[:, :, 1] = tl_and_br_coors[:, :, 1] + padding_y
        trans_tl_and_br_coors = affine_transform_points(tl_and_br_coors, rot_mat)
        return cv2.warpAffine(img_copy, rot_mat,
                              (int(math.ceil(width)), int(math.ceil(height))),
                              flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_REPLICATE), trans_tl_and_br_coors


def make_perspective_transform(src_img, tl_and_br_coors, chance_percent, percent_transform, random_seed):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img, tl_and_br_coors
    else:
        width_src = src_img.shape[1]
        height_src = src_img.shape[0]


        shift_coors = max(1,int(min(width_src, height_src) * (percent_transform / 100.0)))
        delta_lt_x = random.randrange(-shift_coors, shift_coors)
        delta_lt_y = random.randrange(-shift_coors, shift_coors)
        delta_rt_x = random.randrange(-shift_coors, shift_coors)
        delta_rt_y = random.randrange(-shift_coors, shift_coors)
        delta_lb_x = random.randrange(-shift_coors, shift_coors)
        delta_lb_y = random.randrange(-shift_coors, shift_coors)
        delta_rb_x = random.randrange(-shift_coors, shift_coors)
        delta_rb_y = random.randrange(-shift_coors, shift_coors)
        min_left = min(delta_lt_x, delta_lb_x)
        min_top = min(delta_lt_y, delta_rt_y)
        max_right = max(delta_rt_x, delta_rb_x)
        max_bottom = max(delta_lb_y, delta_rb_y)
        padding_left = np.abs(min_left) if min_left < 0 else 0
        padding_top = np.abs(min_top) if min_top < 0 else 0
        padding_right = max_right if max_right > 0 else 0
        padding_bottom = max_bottom if max_bottom > 0 else 0
        cutting_left = min_left if min_left > 0 else 0
        cutting_top = min_top if min_top > 0 else 0
        cutting_right = np.abs(max_right) if max_right < 0 else 0
        cutting_bottom = np.abs(max_bottom) if max_bottom < 0 else 0
        img_copy = cv2.copyMakeBorder(img_copy, padding_top, padding_bottom, padding_left, padding_right, cv2.BORDER_REPLICATE)
        height, width = img_copy.shape[:2]
        pts1 = np.float32([[0 + max(0, -delta_lt_x),
                            0 + max(0, -delta_lt_y)],
                           [width + min(0, -delta_rt_x),
                            0 + max(0, -delta_rt_y)],
                           [0 + max(0, -delta_lb_x),
                            height + min(0, -delta_lb_y)],
                           [width + min(0, -delta_rb_x),
                            height + min(0, -delta_rb_y)]])
        pts2 = np.float32([[0 + max(0, delta_lt_x),
                            0 + max(0, delta_lt_y)],
                           [width + min(0, delta_rt_x),
                            0 + max(0, delta_rt_y)],
                           [0 + max(0, delta_lb_x),
                            height + min(0, delta_lb_y)],
                           [width + min(0, delta_rb_x),
                            height + min(0, delta_rb_y)]])
        trans_mat = cv2.getPerspectiveTransform(pts1, pts2)
        tl_and_br_coors = np.array(tl_and_br_coors)
        tl_and_br_coors[:, :, 0] = tl_and_br_coors[:, :, 0] + padding_left
        tl_and_br_coors[:, :, 1] = tl_and_br_coors[:, :, 1] + padding_top
        trans_tl_and_br_coors = perspective_transform_points(tl_and_br_coors, trans_mat)
        transformed_img = cv2.warpPerspective(img_copy, trans_mat,
                                   (int(math.ceil(width)), int(math.ceil(height))),
                                   flags=cv2.INTER_AREA,
                                    borderMode=cv2.BORDER_REPLICATE)
        trans_tl_and_br_coors = np.array(trans_tl_and_br_coors)
        trans_tl_and_br_coors[:, :, 0] = trans_tl_and_br_coors[:, :, 0] - cutting_left
        trans_tl_and_br_coors[:, :, 1] = trans_tl_and_br_coors[:, :, 1] - cutting_top
        transformed_img = transformed_img[cutting_top:height - cutting_bottom, cutting_left : width - cutting_right]
        # for top_left_corner, bottom_right_corner in tl_and_br_coors:
        #     transformed_tl_c, transformed_br_c = get_new_coors_using_perspective_transform(trans_mat, top_left_corner,
        #                                                                               bottom_right_corner)
        #     trans_tl_and_br_coors.append([transformed_tl_c, transformed_br_c])
        return transformed_img, trans_tl_and_br_coors


def make_another_scale(src_img, tl_and_br_coors, chance_percent, min_scale, max_scale, random_seed):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img, tl_and_br_coors
    else:
        width = src_img.shape[1]
        height = src_img.shape[0]
        scale = random.randint(int(min_scale*100), int(max_scale*100)) / 100.0
        rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), 0, scale)
        trans_tl_and_br_coors = []
        for top_left_corner, bottom_right_corner in tl_and_br_coors:
            transformed_tl_c, transformed_br_c = get_new_coors_using_affine_transform(rot_mat, top_left_corner,
                                                                                      bottom_right_corner)
            trans_tl_and_br_coors.append([transformed_tl_c, transformed_br_c])
        return cv2.warpAffine(img_copy, rot_mat,
                              (int(math.ceil(width)), int(math.ceil(height))),
                              flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_REPLICATE), trans_tl_and_br_coors

def make_resize_min(src_img, tl_and_br_coors, min_size):
    img_copy = src_img.copy()
    width = src_img.shape[1]
    height = src_img.shape[0]
    if width > height:
        new_height = min_size
        ratio = new_height * 1.0 / height
        new_width = width * ratio
    elif width < height:
        new_width = min_size
        ratio = new_width * 1.0 / width
        new_height = height * ratio
    else:
        new_width = min_size
        new_height = min_size
        ratio = new_width * 1.0 / width
    img_copy = cv2.resize(img_copy, (int(round(new_width)), int(round(new_height))), interpolation=cv2.INTER_AREA)
    trans_tl_and_br_coors = []
    for top_left_corner, bottom_right_corner in tl_and_br_coors:
        transformed_tl_c = tuple(np.array(top_left_corner) * ratio)
        transformed_br_c = tuple(np.array(bottom_right_corner) * ratio)
        trans_tl_and_br_coors.append([transformed_tl_c, transformed_br_c])
    return img_copy, trans_tl_and_br_coors

def make_resize_max(src_img, tl_and_br_coors, max_size):
    img_copy = src_img.copy()
    width = src_img.shape[1]
    height = src_img.shape[0]
    if width < height:
        new_height = max_size
        ratio = new_height * 1.0 / height
        new_width = width * ratio
    elif width > height:
        new_width = max_size
        ratio = new_width * 1.0 / width
        new_height = height * ratio
    else:
        new_width = max_size
        new_height = max_size
        ratio = new_width * 1.0 / width
    img_copy = cv2.resize(img_copy, (int(round(new_width)), int(round(new_height))), interpolation=cv2.INTER_AREA)
    trans_tl_and_br_coors = []
    for top_left_corner, bottom_right_corner in tl_and_br_coors:
        transformed_tl_c = tuple(np.array(top_left_corner) * ratio)
        transformed_br_c = tuple(np.array(bottom_right_corner) * ratio)
        trans_tl_and_br_coors.append([transformed_tl_c, transformed_br_c])
    return img_copy, trans_tl_and_br_coors

def make_resize_max_without_coors(src_img, max_size):
    img_copy = src_img.copy()
    width = src_img.shape[1]
    height = src_img.shape[0]
    if width < height:
        new_height = max_size
        ratio = new_height * 1.0 / height
        new_width = width * ratio
    elif width > height:
        new_width = max_size
        ratio = new_width * 1.0 / width
        new_height = height * ratio
    else:
        new_width = max_size
        new_height = max_size
        ratio = new_width * 1.0 / width
    img_copy = cv2.resize(img_copy, (int(round(new_width)), int(round(new_height))), interpolation=cv2.INTER_AREA)
    return img_copy

def change_white_balance(src_img, chance_percent, max_cold_value, max_warm_value, random_seed):
    def make_warm_light(img, max_warm_value):
        warm_value = random.uniform(1.0, max_warm_value)
        blue, green, red = cv2.split(img)
        blue_old = blue.flatten()
        red_old = red.flatten()
        warm_value_norm = (255 - 127 * warm_value) / (255 - 127)
        blue_norm = 255 - warm_value_norm * 255
        b_new = np.uint8(blue_old / warm_value)
        b_new = b_new.reshape(blue.shape[0], blue.shape[1])
        r_new = np.uint8(red_old * warm_value_norm + blue_norm)
        r_new = r_new.reshape(red.shape[0], red.shape[1])
        warm_img = cv2.merge((b_new, green, r_new))
        return warm_img

    def make_cold_light(img, max_cold_value):
        cold_value = random.uniform(1.0, max_cold_value)
        blue, green, red = cv2.split(img)
        blue_old = blue.flatten()
        red_old = red.flatten()
        cold_value_norm = (255 - 127 * cold_value) / (255 - 127)
        blue_norm = 255 - cold_value_norm * 255
        b_new = np.uint8(blue_old * cold_value_norm + blue_norm)
        b_new = b_new.reshape(blue.shape[0], blue.shape[1])
        r_new = np.uint8(red_old / cold_value)
        r_new = r_new.reshape(red.shape[0], red.shape[1])
        cold_img = cv2.merge((b_new, green, r_new))
        return cold_img

    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img
    else:
        light_kinds = ['cold', 'warm']
        light_choice = random.choice(light_kinds)
        if light_choice == 'cold':
            img_copy = make_cold_light(img_copy, max_cold_value)
        else:
            img_copy = make_warm_light(img_copy, max_warm_value)
        return img_copy

def make_shift(src_img, tl_and_br_coors, chance_percent, percent_max_shift, random_seed):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img, tl_and_br_coors
    else:
        width = src_img.shape[1]
        height = src_img.shape[0]
        shift = random.randint(0, int(min(width, height) * (percent_max_shift / 100.0)))
        pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        quarters_list = [1, 2, 3, 4]
        qr = random.choice(quarters_list)
        if qr == 1:
            pts2 = np.float32([[0 + shift, 0 - shift],
                               [width + shift, 0 - shift],
                               [0 + shift, height - shift],
                               [width + shift, height - shift]])
        elif qr == 2:
            pts2 = np.float32([[0 - shift, 0 - shift],
                               [width - shift, 0 - shift],
                               [0 - shift, height - shift],
                               [width - shift, height - shift]])
        elif qr == 3:
            pts2 = np.float32([[0 - shift, 0 + shift],
                               [width - shift, 0 + shift],
                               [0 - shift, height + shift],
                               [width - shift, height + shift]])
        else:
            pts2 = np.float32([[0 + shift, 0 + shift],
                               [width + shift, 0 + shift],
                               [0 + shift, height + shift],
                               [width + shift, height + shift]])
        trans_mat = cv2.getPerspectiveTransform(pts1, pts2)
        trans_tl_and_br_coors = []
        for top_left_corner, bottom_right_corner in tl_and_br_coors:
            transformed_tl_c, transformed_br_c = get_new_coors_using_perspective_transform(trans_mat, top_left_corner,
                                                                                           bottom_right_corner)
            trans_tl_and_br_coors.append([transformed_tl_c, transformed_br_c])
        return cv2.warpPerspective(img_copy, trans_mat,
                                   (int(math.ceil(width)), int(math.ceil(height))),
                                   flags=cv2.INTER_AREA,
                                   borderMode=cv2.BORDER_REPLICATE), trans_tl_and_br_coors


def change_brightness_contrast_saturation(src_img, chance_percent, random_seed,
                                         min_brightness_summand=-25, max_brightness_summand=25,
                                         min_contrast_multiplier=0.85, max_contrast_multiplier=1.25,
                                         min_saturation_multiplier=0.8, max_saturation_multiplier=1.45):
    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img
    else:
        percent_for_choice_distortion = random.random() * 99
        brightness_bit = random.getrandbits(1)
        contrast_bit = random.getrandbits(1)
        saturation_bit = random.getrandbits(1)
        if percent_for_choice_distortion < 50:
            list_choices = ['br','con','sat']
            choice = random.choice(list_choices)
            if choice == 'br':
                if not brightness_bit:
                    brightness_summand = random.randint(min_brightness_summand, 0)
                else:
                    brightness_summand = random.randint(0, max_brightness_summand)
                hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)  # convert it to hsv
                h, s, v = cv2.split(hsv)
                v = cv2.add(v, brightness_summand)
                final_hsv = cv2.merge((h, s, v))
                img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                return img
            elif choice == 'con':
                if not contrast_bit:
                    contrast_multiplier = random.randint(int(min_contrast_multiplier * 100),
                                                        int(1 * 100)) / 100.0
                else:
                    contrast_multiplier = random.randint(int(1 * 100),
                                                         int(max_contrast_multiplier * 100)) / 100.0
                hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)  # convert it to hsv
                h, s, v = cv2.split(hsv)
                v = cv2.convertScaleAbs(v, -1, contrast_multiplier, 0)
                final_hsv = cv2.merge((h, s, v))
                img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                return img
            else:
                if not contrast_bit:
                    saturation_multiplier = random.randint(int(min_saturation_multiplier * 100),
                                                           int(1 * 100)) / 100.0
                else:
                    saturation_multiplier = random.randint(int(1 * 100),
                                                           int(max_saturation_multiplier * 100)) / 100.0
                hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)  # convert it to hsv
                h, s, v = cv2.split(hsv)
                s = cv2.multiply(s, saturation_multiplier)
                final_hsv = cv2.merge((h, s, v))
                img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                return img
        elif percent_for_choice_distortion > 50 and percent_for_choice_distortion <= 80:
            if not brightness_bit:
                brightness_summand = random.randint(int(min_brightness_summand/2.25), 0)
            else:
                brightness_summand = random.randint(0, int(max_brightness_summand/2.25))
            if not contrast_bit:
                contrast_var = random.randint(int(min_contrast_multiplier * 100),
                                                        int(1 * 100)) / 100.0
                contrast_multiplier = contrast_var + (1-contrast_var)/1.75
            else:
                contrast_var = random.randint(int(1 * 100),
                                                        int(max_contrast_multiplier * 100)) / 100.0
                contrast_multiplier = contrast_var + (contrast_var - 1) / 1.75
            if not saturation_bit:
                saturation_var = random.randint(int(min_saturation_multiplier*100), int(1*100)) / 100.0
                saturation_multiplier = saturation_var + (1-saturation_var)/1.75
            else:
                saturation_var = random.randint(int(1 * 100),
                                              int(max_saturation_multiplier * 100)) / 100.0
                saturation_multiplier = saturation_var + (saturation_var - 1) / 1.75
            hsv = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)  # convert it to hsv
            h, s, v = cv2.split(hsv)
            v = cv2.convertScaleAbs(v, -1, contrast_multiplier, 0)
            v = cv2.add(v, brightness_summand)
            s = cv2.multiply(s, saturation_multiplier)
            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            return img
        elif percent_for_choice > 80 and percent_for_choice < 90:
            img_yuv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            return img
        else:
            img_yuv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
            # convert the YUV image back to RGB format
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            return img

def add_glints(src_img, chance_percent, random_seed):
    def add_glint(image):
        min_side = min(image.shape[1], image.shape[0])
        canvas = np.zeros((img_copy.shape[0], img_copy.shape[1])).astype('uint8')
        ellipse_center_x = random.randint(int(img_copy.shape[1] * 0.1), int(img_copy.shape[1] * 0.9))
        #print ellipse_center_x
        ellipse_center_y = random.randint(int(img_copy.shape[0] * 0.1), int(img_copy.shape[0] * 0.9))
        #print ellipse_center_y
        ellipse_shape_x = random.randint(int(min_side*0.01), int(min_side*0.02))
        ellipse_shape_y = int(ellipse_shape_x * (random.random() + 0.5))
        angle = random.randint(0, 360)
        value_white_pixels = random.randint(96, 160)
        cv2.ellipse(canvas, (ellipse_center_x, ellipse_center_y), (ellipse_shape_x, ellipse_shape_y),
                    0, 0, 180, value_white_pixels, -1)
        cv2.ellipse(canvas, (ellipse_center_x, ellipse_center_y), (ellipse_shape_x, ellipse_shape_y),
                    180, 0, 180, value_white_pixels, -1)
        canvas = cv2.GaussianBlur(canvas, (51, 51), 0)
        # cv2.imwrite('canvas.png', canvas)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        img = cv2.add(image, canvas)
        return img

    random.seed(random_seed)
    img_copy = src_img.copy()
    percent_for_choice = random.random() * 99
    if chance_percent <= percent_for_choice:
        return src_img
    else:
        num_glints = random.randint(1, 10)
        for i in xrange(num_glints):
            img_copy = add_glint(img_copy)
        return img_copy

def trans_tl_and_br_coors_to_int(src_img, trans_tl_and_br_coors):
    width = src_img.shape[1]
    height = src_img.shape[0]
    trans_tl_and_br_coors_int = []
    for trans_tl_and_br_coors_rec in trans_tl_and_br_coors:
        x1, y1 = trans_tl_and_br_coors_rec[0]
        x2, y2 = trans_tl_and_br_coors_rec[1]
        x1_int = min(int(round(x1)), width - 2) if x1 > 0 else 0
        y1_int = min(int(round(y1)), height - 2) if y1 > 0 else 0
        x2_int = min(int(round(x2)), width - 1) if x2 > 0 else 1
        y2_int = min(int(round(y2)), height - 1) if y2 > 0 else 1
        trans_tl_and_br_coors_int.append([(x1_int, y1_int), (x2_int, y2_int)])
    return trans_tl_and_br_coors_int


def make_just_resize(src_img, tl_and_br_coors, max_size=None, min_size=None):
    if min_size:
        out_img, trans_tl_and_br_coors = make_resize_min(src_img, tl_and_br_coors, min_size)
    else:
        out_img, trans_tl_and_br_coors = make_resize_max(src_img, tl_and_br_coors, max_size)
    trans_tl_and_br_coors_int = trans_tl_and_br_coors_to_int(out_img, trans_tl_and_br_coors)
    return out_img, trans_tl_and_br_coors_int

def distort_img(src_img, tl_and_br_coors, random_seed=None, min_size=None, max_size=None):
    out_img = src_img.copy()
    trans_tl_and_br_coors = tl_and_br_coors[:]
    out_img = change_white_balance(out_img, 75, 1.15, 1.15, random_seed)
    out_img, trans_tl_and_br_coors = make_rotate(out_img, trans_tl_and_br_coors, 75, 15, random_seed) # 35
    out_img, trans_tl_and_br_coors = make_perspective_transform(out_img, trans_tl_and_br_coors, 75, 15, random_seed) # 35
    #out_img, trans_tl_and_br_coors = make_another_scale(out_img, trans_tl_and_br_coors, 100,  0.975, 1/0.975, random_seed) # 35
    #out_img, trans_tl_and_br_coors = make_shift(out_img, trans_tl_and_br_coors, 35, 3, random_seed) # 35
    out_img = make_sharpness(out_img, 35, random_seed) # 35
    out_img = make_color_noise(out_img, 35, 0.075, random_seed) # 25
    #out_img = make_salt_and_pepper_noise(out_img, 100, 0.05, random_seed) # 20
    out_img = make_motion_blur(out_img, 5, (5, 5), 'random', random_seed) # 15
    out_img = make_gaussian_blur(out_img, 2, (3, 3), (7, 7), random_seed) # 3
    out_img = make_box_blur(out_img, 1, (3, 3), (5, 5), random_seed) # 3
    #out_img = add_glints(out_img, 10, random_seed) # 10
    out_img = change_brightness_contrast_saturation(out_img, 80, random_seed) # 100
    out_img, trans_tl_and_br_coors = make_resize_max(out_img, trans_tl_and_br_coors, max_size)
    trans_tl_and_br_coors_int = trans_tl_and_br_coors_to_int(out_img, trans_tl_and_br_coors)
    return out_img, trans_tl_and_br_coors_int


def draw_rect(src_img, top_left_corner=(369, 625), bottom_right_corner=(636, 1003)):
    img_copy = src_img.copy()
    top_left_corner = tuple(np.array(top_left_corner).astype(int))
    bottom_right_corner = tuple(np.array(bottom_right_corner).astype(int))
    cv2.rectangle(img_copy, top_left_corner, bottom_right_corner, (0, 255, 255), 2)
    return img_copy



def distort_img_without_coors_and_resize(src_img, new_size=(224, 224), random_seed=None):
    out_img = src_img.copy()
    out_img = change_white_balance(out_img, 75, 1.2, 1.2, random_seed)
    out_img = make_rotate(out_img, 35, 5, random_seed)  # 35
    out_img = make_perspective_transform(out_img, 35, 3, random_seed) # 35
    #out_img = make_another_scale(out_img, 35,  0.975, 1/0.975, random_seed) # 35
    out_img = make_shift(out_img, 35, 4, random_seed)  # 35
    out_img = make_sharpness(out_img, 35, random_seed)  # 35
    out_img = make_color_noise(out_img, 35, 0.075, random_seed)  # 25
    # out_img = make_salt_and_pepper_noise(out_img, 100, 0.05, random_seed) # 20
    out_img = make_motion_blur(out_img, 15, (5, 5), 'random', random_seed)  # 15
    out_img = make_gaussian_blur(out_img, 5, (3, 3), (7, 7), random_seed)  # 3
    out_img = make_box_blur(out_img, 3, (3, 3), (5, 5), random_seed)  # 3
    # out_img = add_glints(out_img, 10, random_seed) # 10
    out_img = change_brightness_contrast_saturation(out_img, 95, random_seed)  # 100
    out_img = cv2.resize(out_img, new_size, interpolation=cv2.INTER_AREA)
    return out_img