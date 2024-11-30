# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:18:05 2018

@author: yxie8171
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('TkAgg')
try:
    import pyautogui as pygui
except:
    a = 1
import datetime
import os
import time
import copy


def cv_find_all_pic(filename, region=(), find_thresh_hold=0.05, bool_find_one=False, template_filename=None, bool_output_full_info=False):
    """

    Args:
        filename (str, PIL.Image, np.ndarray): The feature to find, it could be path to a file, PIL.image (pyautogui screenshot), or numpy array
        region (tuple): the region of the place to take the screenshot
        find_thresh_hold (float): lower the thresh hold higher requirement the similarity
        bool_find_one (bool): Only find one
        template_filename (None, str, PIL.Image, np.ndarray): The raw picture, it could be path to a None, in which case a fresh screenshot will be taken,
            it could also be file path, PIL.Image, or numpy array.
        bool_output_full_info (bool): whether to output the full dict of value and found region

    Returns:
        list: list of the regions for the found features in the format of [(left, top, length, height), ...]
    """

    if 'Process input parameters':
        if region == ():
            region = tuple([0, 0] + list(pygui.size()))
        if type(filename) is str:
            img = mpimg.imread(filename)
            if img.shape[2] == 4:
                img = np.uint8(img[:, :, :-1] * 255)
            else:
                img = np.uint8(img * 255)
        elif 'PIL.Image.Image' in str(type(filename)):
            img = (np.asarray(filename)).copy()
        elif 'numpy.ndarray' in str(type(filename)):
            assert len(filename.shape) == 3
            assert filename.shape[2] == 3
            img = filename.copy()
        else:
            raise ValueError(f'Not able to identify file type of filename {str(type(filename))}')


        if template_filename is None:
            template = (np.asarray(pygui.screenshot(region=region))).copy()
        elif type(template_filename) is str:
            template_temp = mpimg.imread(template_filename, 0)
            template1 = template_temp.copy()
            template = template1[region[1]:(region[1] + region[3]),
                       region[0]:(region[0] + region[2]), :].copy()
        elif 'PIL.Image.Image' in str(type(template_filename)):
            template = (np.asarray(template_filename)).copy()
        elif 'numpy.ndarray' in str(type(template_filename)):
            template = template_filename[region[1]:(region[1] + region[3]),
                       region[0]:(region[0] + region[2]), :].copy()
        else:
            raise ValueError(f'Not able to identify file type of template_filename {str(type(template_filename))}. '
                             f'It should be either None, str, PIL.Image, or numpy array')

    d, w, h = img.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
    method = cv.TM_SQDIFF_NORMED
    label_plot = 0
    eval_similar = 0
    count = 0
    dict_found = {}
    label_stop = False
    time_start = time.time()

    # while (eval_similar<find_thresh_hold)&(count_max<5):
    while (not label_stop) & (eval_similar < find_thresh_hold):

        count = count + 1

        # Apply template Matching
        # Template is the big picture, img is the feature to find
        res = cv.matchTemplate(img, template, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        if label_plot == 1:
            plt.suptitle(str(method))

            plt.subplot(131), plt.imshow(res, cmap='gray')
            plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

            matched_image = template[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
            plt.subplot(132), plt.imshow(matched_image, cmap='gray')
            plt.title('Found point'), plt.xticks([]), plt.yticks([])

            plt.subplot(133), plt.imshow(img, cmap='gray')
            plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

            plt.show()

            eval_similar = min_val

            print('min_loc:', min_loc, '\tmin_val:', min_val, '\nmax_loc:', max_loc, '\tmax_val:', max_val, )
            print('selected location:', top_left)

        # The matching result turns out to be smaller than the threshold
        # append this finding
        eval_similar = min_val

        if eval_similar < find_thresh_hold:
            top_list_adjust = (top_left[0] + region[0], top_left[1] + region[1])

            #region_individual = top_list_adjust + (w, h, min_val)
            region_individual = top_list_adjust + (w, h)

            dict_found[region_individual] = eval_similar

            # crops out the found area and keeps matching the rest of the picture
            template[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :] = 0

            if bool_find_one:
                label_stop = True

        time_now = time.time() - time_start
        # if timeout is not reach, force to continue to wait
        if len(dict_found) == 0:
            # print(time_now, timeout, len(region_list), eval_similar)
            eval_similar = 0
            label_stop = True
    if bool_output_full_info:
        return dict_found
    else:
        list_region = sorted(dict_found.keys(), key=lambda x: dict_found[x])
        return list_region


def cv_find_pic(filename, region=(0, 0, 1920, 1080), find_thresh_hold=0.05, label_file=0, template_filename='',
                trial=3, wait_time=0.5):
    for i in range(trial):
        region_list = cv_find_all_pic(
            filename=filename,
            region=region,
            find_thresh_hold=find_thresh_hold,
            bool_find_one=True,
            template_filename=template_filename)
        if len(region_list) > 0:
            break
        else:
            time.sleep(wait_time)

    return region_list[0]


def wait_for_refresh(time_interval=0.2, region=(921, 218, 950, 800), timeout=5):
    time_start = time.time()
    time.sleep(0.15)
    image_ori = np.asarray(pygui.screenshot(region=region))

    label_refreshing = 1

    while (label_refreshing == 1) & ((time.time() - time_start) < timeout):
        time.sleep(time_interval)
        image_current = np.asarray(pygui.screenshot(region=region))

        sum_image_diff = np.sum(np.abs(image_ori - image_current))
        # print(sum_image_diff)
        if sum_image_diff < 255 * 3 * 6:
            label_refreshing = 0
        else:
            image_ori = copy.deepcopy(image_current)
    return True


def cv_wait_for_pic(filename, timeout=5, region=(0, 0, 1920, 1035), find_thresh_hold=0.05, scan_period=0.5, bool_output_full_info=False):
    time_start = time.time()
    label_keep_wait = True
    found_pic_region = cv_find_all_pic(filename=filename, region=region, find_thresh_hold=find_thresh_hold, bool_output_full_info=bool_output_full_info)

    while label_keep_wait & (len(found_pic_region) == 0):

        found_pic_region = cv_find_all_pic(filename=filename, region=region, find_thresh_hold=find_thresh_hold, bool_output_full_info=bool_output_full_info)
        if len(found_pic_region) > 0:
            label_keep_wait = False
        else:
            time_span = time.time() - time_start
            if time_span > timeout:
                label_keep_wait = False
            else:
                time.sleep(scan_period)

    return found_pic_region

