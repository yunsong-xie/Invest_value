# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 10:18:05 2018

@author: yxie8171
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
try:
    import pyautogui as pygui
except:
    a = 1
import datetime
import os
import time
import copy


def cv_find_all_pic(filename, region=(), find_thresh_hold=0.05, label_find_one=False, template_filename='',
                    timeout=0):

    if region == ():
        region = tuple([0, 0] + list(pygui.size()))

    img = mpimg.imread(filename)
    if img.shape[2] == 4:
        img = np.uint8(img[:, :, :-1] * 255)
    else:
        img = np.uint8(img * 255)

    def get_screen_shot(region):
        template = (np.asarray(pygui.screenshot(region=region))).copy()
        return template

    label_screenshot = False
    if type(template_filename) is str:
        if template_filename == '':
            template = get_screen_shot(region)
            label_screenshot = True
        else:
            template_temp = mpimg.imread(template_filename, 0)
            template1 = template_temp.copy()
            template = template1[region[1]:(region[1] + region[3]),
                       region[0]:(region[0] + region[2]), :]
    else:
        template = template_filename.copy()[region[1]:(region[1] + region[3]),
                   region[0]:(region[0] + region[2]), :]

    # plt.imshow(img)

    img2 = img.copy()
    d, w, h = img.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
               'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']



    label_plot = 0
    eval_similar = 0
    count = 0
    region_list = list()
    img = img2.copy()
    label_stop = False
    time_start = time.time()
    # while (eval_similar<find_thresh_hold)&(count_max<5):
    while (not label_stop) & (eval_similar < find_thresh_hold):
        if label_screenshot:
            template = get_screen_shot(region)
            # print('get new screenshot')
        count = count + 1
        for meth in methods[5:]:

            method = eval(meth)
            # Apply template Matching
            res = cv.matchTemplate(img, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc

            bottom_right = (top_left[0] + w, top_left[1] + h)
            # cv.rectangle(template,top_left, bottom_right, 255, 2)

            if label_plot == 1:
                plt.suptitle(meth)

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

            # print(min_val)
            if eval_similar < find_thresh_hold:
                top_list_adjust = (top_left[0] + region[0], top_left[1] + region[1])

                #region_individual = top_list_adjust + (w, h, min_val)
                region_individual = top_list_adjust + (w, h)

                region_list.append(region_individual)

                # print(eval_similar)
                # Finish ploting
                # crops out the found area and keeps matching the rest of the picture
                template[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :] = 0

                if label_find_one:
                    label_stop = True

        time_now = time.time() - time_start
        # if timeout is not reach, force to continue to wait
        if (time_now <= timeout) & (len(region_list) == 0):
            # print(time_now, timeout, len(region_list), eval_similar)
            eval_similar = 0
        else:
            label_screenshot = False



    # time_end=datetime.datetime.now()
    #
    # time_span=(time_end-time_start).seconds+(time_end-time_start).microseconds/1e6
    # print('The found region list is: ')
    # for region in region_list:
    #    print(region)
    # print('This process takes', time_span ,'s.')
    # plt.imshow(template,cmap = 'gray')
    # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    return region_list


def cv_find_pic(filename, region=(0, 0, 1920, 1080), find_thresh_hold=0.05,
                label_file=0, template_filename='',
                trial=3, wait_time=0.5):
    for i in range(trial):
        region_list = cv_find_all_pic(
            filename=filename,
            region=region,
            find_thresh_hold=find_thresh_hold,
            label_find_one=True,
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


def cv_wait_for_pic(filename, timeout=5, region=(0, 0, 1920, 1035), find_thresh_hold=0.05):
    time_start = datetime.datetime.now()
    label_keep_wait = True
    found_pic_region = cv_find_all_pic(filename=filename, region=region, label_find_one=True)

    while label_keep_wait & (len(found_pic_region) == 0):

        found_pic_region = cv_find_all_pic(filename=filename, region=region, label_find_one=True,
                                           find_thresh_hold=find_thresh_hold)
        time_now = (datetime.datetime.now() - time_start).seconds + (
                datetime.datetime.now() - time_start).microseconds / 1e6
        if time_now > timeout:
            label_keep_wait = False

    return found_pic_region[0]


def cv_wait_for_pics(filename, timeout=5, region=(0, 0, 1920, 1035), find_thresh_hold=0.05):
    time_start = datetime.datetime.now()
    label_keep_wait = True
    found_pic_region = cv_find_all_pic(filename=filename, region=region)

    while label_keep_wait & (len(found_pic_region) == 0):

        found_pic_region = cv_find_all_pic(filename=filename, region=region,
                                           find_thresh_hold=find_thresh_hold)
        time_now = (datetime.datetime.now() - time_start).seconds + (
                datetime.datetime.now() - time_start).microseconds / 1e6
        if time_now > timeout:
            label_keep_wait = False

    return found_pic_region



