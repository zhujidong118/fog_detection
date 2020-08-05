#!/usr/bin/env python3.7.3
# -*- coding: utf-8 -*-
# @Time    : 2020.05.13
# @Author  : zhujidong

import cv2 as cv
import numpy as np
import time
from functional import custom_hist_1, custom_hist_2


def all_dark_channel_1(image):
    """
    获取三通道图像的暗像素图像
    @param image: image with RGB
    @return: dark channel imaage
    """
    h, w, channel = image.shape
    print("h, w, ch", h, w, channel)
    bian = 7
    dark_image = np.zeros((h, w), np.uint8)
    for row in range(h):
        for col in range(w):
            m = image[row, col, 0]
            row_left, row_right, col_left, col_right = row - bian, row + bian, col - bian, col + bian
            if row_left < 0:
                row_left = 0
            if row_right > h:
                row_right = h
            if col_left < 0:
                col_left = 0
            if col_right > w:
                col_right = w
            for pix_row in range(row_left, row_right):
                for pix_col in range(col_left, col_right):
                    b, g, r = image[pix_row, pix_col]
                    if m > b:
                        m = b
                    if m > g:
                        m = g
                    if m > r:
                        m = r
            dark_image[row, col] = m

    return dark_image


def all_dark_channel_2(image):
    """
    获取单通道暗像素图像
    @param image:
    @return:
    """
    h, w = image.shape
    print("h, w", h, w)
    bian = 7
    dark_image = np.zeros((h, w), np.uint8)
    for row in range(h):
        for col in range(w):
            row_left, row_right, col_left, col_right = row - bian, row + bian, col - bian, col + bian
            if row_left < 0:
                row_left = 0
            if row_right > h:
                row_right = h
            if col_left < 0:
                col_left = 0
            if col_right > w:
                col_right = w
            mask = np.zeros((h, w), np.uint8)
            mask[row_left:row_right, col_left:col_right] = 1
            m, NULL, NULL, NULL = cv.minMaxLoc(image, mask)
            dark_image[row, col] = m

    return dark_image


def all_dark_channel_3(src, r=7):
    """
    使用腐蚀操作进行暗像素图像的获取，速度较快
    @param src:
    @return:
    """
    src = np.min(src, 2)
    dark_image = cv.erode(src, np.ones((2 * r + 1, 2 * r + 1)), borderType=cv.BORDER_REPLICATE)
    return dark_image

if __name__ == '__main__':
    src = cv.imread('./pic_data/road/road4.jpg')
    if src is None:
        print('could not load image...\n')
    else:
        start = time.time()
        src = cv.resize(src, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC)
        # print(src)
        cv.imshow("input", src)

        # output_img = all_dark_channel_1(src)
        # cv.imshow("output", output_img)
        # cv.imwrite(".\\picture\\room\\" + name_file + '_dark.jpg', output_img)
        # custom_hist_1(output_img, ".\\picture\\room\\" + name_file + '_dark_hist.png')
        output_img = all_dark_channel_3(src, 7)
        cv.imshow('output_img', output_img)
        # cv.imwrite('./picture/room/road_dark2.jpg', output_img)

        end = time.time()
        print(str(round(end - start, 2)) + 's')
        # print(end - start)

    cv.waitKey(0)
    cv.destroyAllWindows()
