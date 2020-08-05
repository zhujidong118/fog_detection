#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.06.17
# @Author :zhujidong
# @Function: 计算直方图分布矩


import cv2 as cv
import time
import numpy as np
import math
from matplotlib import pyplot as plt


def custom_hist_1(gray_img):
    h, w = gray_img.shape
    hist = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w):
            pv = gray_img[row, col]
            hist[pv] += 1

    M00 = int(sum(hist))
    # 求质心x坐标
    M10 = 0
    for i in range(len(hist)):
        M10 += hist[i] * (i + 1)
    M10 = int(M10 // M00 - 1)
    # 求质心y坐标
    M01 = 0
    for i in range(len(hist)):
        M01 += math.pow(hist[i], 2) / 2
    M01 = int(M01 // M00)

    return M10, M01, int(max(hist))


if __name__ == '__main__':

    src = cv.imread(".\\picture\\room\\road_dark2.jpg", cv.IMREAD_GRAYSCALE)
    if src is None:
        print('could not load image...\n')
    else:
        start = time.time()
        # print(src)
        print(src.shape)
        cv.imshow("input", src)

        print(custom_hist_1(src))

        end = time.time()
        print(str(round((end - start) * 1000, 2)) + 'ms')

    cv.waitKey(0)
    cv.destroyAllWindows()
