#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.05.29
# @Author :zhujidong

import cv2 as cv
import time
import numpy as np
from matplotlib import pyplot as plt


def custom_hist_1(image, file_load_name):
    """获取图像
    @param image:
    @param file_load_name:
    @return:
    """
    if image.shape[-1]==3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h, w = image.shape
    hist = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w):
            pv = image[row, col]
            hist[pv] += 1
    y_pos = np.arange(0, 256, 1, dtype=np.int32)
    # plt.bar(y_pos, hist, width=1, align='center', color='g', alpha=0.5)
    plt.bar(y_pos, hist, width=1, color='b', alpha=0.8)
    plt.ylabel('Frequency')
    plt.title('Histogram', y=-0.13)
    # plt.plot(hist, color='r')
    # 设置x轴范围
    plt.xlim([-1, 255])
    # 去除上面和右边的坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(file_load_name)
    plt.show()
    plt.close()
    hist_img = cv.imread(file_load_name)
    return hist_img


def custom_hist_2(gray, file_load_name):
    # 使用matpoltlib.pyplot.hist()绘制直方图
    plt.hist(gray.ravel(), 256, [0, 256])
    plt.ylabel('Frequency')
    plt.title('Histogram', y=-0.13)
    # plt.plot(hist, color='r')
    # 设置x轴范围
    plt.xlim([0, 256])
    # 去除上面和右边的坐标轴
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(file_load_name)
    plt.show()
    plt.close()
    hist_img = cv.imread(file_load_name)
    return hist_img


def image_hist(image):

    cv.imshow("input", image)
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


if __name__ == '__main__':

    src = cv.imread("./pic_data/jiashi_dark/4_siji/siji_1_dark.jpg", cv.IMREAD_GRAYSCALE)
    src1 = cv.imread('./pic_data/road/k_means/road6_means.jpg')
    # src1= cv.imread('./interface/fog_simple_detection/pic_data/road4_dark.jpg')
    if src is None:
        print('could not load image...\n')
    else:
        start = time.time()

        cv.imshow("input", src1)

        # custom_hist_1(src, "./result/siji_1.png")
        custom_hist_1(src1, './pic_data/road/k_means/road6_means_hist.png')
        # image_hist(src1)
        # custom_hist_2(src1, "./pic_data/road/k_means/road1_means_hist.png")

        end = time.time()
        print(str(round((end - start), 2)) + 's')

    cv.waitKey(0)
    cv.destroyAllWindows()
