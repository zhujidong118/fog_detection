#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020.05.21
# @Author  : zhujidong

import numpy as np
import cv2
import matplotlib.pyplot as plt

########     四个不同的滤波器    #########
img = cv2.imread('./picture/highway/road1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 均值滤波
img_mean = cv2.blur(img, (7, 7))

# 高斯滤波
img_Guassian = cv2.GaussianBlur(img, (7, 7), 0)

# 中值滤波
img_median = cv2.medianBlur(img, 7)

# 双边滤波
img_bilater = cv2.bilateralFilter(img, -1, 20, 100)

# 展示不同的图片
titles = ['原始图像', '均值滤波', '高斯滤波', '中值滤波', '双边滤波']
imgs = [img, img_mean, img_Guassian, img_median, img_bilater]
titles_1 = ['img', 'img_mean', 'img_Guassian', 'img_median', 'img_bilater']

for i in range(5):
    # plt.subplot(2, 3, i + 1)  # 注意，这和matlab中类似，没有0，数组下标从1开始
    # cv2.imshow("hhh", imgs[i])
    cv2.imwrite("./picture/highway/" + titles_1[i] + ".jpg", imgs[i])
    b, g, r = cv2.split(imgs[i])
    imgs[i] = cv2.merge([r, g, b])
    plt.imshow(imgs[i])
    plt.title(titles[i], fontproperties='SimHei')
    plt.axis('off')
    plt.savefig('./picture/highway/' + titles[i] + '.png')
    plt.show()

# cv2.waitKey(0)
# cv2.destroyAllWindows()