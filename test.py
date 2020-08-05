#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.06.29
# @Author :zhujidong
# @Function: 支持向量机团雾等级分类


import cv2
import numpy as np



if __name__ == '__main__':
    def equalHist(img):
        # 灰度图像矩阵的高、宽
        h, w = img.shape
        # 第一步：计算灰度直方图
        grayHist = calcGrayHist(img)
        # 第二步：计算累加灰度直方图
        zeroCumuMoment = np.zeros([256], np.uint32)
        for p in range(256):
            if p == 0:
                zeroCumuMoment[p] = grayHist[0]
            else:
                zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
        # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
        outPut_q = np.zeros([256], np.uint8)
        cofficient = 256.0 / (h * w)
        for p in range(256):
            q = cofficient * float(zeroCumuMoment[p]) - 1
            if q >= 0:
                outPut_q[p] = math.floor(q)
            else:
                outPut_q[p] = 0
        # 第四步：得到直方图均衡化后的图像
        equalHistImage = np.zeros(img.shape, np.uint8)
        for i in range(h):
            for j in range(w):
                equalHistImage[i][j] = outPut_q[img[i][j]]
        return equalHistImage


    img = cv.imread("../testImages/4/img1.jpg", 0)
    # 使用自己写的函数实现
    equa = equalHist(blur)
    # grayHist(img, equa)
    # 使用OpenCV提供的直方图均衡化函数实现
    # equa = cv.equalizeHist(img)
    cv.imshow("img", img)
    cv.imshow("equa", equa)
    cv.waitKey()

