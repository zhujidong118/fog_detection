#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.06.28
# @Author :zhujidong
# @Function: 计算图像平局梯度

import cv2
import time


def average_grad_Scharr(image):
    """
    计算图像平局梯度
    @param image:
    @return:
    """
    sum_grad = 0
    scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
    scharrx = cv2.convertScaleAbs(scharrx)
    # print(scharrx)
    scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)
    scharry = cv2.convertScaleAbs(scharry)
    # print(scharry)
    scharrxy = cv2.addWeighted(scharrx, 1, scharry, 1, 0)
    # print(scharrxy)

    # cv2.imshow('a', image)
    cv2.imshow('scharrx', scharrx)
    cv2.imshow('scharry', scharry)
    cv2.imshow('scharrxy', scharrxy)
    for i in range(len(scharrxy)):
        for j in range(i):
            sum_grad += scharrxy[i][j]
    # print(sum_grad)
    aver_grad = int(sum_grad) // (480 * 640)
    return aver_grad


def average_grad_Sobel(a):
    """
    使用Sobel算子求图像梯度
    @param image:
    @return:
    """
    sobelx = cv2.Sobel(a, cv2.CV_64F, 1, 0, ksize=1)  # 水平梯度
    sobely = cv2.Sobel(a, cv2.CV_64F, 0, 1, ksize=1)  # 竖直梯度
    sobelx = cv2.convertScaleAbs(sobelx)  # 负数取绝对值
    sobely = cv2.convertScaleAbs(sobely)  # 负数取绝对值
    sobelxy = cv2.addWeighted(sobelx, 1, sobely, 1, 0)  # 求sobel算子，系数为0.5，0.5。修正值为0
    # cv2.imshow('a', a)
    # cv2.imshow('sobelx', sobelx)
    # cv2.imshow('sobelx1', sobely)
    cv2.imshow('sobelxy', sobelxy)



if __name__ == '__main__':
    src = cv2.imread('./pic_data/jiashi/4_siji/siji_1.jpg', cv2.IMREAD_GRAYSCALE)
    # src = cv2.imread('./picture/test_grad.jpg', cv2.IMREAD_GRAYSCALE)
    if src is None:
        print('could not load image...\n')
    else:

        start = time.time()

        # print(average_grad_Sobel(src))
        print(average_grad_Scharr(src))

        end = time.time()
        print(str(round((end - start) * 1000, 2)) + 'ms')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
