#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.05.31
# @Author :zhujidong
import cv2
import numpy as np

img0 = cv2.imread('./picture/room/0.jpg', cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread('./picture/room/road.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('img0', img0)
cv2.imshow('img1', img1)

img0 = img0.reshape(img0.size, order='C')
# 将矩阵转换成向量。按行转换成向量，第一个参数就是矩阵元素的个数
img1 = img1.reshape(img1.size, order='C')

# 计算相关系数
'''
变量矩阵的一行表示一个随机变量；
输出结果是一个相关系数矩阵, results[i][j]表示第i个随机变量与第j个随机变量的相关系数.
np.corrcoef是求两条数据（或者是两个list）数据之间的相关系数（coefficient)
所以就是求了这两列数的相关系数，结果为一个二维矩阵(2*2数组形式)的形式体现，对角线为1，反对角线则为该相关系数。
"[0, 1]"这个代表第0行第一列的那个数值 即为 coefficient
'''
print("Correlation coefficient of image 0 and image 1: %f\n" % np.corrcoef(img0, img1)[0, 1])

cv2.waitKey(0)
cv2.destroyAllWindows()
