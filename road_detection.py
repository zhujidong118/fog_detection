#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.16
# @Author :zhujidong
# @Function: 识别道路

import math
import cv2
import numpy as np


def otsu_detection(image):
    """阈值分割图像
    @param image: 单通道图像
    @return:
    """
    ret, th = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow('jj', th)


def image_entropy(src):
    """获取二维熵图像
    @param src: 三通道图像（BGR）
    @return: 单通道图像
    """
    rows, cols, channel = src.shape
    entropy_img = np.zeros((rows, cols))
    print(src.shape)
    for row in range(rows):
        for col in range(cols):
            bgr_sun = int(src[row][col][0]) + int(src[row][col][1]) + int(src[row][col][2])
            if bgr_sun == 0:
                entropy_img[row][col] = 255
                continue
            Rb = int(src[row][col][0]) / bgr_sun
            Rg = int(src[row][col][1]) / bgr_sun
            Rr = int(src[row][col][2]) / bgr_sun
            if Rb == 0:
                hb = 0
            else:
                hb = -Rb * math.log(Rb, 2)
            if Rg == 0:
                hg = 0
            else:
                hg = -Rg * math.log(Rg, 2)
            if Rr == 0:
                hr = 0
            else:
                hr = -Rr * math.log(Rr, 2)
            H = (hb + hg + hr) // 1
            entropy_img[row][col] = H
    # print(entropy_img.shape)
    # return entropy_img
    val_min = min(map(min, entropy_img))
    val_max = max(map(max, entropy_img))
    print(val_max, val_min)
    for row in range(rows):
        for col in range(cols):
            if val_max - val_min == 0:
                entropy_img[row][col] = 255
            else:
                entropy_img[row][col] = (int(entropy_img[row][col]) - val_min) / (val_max - val_min) * 255 // 1
    # print(entropy_img)
    cv2.imshow('output', entropy_img)
    return entropy_img


def regional_growth():
    # 区域生长 programmed by changhao
    from PIL import Image
    import matplotlib.pyplot as plt  # plt 用于显示图片
    import numpy as np

    im = Image.open('./pic_data/road/road1.jpg').convert('L')  # 读取图片
    # im.show()

    im_array = np.array(im)

    # print(im_array)
    [m, n] = im_array.shape

    a = np.zeros((m, n))  # 建立等大小空矩阵
    a[70, 70] = 1  # 设立种子点
    k = 40  # 设立区域判断生长阈值

    flag = 1  # 设立是否判断的小红旗
    while flag == 1:
        flag = 0
        lim = (np.cumsum(im_array * a)[-1]) / (np.cumsum(a)[-1])
        for i in range(2, m):
            for j in range(2, n):
                if a[i, j] == 1:
                    for x in range(-1, 2):
                        for y in range(-1, 2):
                            if a[i + x, j + y] == 0:
                                if abs(im_array[i + x, j + y] - lim) <= k:
                                    flag = 1
                                    a[i + x, j + y] = 1

    data = im_array * a  # 矩阵相乘获取生长图像的矩阵
    new_im = Image.fromarray(data)  # data矩阵转化为二维图片

    # if new_im.mode == 'F':
    #    new_im = new_im.convert('RGB')
    # new_im.save('new_001.png') #保存PIL图片

    # 画图展示
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    plt.subplot(1, 2, 2)
    plt.imshow(new_im, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def complete_road(image, src_image):
    """补全初步识别的道路区域并整理边缘
    @param image: 单通道图像或三通道图像
    @param src_image: 源图像，绘制区域轮廓
    @return:
    """
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.bilateralFilter(gray_image, -1, 20, 100)  # 双边滤波
    ret, bin_image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)  # 二值化
    print(ret)
    # cv2.imshow('bin_image', bin_image)

    # 闭操作（填补孔洞）
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close_image = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel1, iterations=2)
    # cv2.imshow('close_image', close_image)

    # 开操作（处理边缘）
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    open_image = cv2.morphologyEx(close_image, cv2.MORPH_OPEN, kernel2, iterations=1)
    # cv2.imshow('open_image', open_image)

    # 绘制外轮廓
    contours, hierarchy = cv2.findContours(open_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(src_image, contours, -1, (0, 0, 255), 4)
    # cv2.imshow('contours_image', src_image)

    # 在源图像上绘制道路区域
    # only_road_image =


    return open_image


def add_image(image1, image2):
    """融合两张图像，大小相同的图像
    @param image1:
    @param image2:
    @return:
    """
    dst = cv2.addWeighted(image1, 1, image2, 1, 0)
    cv2.imshow('add_image', dst)
    cv2.imwrite('./pic_data/road/test_road/road6_dete_1234.jpg', dst)


def k_means(k=8):
    """图像聚类
    @param k: 聚类数
    @return:
    """
    image = cv2.imread('./pic_data/road/test_road/road6.jpg')
    z = image.reshape((-1, 3))
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(image.shape)
    cv2.imshow('res2', res2)
    cv2.imwrite('./pic_data/road/test_road/road6_means.jpg', res2)


def edge_detection(gray_image):
    """图像边缘提取，使用canny算子
    @param gray_image: 单通道图像
    @return:
    """
    filt_iamge = cv2.GaussianBlur(gray_image, (3, 3), 0)
    canny_image = cv2.Canny(filt_iamge, 50, 150)
    cv2.imshow('canny_image', canny_image)


# 标准霍夫线变换
def line_detection_demo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 150, apertureSize=3)
    cv2.imshow('edges', edges)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    for line in lines:
        rho, theta = line[0]  # line[0]存储的是点到直线的极径和极角，其中极角是弧度表示的
        a = np.cos(theta)  # theta是弧度
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))  # 直线起点横坐标
        y1 = int(y0 + 1000 * (a))  # 直线起点纵坐标
        x2 = int(x0 - 1000 * (-b))  # 直线终点横坐标
        y2 = int(y0 - 1000 * (a))  # 直线终点纵坐标
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3, lineType=1)
        print(line)
    cv2.imshow("image_lines", image)


# 统计概率霍夫线变换
def line_detect_possible_demo(image):
    if image.shape[-1] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    edges = cv2.Canny(gray, 0, 100, apertureSize=3)
    cv2.imshow('edges', edges)
    # 函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=20)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("line_detect_possible_demo", image)


if __name__ == '__main__':
    src = cv2.imread('./pic_data/road/road61.jpg')
    src = cv2.imread('./pic_data/road/k_means/road61_means.jpg')
    if src is None:
        print('could not load image...\n')
    else:
        cv2.imshow('input', src)
        # src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        src_image = cv2.imread('./pic_data/road/road4.jpg')
        # src = image_entropy(src)
        # otsu_detection(src)
        # regional_growth()
        open_image = complete_road(src, src_image)
        open_image = cv2.cvtColor(open_image, cv2.COLOR_GRAY2BGR)
        # k_means(4)
        # edge_detection(src)
        line_detection_demo(open_image)
        # line_detect_possible_demo(open_image)

        # 融合图像
        # image1 = cv2.imread('./pic_data/road/test_road/road6_dete_123.jpg')
        # image2 = cv2.imread('./pic_data/road/test_road/road6_dete_4.jpg')
        # add_image(image1, image2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
