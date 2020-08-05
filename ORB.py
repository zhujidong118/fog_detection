#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.05.29
# @Author :zhujidong

import cv2
import numpy as np

# 自定义计算两个图片相似度函数
def img_similarity_1(img1_path, img2_path):
    """
    :param img1_path: 图片1路径
    :param img2_path: 图片2路径
    :return: 图片相似度
    """
    try:
        # 读取图片
        img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
        # w, h = img1.shape
        # img1 = img1[0 + 45:h]
        # img2 = img2[0 + 45:h1]
        # cv2.imshow("img1", img1)
        # cv2.imshow("img2", img2)


        # 初始化ORB检测器
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # 提取并计算特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        # knn筛选结果
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
        print(matches)

        # 查看最大匹配点数目
        good = [m for (m, n) in matches if m.distance < 0.75 * n.distance]
        print(len(good))
        print(len(matches))
        similary = float(len(good)) / len(matches)
        # print("(ORB算法)两张图片相似度为:%s" % similary)
        return similary

    except:
        print('无法计算两张图片相似度')
        return '0'


def img_similarity_2(img_file_1, img_file_2):
    """

    @param img_file_1:
    @param img_file_2:
    @return:
    """
    # imgname1 = 'E:/other/test1.jpg'
    # imgname2 = 'E:/other/test2.jpg'

    orb = cv2.ORB_create()

    img1 = cv2.imread(img_file_1)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
    kp1, des1 = orb.detectAndCompute(img1, None)  # des是描述子

    img2 = cv2.imread(img_file_2)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(img2, None)

    hmerge = np.hstack((gray1, gray2))  # 水平拼接
    cv2.imshow("gray", hmerge)  # 拼接显示为gray


    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))

    hmerge = np.hstack((img3, img4))  # 水平拼接
    cv2.imshow("point", hmerge)  # 拼接显示为gray

    # BFMatcher解决匹配
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # 调整ratio
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    print(len(good))
    print(len(matches))
    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    cv2.imshow("ORB", img5)
    cv2.imwrite('./picture/0_4.jpg', img5)


if __name__ == '__main__':
    name0 = './pic_data/jiashi/0_wu/wu_1.jpg'
    name1 = './pic_data/jiashi/1_yiji/yiji_1.jpg'
    name2 = './pic_data/jiashi/2_erji/erji_1.jpg'
    name3 = './pic_data/jiashi/3_sanji/sanji_1.jpg'
    name4 = './pic_data/jiashi/4_siji/siji_10.jpg'
    name5 = './pic_data/jiashi/0_wu/wu_30.jpg'


    img_similarity_2(name0, name4)

    # print(img_similarity_1(name0, name4))

    cv2.waitKey(0)
    cv2.destroyAllWindows()