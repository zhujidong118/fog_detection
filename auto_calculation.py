#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.06.17
# @Author :zhujidong
# @Function: 自动计算暗像素分布质心、平均梯度、相关性等特征

import os
import cv2
import xlwt
from dark_channel import all_dark_channel_3
from centroid_calculation import custom_hist_1
from average_grad_calculation import average_grad_Scharr
from ORB import img_similarity_1


def dark_calculation(dengji, paiwei):
    """
    自动获取图像暗像素
    @param dengji:
    @param paiwei:
    @return:
    """
    # dengji = '4_siji/'
    path = './pic_data/jiashi/' + dengji
    savepath = './pic_data/jiashi_dark/' + dengji
    filelist = os.listdir(path)
    filelist.sort(key=lambda x: int(x[paiwei:-4]))
    # total_num = len(filelist)

    # print(filelist)
    for i in filelist:
        name_img = path + i
        src = cv2.imread(name_img)
        output_img = all_dark_channel_3(src, 7)
        cv2.imwrite(savepath + i[:-4] + '_dark.jpg', output_img)
        # print(savepath + i[:-4] + '_dark.jpg')


def write_data(dengji, paiwei):
    """
    自动计算暗像素直方图分布中心，并写入.xls文件中
    @param dengji:
    @param paiwei:
    @return:
    """
    # dengji = '4_siji/'
    path = './pic_data/jiashi_dark/' + dengji
    savepath = './num_data/jiashi_dark_centroid/' + dengji[:-1] + '.xls'
    filelist = os.listdir(path)
    filelist.sort(key=lambda x: int(x[paiwei:-9]))
    total_num = len(filelist)

    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    for i in range(total_num):
        name_img = path + filelist[i]
        src = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
        data = custom_hist_1(src)
        # print(data)
        for j in range(len(data)):
            sheet1.write(i, j, data[j])

        print(name_img)
    f.save(savepath)


def average_grad_calculation(dengji, paiwei):
    """
    自动计算图像平均梯度
    @param dengji:
    @param paiwei:
    @return:
    """
    path = './pic_data/jiashi/' + dengji
    savepath = './num_data/average_grad/' + dengji[:-1] + '.xls'
    filelist = os.listdir(path)
    filelist.sort(key=lambda x: int(x[paiwei:-4]))
    total_num = len(filelist)

    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    for i in range(total_num):
        name_img = path + filelist[i]
        src = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
        data = average_grad_Scharr(src)
        sheet1.write(i, 0, data)

        print(name_img)
    f.save(savepath)


def similarity_calculation(image0_path, dengji, paiwei):
    """
    自动计算图像相关性
    @param image0_path:
    @param dengji:
    @param paiwei:
    @return:
    """
    # image0 = cv2.imread(image0_path)
    path = './pic_data/jiashi/' + dengji
    savepath = './num_data/similarity/' + dengji[:-1] + '.xls'
    filelist = os.listdir(path)
    filelist.sort(key=lambda x: int(x[paiwei:-4]))
    total_num = len(filelist)

    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)

    for i in range(total_num):
        src_path = path + filelist[i]
        # src = cv2.imread(name_img, cv2.IMREAD_GRAYSCALE)
        data = img_similarity_1(image0_path, src_path)
        sheet1.write(i, 0, data)


    f.save(savepath)

if __name__ == '__main__':
    # 计算暗像素直方图分布中心
    # dengji = ['0_wu/', '1_yiji/', '2_erji/', '3_sanji/', '4_siji/']
    # paiwei = [3, 5, 5, 6, 5]
    # for i in range(5):
    #     write_data(dengji[i], paiwei[i])

    # 计算图像平局梯度
    # dengji = ['0_wu/', '1_yiji/', '2_erji/', '3_sanji/', '4_siji/']
    # paiwei = [3, 5, 5, 6, 5]
    # for i in range(5):
    #     average_grad_calculation(dengji[i], paiwei[i])

    # 计算图像相似性
    image0_path = ['wu_500.jpg', 'yiji_500.jpg', 'erji_500.jpg', 'sanji_500.jpg', 'siji_500.jpg']
    dengji = ['0_wu/', '1_yiji/', '2_erji/', '3_sanji/', '4_siji/']
    paiwei = [3, 5, 5, 6, 5]
    for i in range(5):
        image0_path = './pic_data/jiashi/' + dengji[i] + image0_path[i]
        similarity_calculation(image0_path, dengji[i], paiwei[i])
