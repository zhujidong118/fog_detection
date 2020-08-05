#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.02
# @Author :zhujidong
# @Function: 将深度图转为热力图

import cv2
import numpy as np

if __name__ == '__main__':
    org_img = cv2.imread('F:/baidudownload/NYUV2/NYUv2pics/nyu_depths/0.png')
    gray_img = cv2.imread('F:/baidudownload/NYUV2/NYUv2pics/nyu_depths/0.png', cv2.IMREAD_GRAYSCALE)
    norm_img = np.zeros(gray_img.shape)
    cv2.normalize(gray_img, norm_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = np.asarray(norm_img, dtype=np.uint8)

    heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_RAINBOW)  # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像

    img_add = cv2.addWeighted(org_img, 0.3, heat_img, 0.7, 0)
    # 五个参数分别为 图像1 图像1透明度(权重) 图像2 图像2透明度(权重) 叠加后图像亮度
    cv2.imshow('jj', img_add)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
