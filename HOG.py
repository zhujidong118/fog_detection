#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.18
# @Author :zhujidong

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def gamma(img):
    return np.power(img / 255.0, 1)

# 渲染图像，即将计算出来的该图像的梯度方向和梯度幅值显示出来
def render_gradient( image, cell_gradient):
    cell_size  = 16
    bin_size = 9
    angle_unit = 180// bin_size
    cell_width =  cell_size / 2
    max_mag = np.array(cell_gradient).max()
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = angle_unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(image, (y1, x1), (y2, x2), int(255 * math.sqrt(abs(magnitude))))
                angle += angle_gap
    return image

# 获取梯度值cell图像，梯度方向cell图像
def div(img, cell_x, cell_y, cell_w):
    cell = np.zeros(shape=(cell_x, cell_y, cell_w, cell_w))
    img_x = np.split(img, cell_x , axis=0)
    for i in range(cell_x):
        img_y = np.split(img_x[i], cell_y, axis=1)
        for j in range(cell_y):
            cell[i][j] = img_y[j]
    return cell

# 获取梯度方向直方图图像，每个像素点有9个值
def get_bins(grad_cell, ang_cell):
    bins = np.zeros(shape=(grad_cell.shape[0], grad_cell.shape[1], 9))
    for i in range(grad_cell.shape[0]):
        for j in range(grad_cell.shape[1]):
            binn = np.zeros(9)
            grad_list = np.int8(grad_cell[i, j].flatten())  # .flatten()为降维函数，将其降维为一维，每个cell中的64个梯度值展平，并转为整数
            ang_list = ang_cell[i, j].flatten()  # 每个cell中的64个梯度方向展平
            ang_list = np.int8(ang_list / 20.0)  # 0-9
            ang_list[ang_list >= 9] = 0
            for m in range(len(ang_list)):
                binn[ang_list[m]] += int(grad_list[m])  # 直方图的幅值
            bins[i][j] = binn

    return bins

# 计算图像HOG特征向量并显示
def hog(img, cell_x, cell_y, cell_w):
    gradient_values_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
    gradient_values_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)  # y
    gradient_magnitude = np.sqrt(np.power(gradient_values_x, 2) + np.power(gradient_values_y, 2))
    gradient_angle = np.arctan2(gradient_values_x, gradient_values_y)
    print(gradient_magnitude.shape, gradient_angle.shape)
    gradient_angle[gradient_angle > 0] *= 180 / 3.14
    gradient_angle[gradient_angle < 0] = (gradient_angle[gradient_angle < 0] + 3.14) * 180 / 3.14
    # plt.imshow()是以图像的大小，显示当前每一个像素点计算出来的梯度方向值
    # plt.imshow(gradient_magnitude ) #显示该图像的梯度大小值
    plt.imshow(gradient_angle ) #显示该图像的梯度方向值
    # 该图像的梯度大小值和方向值只能显示一个，如果用 plt.imshow()想要同时显示，则要分区
    plt.show()
    grad_cell = div(gradient_magnitude, cell_x, cell_y, cell_w)
    ang_cell = div(gradient_angle, cell_x, cell_y, cell_w)
    bins = get_bins(grad_cell, ang_cell)
    hog_image = render_gradient(np.zeros([img.shape[0], img.shape[1]]), bins)
    plt.imshow(hog_image, cmap=plt.cm.gray)
    plt.show()
    feature = []
    for i in range(cell_x - 1):
        for j in range(cell_y - 1):
            tmp = [bins[i, j], bins[i + 1, j], bins[i, j + 1], bins[i + 1, j + 1]]
            tmp -= np.mean(tmp)
            feature.append(tmp.flatten())
    return np.array(feature).flatten()


def main():
    cell_w = 8
    img = cv2.imread('./pic_data/road/road1.jpg', cv2.IMREAD_GRAYSCALE)
    print(img.shape)
    x = img.shape[0] - img.shape[0] % cell_w #找到离原图像行值最近的能够被8整除的数
    y = img.shape[1] - img.shape[1] % cell_w #找到离原图像列值最近的能够被8整除的数
    resizeimg = cv2.resize(img, (y, x), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("resizeimg",resizeimg)
    cell_x = int(resizeimg.shape[0] // cell_w)  # cell行数
    cell_y = int(resizeimg.shape[1] // cell_w)  # cell列数
    gammaimg = gamma(resizeimg) * 255
    feature = hog(gammaimg, cell_x, cell_y, cell_w)
    print(feature.shape)

if __name__ == '__main__':
    main()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

