#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.14
# @Author :zhujidong
# @Function: 不同能见度等级暗像素对比

import numpy as np
import xlrd
import xlwt


def load_data():
    """载入暗像素数据
    @return:
    """
    data = xlrd.open_workbook('./num_data/all_100.xls')
    table = data.sheet_by_name('Sheet1')
    # 获取行数
    row_num = table.nrows

    y_label = np.array([table.col_values(5)])
    # 将y_label从1*500转为500*1
    y_label = y_label.reshape(y_label.size, 1)
    x_data = np.array(
        [[table.cell_value(i, 0), table.cell_value(i, 1), table.cell_value(i, 2)] for i in range(row_num)])

    return x_data, y_label


def draw_compared_dark_channel_3d(x_data, y_label):
    """不同能见度等级的暗像素分布对比
    @param x_data: 暗像素特征值
    @param y_label: 标签
    @return: 绘制三维图形
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['red', 'blue', 'black', 'green', 'pink']
    markers = 'o*s<>'
    labels = ['visibility = 0', 'visibility = 1', 'visibility = 2', 'visibility = 3',
              'visibility = 4']
    # for target, color, marker in zip([0, 1, 2, 3, 4], colors, markers):
    #     pos = (y_label == target).ravel()
    #     # print(pos)
    #     X = x_data[pos, :]
    #     ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, marker=marker,
    #                label=labels[target])
    for target in [1, 2, 3, 4]:
        pos = (y_label == target).ravel()
        # print(pos)
        X = x_data[pos, :]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=colors[target], marker=markers[target],
                   label=labels[target])
    ax.legend(loc="best")
    fig.suptitle("Fog Level After DC")
    plt.show()


def draw_compared_dark_channel_2d(x_data, y_label):
    """不同能见度等级的暗像素分布对比
    @param x_data: 暗像素特征值
    @param y_label: 标签
    @return: 绘制二维图形
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt
    colors = ['red', 'blue', 'black', 'green', 'pink']
    markers = 'o*s<>'
    labels = ['visibility = 0', 'visibility = 1', 'visibility = 2', 'visibility = 3',
             'visibility = 4']
    # for target, color, marker in zip([0, 1, 2, 3, 4], colors, markers):
    #     pos = (y_label == target).ravel()
    #     X = x_data[pos, :]
    #     ax.scatter(X[:, 0], X[:, 1], color=color, marker=marker,
    #                label=labels[target])
    for target in [1, 2, 3, 4]:
        pos = (y_label == target).ravel()
        # print(pos)
        X = x_data[pos, :]
        ax.scatter(X[:, 0], X[:, 1], color=colors[target],
                   marker=markers[target], label=labels[target])
    ax.legend(loc="best")
    fig.suptitle("Fog Level After DC")
    plt.show()


if __name__ == '__main__':
    x_data, y_label = load_data()
    draw_compared_dark_channel_3d(x_data, y_label)
    draw_compared_dark_channel_2d(x_data, y_label)
