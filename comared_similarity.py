#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.15
# @Author :zhujidong
# @Function: 不同能见度等级相似度对比

import numpy as np
import xlrd
import xlwt

def load_data():
    """载入平局梯度数据
    @return:
    """
    data = xlrd.open_workbook('./num_data/standard_100_st.xls')
    table = data.sheet_by_name('Sheet1')
    # 获取行数
    row_num = table.nrows

    y_label = np.array([table.col_values(5)])
    # 将y_label从1*500转为500*1
    y_label = y_label.reshape(y_label.size, 1)
    x_data = np.array([[table.cell_value(i, 4)] for i in range(row_num)])

    return x_data, y_label

def draw_compared_similarity(x_data, y_label):
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
    for target in [0, 1, 2, 3, 4]:
        pos = (y_label == target).ravel()
        # print(pos)
        X = x_data[pos, :]
        ax.scatter(list(range(len(X[:, 0]))), X[:, 0], color=colors[target],
                   marker=markers[target], label=labels[target])
    ax.legend(loc="best")
    fig.suptitle("Fog Level After ORB")
    plt.show()

if __name__ == '__main__':
    x_data, y_data = load_data()
    draw_compared_similarity(x_data, y_data)