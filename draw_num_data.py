#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.01
# @Author :zhujidong
# @Function: 绘制分类


import xlrd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = xlrd.open_workbook('./num_data/all.xls')
    data.sheet_names()
    # 获取“Sheet1”工作表的名称及行列内容
    table = data.sheet_by_name('Sheet1')
    # 获取工作表名称
    name = table.name
    # 获取行数
    row_num = table.nrows
    # 获取列数
    col_num = table.ncols
    # print(name, row_num, col_num)
    x_data0 ,y_data0 ,x_data1, y_data1, x_data2, y_data2 = [], [], [], [], [], []
    for i in range(1000):
        temp_x = table.cell(i, 0).value
        temp_y = table.cell(i, 3).value
        x_data0.append(temp_x)
        y_data0.append(temp_y)

    for i in range(1000, 2000):
        temp_x = table.cell(i, 0).value
        temp_y = table.cell(i, 3).value
        x_data1.append(temp_x)
        y_data1.append(temp_y)

    for i in range(2000, 3000):
        temp_x = table.cell(i, 0).value
        temp_y = table.cell(i, 3).value
        x_data2.append(temp_x)
        y_data2.append(temp_y)

    plt.plot(x_data0, y_data0, 'ro', label='0')
    plt.plot(x_data1, y_data1, 'kx', label='1')
    plt.plot(x_data2, y_data2, 'gv', label='2')
    plt.title('Gaussian SVM Results on Iris Data')
    plt.xlabel('Pedal Length')
    plt.ylabel('Sepal Width')
    plt.legend(loc='lower right')
    # plt.ylim([-0.5, 3.0])
    # plt.xlim([3.5, 8.5])
    plt.show()


