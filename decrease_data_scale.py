#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.02
# @Author :zhujidong
# @Function: 将数据规模从1000将为100


import xlrd
import xlwt

if __name__ == '__main__':
    data = xlrd.open_workbook('./num_data/all_1000_st.xls')
    data.sheet_names()
    # 获取“Sheet1”工作表的名称及行列内容
    table = data.sheet_by_name('Sheet1')
    # 获取工作表名称
    name = table.name
    # 获取行数
    row_num = table.nrows
    # 获取列数
    col_num = table.ncols

    data_decrease = []
    for i in range(100, 200):
        data_decrease.append(table.row_values(i))
    for i in range(1100, 1200):
        data_decrease.append(table.row_values(i))
    for i in range(2100, 2200):
        data_decrease.append(table.row_values(i))
    for i in range(3100, 3200):
        data_decrease.append(table.row_values(i))
    for i in range(4100, 4200):
        data_decrease.append(table.row_values(i))

    f = xlwt.Workbook()
    Sheet1 = f.add_sheet(u'Sheet1', cell_overwrite_ok=True)

    for i in range(len(data_decrease)):
        for j in range(6):
            Sheet1.write(i, j, data_decrease[i][j])


    f.save('./num_data/all_100_200_st.xls')



