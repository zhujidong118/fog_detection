#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.08
# @Author :zhujidong


import xlrd
import xlwt
from typing import List


def normalize_nums_data(nums: List[float]):
    """

    @return:
    """
    max_num, min_num = max(nums), min(nums)
    for i, val in enumerate(nums):
        nums[i] = round((val - min_num) / (max_num - min_num), 4)

    return nums


if __name__ == '__main__':
    data = xlrd.open_workbook('./num_data/all_100.xls')
    data.sheet_names()
    # 获取“Sheet1”工作表的名称及行列内容
    table = data.sheet_by_name('Sheet1')
    # 获取工作表名称
    name = table.name
    # 获取行数
    row_num = table.nrows
    # 获取列数
    col_num = table.ncols

    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'Sheet1', cell_overwrite_ok=True)
    for j in range(5):
        convert_data = normalize_nums_data(table.col_values(j))

        for i in range(len(convert_data)):
            sheet1.write(i, j, convert_data[i])

    cetegory_list = table.col_values(5)
    for i in range(len(cetegory_list)):
        sheet1.write(i, 5, cetegory_list[i])

    f.save('./num_data/all_100_st.xls')
