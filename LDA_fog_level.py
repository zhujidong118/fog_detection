#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.08
# @Author :zhujidong
# @Function: 使用线性判别构建能见度等级分类模型

from sklearn import datasets, model_selection, discriminant_analysis
import numpy as np
import xlrd
import xlwt


###############################################################################
# train_data
def load_data():
    data = xlrd.open_workbook('./num_data/standard_all_100_st.xls')
    data.sheet_names()
    # 获取“Sheet1”工作表的名称及行列内容
    table = data.sheet_by_name('Sheet1')
    # 获取工作表名称
    name = table.name
    # 获取行数
    row_num = table.nrows
    # 获取列数
    col_num = table.ncols
    x_vals = np.array([[table.cell_value(i, 0), table.cell_value(i, 1), table.cell_value(i, 2),
                        table.cell_value(i, 3), table.cell_value(i, 4)] for i in range(row_num)])
    # x_vals = np.array([[table.cell_value(i, 0), table.cell_value(i, 1), table.cell_value(i, 2)] for i in range(row_num)])
    y_vals = []
    for i in range(row_num):
        y_vals.append(int(table.cell_value(i, 5)))
    y_vals = np.array(y_vals)
    return model_selection.train_test_split(x_vals, y_vals, test_size=0.025, random_state=0,
                                            stratify=y_vals)
    # 返回为: 一个元组，依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记


# test_data
def load_data2():
    data = xlrd.open_workbook('./num_data/standard_all_100_200_st.xls')
    data.sheet_names()
    # 获取“Sheet1”工作表的名称及行列内容
    table = data.sheet_by_name('Sheet1')
    # 获取工作表名称
    name = table.name
    # 获取行数
    row_num = table.nrows
    # 获取列数
    col_num = table.ncols
    x_vals = np.array([[table.cell_value(i, 0), table.cell_value(i, 1), table.cell_value(i, 2),
                        table.cell_value(i, 3), table.cell_value(i, 4)] for i in range(row_num)])
    # x_vals = np.array([[table.cell_value(i, 0), table.cell_value(i, 1), table.cell_value(i, 2)] for i in range(row_num)])
    y_vals = []
    for i in range(row_num):
        y_vals.append(int(table.cell_value(i, 5)))
    y_vals = np.array(y_vals)
    return x_vals, y_vals


###############################################################################
def test_LinearDiscriminantAnalysis(*data):
    x_train, x_test, y_train, y_test = data
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    print('Coefficients:%s, intercept %s' % (lda.coef_, lda.intercept_))  # 输出权重向量w和偏置b
    print('Score: %.2f' % lda.score(x_test, y_test))  # 测试集
    print('Score: %.2f' % lda.score(x_train, y_train))  # 训练集
    test_x, test_y = load_data2()
    print('Score: %.2f' % lda.score(test_x, test_y))

    # 分等级显示正确率
    print('Score: %.2f' % lda.score(test_x[0:100], test_y[0:100]))
    print('Score: %.2f' % lda.score(test_x[100:200], test_y[100:200]))
    print('Score: %.2f' % lda.score(test_x[200:300], test_y[200:300]))
    print('Score: %.2f' % lda.score(test_x[300:400], test_y[300:400]))
    print('Score: %.2f' % lda.score(test_x[400:500], test_y[400:500]))

    # 删除异常点，及补全数据
    flag = []
    for i in range(len(test_y)):
        if lda.predict([test_x[i]]) == test_y[i]:
            flag.append(i)

    print(lda.get_params())
    return lda.predict_proba(test_x), np.array(flag).reshape(np.array(flag).size, 1)

    # print(lda.predict([[0.418, 0.0039, 0.0055, 0.641, 0.4097],
    #                    [0.6393, 0.0063, 0.0176, 0.4872, 0.5049],
    #                    [0.6639, 0.0061, 0.0114, 0.4359, 0.4626],
    #                    [0.5246, 0.006, 0.0584, 0.6154, 0.7684],
    #                    [0.8115, 0.0103, 0.0216, 0.4103, 0.4414],
    #                    [0.4508, 0.0031, 0.012, 0.7436, 0.6122]]))
    # print(lda.predict([[0, 0.9824, 0.9908, 0.9714, 0.9183],
    #                    [0, 0.9781, 0.9885, 0.9714, 0.8979]]))
    # print(lda.predict([[0.8165, 0.01, 0.0182, 0.1714, 0.2134],
    #                    [0.8899, 0.0144, 0.0204, 0.1143, 0.1896],
    #                    [0.8165, 0.0077, 0.0085, 0.4, 0.1975]]))


# 绘制三维图
def plot_LDA(converted_X, y):
    '''
    绘制经过 LDA 转换后的数据
    :param converted_X: 经过 LDA转换后的样本集
    :param y: 样本集的标记
    :return:  None
    '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = ['red', 'blue', 'black', 'green', 'pink']
    markers = 'o*s<>'
    for target, color, marker in zip([0, 1, 2, 3, 4], colors, markers):
        pos = (y == target).ravel()
        # print(pos)
        X = converted_X[pos, :]
        ax.scatter(X[:, 0], X[:, 2], X[:, 1], color=color, marker=marker,
                   label="Label %d" % target)
    ax.legend(loc="best")
    fig.suptitle("Fog Level After LDA")
    plt.show()


# 绘制二维图
def plot_LDA_2(converted_X, y):
    '''
    绘制经过 LDA 转换后的数据
    :param converted_X: 经过 LDA转换后的样本集
    :param y: 样本集的标记
    :return:  None
    '''
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt
    colors = ['red', 'blue', 'black', 'green', 'pink']
    markers = 'o*s<>'
    for target, color, marker in zip([0, 1, 2, 3, 4], colors, markers):
        pos = (y == target).ravel()
        X = converted_X[pos, :]
        ax.scatter(X[:, 1], X[:, 2], color=color, marker=marker,
                   label="Label %d" % target)
    ax.legend(loc="best")
    fig.suptitle("Fog Level After LDA")
    plt.show()


# 删除异常点
def delet(flag):
    data = xlrd.open_workbook('./num_data/all_1000_st.xls')
    # 获取文件中所有工作表的名称
    # data.sheet_names()
    # 获取“Sheet1”工作表的名称及行列内容
    table = data.sheet_by_name('Sheet1')

    f = xlwt.Workbook()

    Sheet1 = f.add_sheet(u'Sheet1', cell_overwrite_ok=False)
    hangshu = 0
    for i in flag:
        select_data = table.row_values(int(i))
        for j in range(len(select_data)):
            Sheet1.write(hangshu, j, select_data[j])
        hangshu += 1

    f.save('./num_data/standard_all_1000_st.xls')


# 将分类概率写入文件
def write_fre(fre):
    """将分类概率写入文件
    @param fre: 概率矩阵
    @return:
    """
    f = xlwt.Workbook()

    Sheet1 = f.add_sheet(u'Sheet1', cell_overwrite_ok=False)
    for i, val in enumerate(fre):
        for j in range(len(val)):
            Sheet1.write(i, j, val[j])

    f.save('./num_data/fre_standard_all_100_200_st.xls')





if __name__ == '__main__':
    # 构建分类器
    x_train, x_test, y_train, y_test = load_data()
    fre, flag = test_LinearDiscriminantAnalysis(x_train, x_test, y_train, y_test)
    # delet(flag)
    # write_fre(fre)

    # 绘制
    # x_train, x_test, y_train, y_test = load_data()
    # X = np.vstack((x_train, x_test))  # 沿着竖直方向将矩阵堆叠起来，把训练与测试的数据放一起来看
    # Y = np.vstack((y_train.reshape(y_train.size, 1), y_test.reshape(y_test.size, 1)))  # 沿着竖直方向将矩阵堆叠起来
    # X, Y = load_data2()
    #
    # lda = discriminant_analysis.LinearDiscriminantAnalysis()
    # converted_X = lda.fit_transform(X, Y.ravel())
    # plot_LDA(converted_X, Y)
    # plot_LDA_2(converted_X, Y)
    # print(converted_X.shape)
