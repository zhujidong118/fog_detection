#!/usr/bin/env python3.7.3
# @coding :utf-8
# @Time   :2020.07.08
# @Author :zhujidong
# @Function: 使用线性判别构建能见度等级分类模型

from sklearn import datasets, model_selection, discriminant_analysis


###############################################################################
# 用莺尾花数据集
def load_data():
    iris = datasets.load_iris()
    return model_selection.train_test_split(iris.data, iris.target, test_size=0.25, random_state=0,
                                            stratify=iris.target)
    # 返回为: 一个元组，依次为：训练样本集、测试样本集、训练样本的标记、测试样本的标记


###############################################################################
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
    colors = 'rgb'
    markers = 'o*s'
    for target, color, marker in zip([0, 1, 2], colors, markers):
        pos = (y == target).ravel()
        X = converted_X[pos, :]
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], color=color, marker=marker,
                   label="Label %d" % target)
    ax.legend(loc="best")
    fig.suptitle("Fog Level After LDA")
    plt.show()


###############################################################################
import numpy as np

x_train, x_test, y_train, y_test = load_data()
# print(type(y_test))
X = np.vstack((x_train, x_test))  # 沿着竖直方向将矩阵堆叠起来，把训练与测试的数据放一起来看
Y = np.vstack((y_train.reshape(y_train.size, 1), y_test.reshape(y_test.size, 1)))  # 沿着竖直方向将矩阵堆叠起来
# print(Y)
print(X.shape)
lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X, Y.ravel())
print(X.shape)
# converted_X = np.dot(X, np.transpose(lda.coef_)) + lda.intercept_
converted_X = np.dot(X, lda.scalings_[:, 0: 2]) + lda.intercept_
# plot_LDA(converted_X, Y)
print(converted_X.shape)
print(converted_X)
# # print(Y)
# print(lda.coef_)
# print(lda.intercept_)
