# -*- coding: utf-8 -*-
"""
@Time ： 2023/9/28 2:48
@Auth ： Dexter ZHANG
@File ：SVD2.py
@IDE ：PyCharm
"""

import numpy as np
from sklearn.model_selection import train_test_split  # 划分数据集
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pandas as pd  # for reading csv files
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class SVD:
    '''
    Using SVM to train the dataset
    best acc: 0.82
    '''

    def __init__(self, feature_columns=[], decision_function_shape='ovo'):
        self.data = pd.read_csv('training_data.csv', sep=',')
        self.data = self.data.sort_values('label', ascending=False)
        self.train_data = self.Normalization()  # 数据集
        # print(self.data)
        # print(self.train_data)
        self.Normalization()
        self.SVD_test()

    def Normalization(self):
        features = self.data.iloc[:, :-1]
        # 特征正则化
        scaler = MinMaxScaler()
        normalized_features = scaler.fit_transform(features)
        return normalized_features

    def SVD_test(self):
        # 对数据进行SVD分解
        U, s, V = np.linalg.svd(self.train_data)

        # 定义要显示的最大奇异值数量
        n = 13

        # 提取最大的n个奇异值和对应的特征向量
        max_singular_indices = np.argsort(s)[-n:]
        max_singular_values = s[max_singular_indices]
        max_singular_features = V[max_singular_indices]

        # 可视化所有奇异值的分布
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 14), s, marker='o', linestyle='-', color='b')
        plt.xlabel('Singular Value Index')
        plt.ylabel('Singular Value')
        plt.title('Distribution of Singular Values')
        plt.grid(True)
        plt.show()

        # 获取与最大奇异值对应的特征在原始数据数组中的位置
        max_singular_features_indices = np.argmax(np.abs(V.T), axis=1)[max_singular_indices]

        # 打印特征在原始数据数组中的位置
        print("Features corresponding to max singular values:")
        for i, feature_index in enumerate(max_singular_features_indices):
            print(f"Max Singular Value {i}: Column {feature_index}")


        # 可视化最大的n个奇异值及其对应的特征
        plt.figure(figsize=(10, 6))
        for i in range(n):
            plt.plot(range(1, 14), max_singular_features[i], marker='o', linestyle='-',
                     label=f'Singular Value: {max_singular_values[i]:.3f}')
        plt.xlabel('Feature Index')
        plt.ylabel('Singular Feature Value')
        plt.title(f'Top {n} Singular Features')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    SVD()