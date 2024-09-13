# -*- coding: utf-8 -*-
"""
@Time ： 2023/9/28 3:19
@Auth ： Dexter ZHANG
@File ：SVD+KNN.py
@IDE ：PyCharm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

class KNNClassifier:
    def __init__(self, dataset_path, test_data_path, k=None):
        self.dataset_path = dataset_path
        self.test_data_path = test_data_path
        self.k = k
        self.data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.N = None

    def load_data(self):
        # 读取数据集
        self.data = pd.read_csv(self.dataset_path)
        self.test_data = pd.read_csv(self.test_data_path)

    def SVD(self, X, N=11):
        U, S, VT = np.linalg.svd(X)
        # 重构数据，仅保留前N个奇异值
        reconstructed_data = np.dot(U[:, :N], np.dot(np.diag(S[:N]), VT[:N, :]))
        return reconstructed_data

    def  outlier_detection (self):
        # 创建 IsolationForest 模型
        model = IsolationForest(contamination=0.01)  # contamination 参数控制被认为是离群值的比例

        # 使用 fit_predict 方法进行离群值检测
        outliers = model.fit_predict(self.feature_data)  # 假设 features 是你的特征数据

        # 找出被标记为离群值的样本的索引
        outlier_indices = np.where(outliers == -1)

        # 剔除离群值
        filtered_X = np.delete(self.feature_data, outlier_indices, axis=0)
        filtered_y = np.delete(self.label_data, outlier_indices, axis=0)

        return filtered_X,filtered_y

    def preprocess_data(self, test_size=0.2):
        # 数据预处理和划分
        X = self.data.iloc[:, :-1].values
        y = self.data.iloc[:, -1].values

        scaler = MinMaxScaler()
        self.feature_data = scaler.fit_transform(X)
        self.label_data = y

        filtered_X, filtered_y = self.outlier_detection()

        # 归一化处理


        X_train, X_test, y_train, y_test = train_test_split(filtered_X, filtered_y, test_size=test_size, random_state=4321)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)



        self.X_train = self.SVD(X_train)
        self.y_train = y_train
        self.X_test = self.SVD(X_test)
        self.y_test = y_test

    def train(self):
        # 训练 KNN 模型
        self.model = KNeighborsClassifier(n_neighbors=self.k, weights='distance')
        self.model.fit(self.X_train, self.y_train)
        score = self.model.score(self.X_test, self.y_test)
        print("准确度：", score)

        scores = cross_val_score(self.model, self.SVD(self.feature_data), self.label_data, cv=10,
                                 scoring='accuracy')  # 采用K折交叉验证的方法来验证算法效果
        print('K折准确度:', scores)

    def predict(self, X):
        # 预测
        scaler = MinMaxScaler()
        normalize_X = scaler.fit_transform(X)
        return self.model.predict(normalize_X)

    def visualize_results(self):
        # 使用PCA进行降维到3个主成分
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(self.feature_data)

        # 预测测试数据的标签
        # test_labels = self.model.predict(self.feature_data)
        test_labels = self.model.predict(self.feature_data)

        # 创建一个三维散点图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r' if label == 0 else 'b' for label in test_labels]
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=colors, marker='o')

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.title(f'SVD+KNN Classification (k={self.k})')

        plt.show()


def main():
    display = 0
    # statsmodels\ seaborn
    # # Heat map
    # n_train = train.select_dtypes(include=['float64', 'int64'])
    # plt.figure(figsize=(14, 14))
    # sns.heatmap(n_train.corr(), annot=True, annot_kws={"size": 6}, cmap='coolwarm', linewidths=.5, square=True)
    #
    # correlation_matrix = n_train.corr()
    # cols = correlation_matrix.nlargest(16, 'SalePrice')['SalePrice'].index
    # cm = np.corrcoef(train[cols].values.T)
    # sns.set(font_scale=1.25)
    # plt.figure(figsize=(12, 12))
    # heatmap = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
    #                       yticklabels=cols.values, xticklabels=cols.values)
    # plt.title('Top 16 Correlation Matrix')  # 选出15组与sale price相关性高的数据
    # plt.show()

    # 创建 KNNClassifier 实例
    knn_classifier = KNNClassifier('training_data.csv', 'songs_to_classify.csv', k=5)

    # 加载数据
    knn_classifier.load_data()

    # 数据预处理和划分
    knn_classifier.preprocess_data()

    # 训练 KNN 模型
    knn_classifier.train()

    test_data = knn_classifier.test_data.values

    # 预测
    predicted_labels = knn_classifier.predict(knn_classifier.SVD(test_data))
    array_string = np.array2string(predicted_labels)
    labels = array_string.replace('\n', '').replace(' ', '')
    print("Predicted Labels:", labels)

    # 可视化分类结果
    if display == 1:
        knn_classifier.visualize_results()


if __name__ == '__main__':
    main()
