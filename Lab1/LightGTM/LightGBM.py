# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/16 1:45
@Auth ： Dexter ZHANG
@File ：LightGBM.py
@IDE ：PyCharm
"""

import seaborn as sns

sns.set_style("whitegrid")
sns.color_palette("husl", 10)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


class LightBGM:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def outlier_detection(self):
        # 创建 IsolationForest 模型
        model = IsolationForest(contamination=0.02)  # contamination 参数控制被认为是离群值的比例

        # 使用 fit_predict 方法进行离群值检测
        outliers = model.fit_predict(self.feature_data)  # 假设 features 是你的特征数据

        # 找出被标记为离群值的样本的索引
        outlier_indices = np.where(outliers == -1)

        # 剔除离群值
        filtered_X = np.delete(self.feature_data, outlier_indices, axis=0)
        filtered_y = np.delete(self.label_data, outlier_indices, axis=0)

        return filtered_X, filtered_y

    def load_data(self):
        # 读取数据集
        data = pd.read_csv(self.dataset_path)
        data = data.fillna(0)
        selected_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF',
                            'FullBath', 'TotRmsAbvGrd', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces',
                            'GarageYrBlt', 'BsmtFinSF1', 'LotFrontage', 'SalePrice']
        self.data = data[selected_columns]

    def preprocess_data(self, test_size=0.2):
        # 数据预处理和划分
        X = self.data.drop(columns=['SalePrice'])
        y = self.data['SalePrice']
        scaler = MinMaxScaler()
        self.feature_data = scaler.fit_transform(X)
        self.label_data = y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=1234)

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self):
        # 训练 模型
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',  # 设置提升类型
            'objective': 'regression',
            'metric': 'l2',  # 评估函数
            'max_depth': 20 ,
            'num_leaves': 30,
            'learning_rate': 0.05,  # 学习速率

        }
        self.model = RandomForestRegressor(n_estimators=100, random_state=1234)
        self.model.fit(self.X_train, self.y_train)
        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        # y_pred = self.model.predict(self.X_train)
        # mse = mean_squared_error(self.y_train, y_pred)
        print(f"Mean Squared Error (MSE): {mse}")

    def visualize_results(self):
        # 使用PCA进行降维到3个主成分
        pca = PCA(n_components=3)
        reduced_data = pca.fit_transform(self.feature_data)

        # 预测测试数据的标签
        # test_labels = self.model.predict(self.feature_data)
        # test_labels = self.label_data
        test_labels = self.model.predict(self.feature_data)

        # 创建一个三维散点图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r' if label == 0 else 'b' for label in test_labels]
        ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=colors, marker='o')

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.title(f'SVD+LR Classification')

        plt.show()


def main():
    display = 1
    # 创建 KNNClassifier 实例
    lbgm_model = LightBGM('train.csv')

    # 加载数据
    lbgm_model.load_data()

    # 数据预处理和划分
    lbgm_model.preprocess_data()

    # 训练模型
    lbgm_model.train()


if __name__ == '__main__':
    main()
