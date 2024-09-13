import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def load_data(data_path):
    # 加载训练数据
    train_data = pd.read_csv(data_path)  # 替换为你的训练数据文件路径
    feature_names = list(
        ["acousticness", "danceability", "duration", "energy", "instrumentalness","key", "liveness", "loudness","mode",
         "speechiness", "tempo", "time_signature","valence"])

    # 提取特征和标签
    X_train = train_data.iloc[:, :-1].values  # 特征矩阵，不包括最后一列

    return X_train,feature_names


def normalize(X_train):
    # 初始化标准化器
    scaler = MinMaxScaler()
    # 对特征进行标准化
    X_train = scaler.fit_transform(X_train)
    return X_train


def main():
    data_path = 'training_data.csv'
    X_train,feature_names = load_data(data_path)
    X_train = normalize(X_train)
    # 使用 SVD 分解
    U, S, VT = np.linalg.svd(X_train)

    # # 获取逆奇异值矩阵
    # S_inverse = np.zeros_like(X_train)
    # S_inverse[:len(S), :len(S)] = np.diag(1 / S)
    #
    # # 重构原始数据
    # original_data_reconstructed = np.dot(np.dot(U, S_inverse), VT)
    #
    # # 计算每一列对应的奇异值大小
    # singular_values_per_column = np.linalg.norm(original_data_reconstructed, axis=0)
    #
    # # 创建包含特征名和奇异值的元组列表
    # feature_singular_value_tuples = [(feature_names[i], singular_values_per_column[i]) for i in
    #                                  range(len(feature_names))]
    #
    # # 按奇异值大小排序特征
    # sorted_features = sorted(feature_singular_value_tuples, key=lambda x: x[1], reverse=True)
    #
    # # 打印排序后的特征名和奇异值
    # for i, (feature_name, singular_value) in enumerate(sorted_features):
    #     print(f"{i + 1}. Feature: {feature_name}, Singular Value = {singular_value:.4f}")

    # 选择要保留的前N个奇异值
    N = 8

    # 重构数据，仅保留前N个奇异值
    reconstructed_data = np.dot(U[:, :N], np.dot(np.diag(S[:N]), VT[:N, :]))

    # 比较重构数据与原始数据，筛选出需要的特征列
    selected_feature_indices = []
    threshold = 0.9 # 选择一个合适的阈值，根据需要调整

    for i in range(X_train.shape[1]):
        original_feature = X_train[:, i]
        reconstructed_feature = reconstructed_data[:, i]

        # 计算特征的相关性（可以使用其他比较方法）
        correlation = np.corrcoef(original_feature, reconstructed_feature)[0, 1]

        # 如果相关性超过阈值，将特征列索引添加到已选特征列表中
        if abs(correlation) > threshold:
            selected_feature_indices.append(i+1)

    # 输出筛选后的特征列索引
    print("Selected Feature Indices:", selected_feature_indices)


if __name__ == '__main__':
    main()
