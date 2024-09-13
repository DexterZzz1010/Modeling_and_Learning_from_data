import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def SVD(X, N=9):
    U, S, VT = np.linalg.svd(X)
    # 重构数据，仅保留前N个奇异值
    reconstructed_data = np.dot(U[:, :N], np.dot(np.diag(S[:N]), VT[:N, :]))
    return reconstructed_data

test_data = pd.read_csv('songs_to_classify2.csv')
# test_data = pd.read_csv('training_data.csv')
# test_data = test_data.iloc[:, :-1].values
# test_data = np.delete(test_data, 3, axis=1)
model = tf.keras.models.load_model('my_model2.h5')

scaler = MinMaxScaler()
test_features = scaler.fit_transform(test_data)
# test_features= SVD(test_features)


predictions = model.predict(test_features)
binary_predictions = ['1' if prediction > 0.5 else '0' for prediction in predictions]
result = ''.join(binary_predictions)
print(result)