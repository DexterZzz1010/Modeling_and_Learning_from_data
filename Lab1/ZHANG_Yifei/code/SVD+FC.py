# -*- coding: utf-8 -*-
"""
@Time ： 2023/9/28 11:18
@Auth ： Dexter ZHANG
@File ：SVD+FC.py
@IDE ：PyCharm
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def preprocess(x, y):
    """
    x is a simple vector
    """
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y

def SVD(X, N=9):
    U, S, VT = np.linalg.svd(X)
    # 重构数据，仅保留前N个奇异值
    reconstructed_data = np.dot(U[:, :N], np.dot(np.diag(S[:N]), VT[:N, :]))
    return reconstructed_data

np.random.seed(123)

# Step 1: Load and Preprocess Data
data = pd.read_csv('training_data.csv')  
test_data = pd.read_csv('songs_to_classify.csv')

# data = data.abs()
X = data.iloc[:, :-1].values  # Features (all columns except the last one)
Y = data.iloc[:, -1].values  # Labels (the last column)


batchsz = 32

data = data.abs()
np.random.shuffle(X)
np.random.shuffle(Y)


# Split the data into a training set and a test set
x, x_val, y, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)



scaler = MinMaxScaler()
x = scaler.fit_transform(x)
x_val = scaler.fit_transform(x_val)

x= SVD(x)
x_val = SVD(x_val)


db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(800).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

# Step 2: Build the Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])
# Step 3: Compile the Model
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              # loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.build(input_shape=(None, x.shape[1]))
model.summary()


# Step 4: Train the Model
model.fit(db, epochs=180, validation_data=ds_val)  # Adjust epochs and batch size as needed

# Step 5: Evaluate the Model
test_loss, test_accuracy = model.evaluate(ds_val)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Step 6:   Save the Model
model.save('my_model5.h5')
