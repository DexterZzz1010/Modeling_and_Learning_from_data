import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(x, y):
    """
    x is a simple vector
    """
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)
    return x, y


np.random.seed(123)

# Step 2: Load and Preprocess Data
data = pd.read_csv('training_data2.csv')  # Replace 'your_dataset.csv' with your dataset file path
test_data = pd.read_csv('songs_to_classify2.csv')

# data = data.abs()
X = data.iloc[:, :-1].values  # Features (all columns except the last one)
Y = data.iloc[:, -1].values  # Labels (the last column)
# X = X*1/np.max(np.abs(X),axis=0)

# X = np.delete(X, 3, axis=1)

batchsz = 32

data = data.abs()
# np.random.shuffle(X)
# np.random.shuffle(Y)

# X = tf.convert_to_tensor(X, dtype=tf.float32)
# Y = tf.convert_to_tensor(Y, dtype=tf.float32)

# Split the data into a training set and a test set
x, x_val, y, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)



scaler = StandardScaler()
x = scaler.fit_transform(x)
x_val = scaler.fit_transform(x_val)

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(800).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

# Step 3: Build the Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='tanh'),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(32, activation='tanh'),

    # tf.keras.layers.Dense(256, activation='relu'),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
    # tf.keras.layers.Dense(1)
])
model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
              # loss=tf.losses.CategoricalCrossentropy(from_logits=True),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.build(input_shape=(None, x.shape[1]))
model.summary()

# Step 4: Compile the Model


# Step 5: Train the Model
model.fit(db, epochs=20, validation_data=ds_val)  # Adjust epochs and batch size as needed

# Step 6: Evaluate the Model
test_loss, test_accuracy = model.evaluate(ds_val)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

model.save('my_model4.h5')

# Step 7: Inference

# Assuming you have new, unlabeled data in a variable named 'new_data'
# predictions = model.predict(test_data)
# for i in predictions:
#     print("%.6f" % i)
#
# print(predictions)
# You can use predictions for inference; it will contain the model's output (probabilities).

# test_data = test_data.abs()
# scaler = StandardScaler()
# test_features = test_data.values
# test_features = scaler.fit_transform(test_features)
#
# predictions = model.predict(test_features)
# binary_predictions = ['1' if prediction > 0.5 else '0' for prediction in predictions]
# result = ''.join(binary_predictions)