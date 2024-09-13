# -*- coding: utf-8 -*-
"""
This code snippet loads the data, trains a k-NN classifier using the
features "danceability", "key", "loudness", "instrumentalness" and "liveness",
and makes a prediction for the test data. The obtained predictions can be
copy-pasted from the python terminal and uploaded to the leaderboard

@author: Andreas
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd #for reading csv files

training=pd.read_csv('training_data.csv', sep=',')
test=pd.read_csv('songs_to_classify.csv', sep=',')

# select which features to use
features = ['danceability','key','loudness','instrumentalness','liveness']
X_train = training.loc[:,features].values
y_train = training.loc[:,'label'].values
X_test = test.loc[:,features].values

# Normalize data. Can also be done using sklearn methods such as
# MinMaxScaler()
X_trainn = X_train*1/np.max(np.abs(X_train),axis=0)
X_testn = X_test*1/np.max(np.abs(X_train),axis=0)


# note: all inputs/features are treated as quantitative/numeric
# some of the features are perhaps more sensible to treat as
# qualitative/cathegorical. For that sklearn preprocessing methods
# such as OneHotEncoder() can be used

# define the k-NN model. To set n_neighbors in a systematic way, use cross validation!
knnmodel = KNeighborsClassifier(n_neighbors = 5)

# feed it with data and train it
knnmodel.fit(X=X_trainn,y=y_train)

# make predictions
predictions = knnmodel.predict(X=X_testn).reshape(-1,1).astype(int).reshape(1,-1)
print(predictions)