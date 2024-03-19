#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 26 2021

@author: sparshgupta
"""

import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Dropout, Flatten, Reshape
from tensorflow.keras.layers import LSTM, BatchNormalization
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from tensorflow.keras.regularizers import l1, l2
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae
from attention import Attention

#Loading Data

data = pd.read_excel("file:///Users/sparshg/Desktop/DNAMethyl/Project/Final_Dataset/Healthy.xlsx")

X = data.iloc[:, 1:9].values
y = data.iloc[:, 0].values


#splitting the data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

X_train = X_train.reshape(-1, 1, 8)
X_test = X_test.reshape(-1, 1, 8)

print("Training & Testing Data Shape: ",X_train.shape, X_test.shape, y_train.shape, y_test.shape)




#Model

model = tf.keras.Sequential()

model.add(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(1, 8)))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.LSTM(128, return_sequences=False))

model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Dense(1, activation = 'linear'))


#Compile Model

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01,
                                     beta_1=0.9,
                                     beta_2=0.999,
                                     epsilon=1e-7
                                        )

model.compile(
              loss='mean_absolute_error',
              optimizer=optimizer
              )




#Fitting the model

history = model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=1, validation_split=0.2, shuffle=True)



#Model summary

model.summary()

#Scoring metrics

X_pred = model.predict(X_train)
y_pred = model.predict(X_test)


R = r2_score(y_train, X_pred)
print("R^2 Score on training set:", R)

MAD = mae(y_train, X_pred)
print("Mean Absolute Deviation (MAD) on training set:", MAD)

MSE = mse(y_train, X_pred)
print("Mean Squared Error (MSE) on training set: {:.4f}".format(MSE))

RMSE = math.sqrt(MSE)
print("Root Mean Square Error (RMSE) on training set:", RMSE)

print('\n')

Rt = r2_score(y_test, y_pred)
print("R^2 Score on test set:", Rt)

MADt = mae(y_test, y_pred)
print("Mean Absolute Deviation (MAD) on test set:", MADt)

MSEt = mse(y_test, y_pred)
print("Mean Squared Error (MSE) on test set: {:.4f}".format(MSEt))

RMSEt = math.sqrt(MSEt)
print("Root Mean Square Error (RMSE) on test set:", RMSEt)

#Plots

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,31)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.scatter(y_test, y_pred)
plt.plot(range(105), range(105))
plt.xlabel("Actual age")
plt.ylabel("Predicted age")
plt.title("Results Plot")
plt.show()

