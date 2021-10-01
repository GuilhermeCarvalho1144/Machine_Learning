#!/usr/bin/python3
# -*- codeing:utf-8 -*-
"""
@author: Guilherme Carvalho Pereira
@project: Machine_Learning 
@file: lstm_autoencoder.py
@time:
@desc:
"""

# libs
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Input, Dropout, Dense, RepeatVector, TimeDistributed
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Load data
dataframe = pd.read_csv('GE.csv')
df = dataframe[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])

# debug
# sns.lineplot(x=df['Date'], y=df['Close'])
# plt.grid('on')
# plt.show()

train, test = df.loc[df['Date'] <= '2003-12-31'], df.loc[df['Date'] > '2003-12-31']

# # debug
# sns.lineplot(x=train['Date'], y=train['Close'])
# plt.grid('on')
# plt.show()
#
# # debug
# sns.lineplot(x=test['Date'], y=test['Close'])
# plt.grid('on')
# plt.show()

# preprocessing the data
# LSTM use sigmoid and tanh so the values must needs to be normalize
scaler = StandardScaler()
scaler = scaler.fit(train[['Close']])

train['Close'] = scaler.transform(train[['Close']])
test['Close'] = scaler.transform(test[['Close']])

# As required for LSTM, we require to reshape an input data into n_samples x timesteps x n_features
# In this example n_features = 2; timesteps = 3; n_sample = 5; 5*2*3=30
seq_size = 30  # Number of timesteps to look back

def to_sequence(x, y, seq_size=1):
    x_values = []
    y_values = []
    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])

    return np.array(x_values), np.array(y_values)

trainX, trainY = to_sequence(train[['Close']], train[['Close']], seq_size)
testX, testY = to_sequence(test[['Close']], test[['Close']], seq_size)

model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.2))

model.add(RepeatVector(trainX.shape[1]))

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(trainX.shape[2])))


model.compile(optimizer='adam', loss='mae')
model.summary()

history = model.fit(trainX, trainY, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label="Training loss")
plt.plot(history.history['val_loss'], label="Validation loss")
plt.legend()
plt.show()

