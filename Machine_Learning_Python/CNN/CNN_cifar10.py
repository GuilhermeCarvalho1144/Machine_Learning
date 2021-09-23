import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

def gen_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train/255.0
    X_test = X_test/255.0

    return X_train, y_train, X_test, y_test


def gen_model(X_train, y_train, X_test, y_test):
    model = tf.keras.Sequential()
    model.add(
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=X_train.shape[1:]))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),callbacks=[tensorboard_callback])

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = gen_dataset()
    gen_model(X_train, y_train, X_test, y_test)