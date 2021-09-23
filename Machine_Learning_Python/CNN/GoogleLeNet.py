from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Dense, Input, concatenate, GlobalAveragePooling2D, AveragePooling2D, Flatten
import cv2
from tensorflow.keras.utils import to_categorical
import math
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import LearningRateScheduler
import datetime
import tensorflow as tf
import os
from inception import inception_module

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epochs_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def load_cifar10_data(img_rows, img_cols):
    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # Resize training images
    X_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_train[:, :, :, :]]).astype('float32')
    X_valid = np.array([cv2.resize(img, (img_rows, img_cols)) for img in X_valid[:, :, :, :]]).astype('float32')

    # Transform targets to keras compatible format
    Y_train = to_categorical(Y_train, 10)
    Y_valid = to_categorical(Y_valid, 10)

    # preprocess data
    X_train = X_train / 255.0
    X_valid = X_valid / 255.0

    return X_train, Y_train, X_valid, Y_valid


def inception_model(X_train):

    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)

    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    X = Conv2D(64, kernel_size=(7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2',
                kernel_initializer=kernel_init, bias_initializer=bias_init, bias_regularizer=keras.regularizers.l2(0.0001))(input_layer)
    X = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='max_pool_1_3x3/2')(X)
    X = Conv2D(64, kernel_size=(1, 1), padding='same', strides=(1, 1), activation='relu', name='conv_2a_3x3/1',
                kernel_initializer=kernel_init, bias_initializer=bias_init, bias_regularizer=keras.regularizers.l2(0.0001))(X)
    X = Conv2D(192, kernel_size=(3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2b_3x3/1',
               kernel_initializer=kernel_init, bias_initializer=bias_init, bias_regularizer=keras.regularizers.l2(0.0001))(X)
    X = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='max_pool_2_3x3/2')(X)

    X = inception_module(X, 64, 96, 128, 16, 32, 32, name='inception_3a')
    X = inception_module(X, 128, 128, 192, 32, 96, 64, name='inception_3b')

    X = MaxPool2D(pool_size=(3, 3), padding='same', strides=(2, 2), name='max_pool_3_3x3/2')(X)
    X = inception_module(X, 192, 96, 208, 16, 48, 64, name='inception_4a')

    #1st output
    X1 = AveragePooling2D(pool_size=(5, 5), strides=3)(X)
    X1 = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu', bias_regularizer=keras.regularizers.l2(0.0001))(X1)
    X1 = Flatten()(X1)
    X1 = Dense(1024, activation='relu')(X1)
    X1 = Dropout(0.7)(X1)
    X1 = Dense(10, activation='softmax', name='auxiliary_output_1')(X1)

    X = inception_module(X, 160, 112, 224, 24, 64, 64, name='inception_4b')
    X = inception_module(X, 128, 128, 256, 24, 64, 64, name='inception_4c')
    X = inception_module(X, 112, 144, 288, 32, 64, 64, name='inception_4d')

    #2nd output
    X2 = AveragePooling2D(pool_size=(5, 5), strides=3)(X)
    X2 = Conv2D(128, kernel_size=(1, 1),bias_regularizer=keras.regularizers.l2(0.0001))(X2)
    X2 = Flatten()(X2)
    X2 = Dense(1024, activation='relu')(X2)
    X2 = Dropout(0.7)(X2)
    X2 = Dense(10, activation='softmax', name='auxiliary_output_2')(X2)

    X = inception_module(X, 256, 160, 320, 32, 128, 128, name='inception_4e')
    X = MaxPool2D(pool_size=(3, 3), padding='same', strides=(2, 2), name='max_pool_4_3x3/2')(X)

    X = inception_module(X, 256, 160, 320, 32, 128, 128, name='inception_5a')
    X = inception_module(X, 384, 192, 384, 48, 128, 128, name='inception_5b')

    X = GlobalAveragePooling2D(name='avg_pool_5_3x3/1')(X)
    X = Dropout(0.4)(X)

    X = Dense(10, activation='softmax', name='output')(X)

    model = Model(input_layer, [X, X1, X2], name='inception_v1')

    return model

def run(X_train, y_train, X_test, y_test):
    model = inception_model(X_train)
    model.summary()
    epochs = 25
    initial_lrate = 0.01
    sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)

    lr_sc = LearningRateScheduler(decay, verbose=1)
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])
    history = model.fit(X_train, [y_train, y_train, y_train], validation_data=(X_test, [y_test, y_test, y_test]), epochs=epochs, batch_size=256, callbacks=[lr_sc, tensorboard_callback])

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_cifar10_data(100, 100)
    run(X_train, y_train, X_test, y_test)