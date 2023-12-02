import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, BatchNormalization, \
    AveragePooling2D, Dense, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
import datetime
from tensorflow.keras.optimizers import Adam
from resnet import res_identity, res_conv
import os
from tensorflow.keras.datasets import cifar10

class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck'] # from cifar-10 website

def gen_dataset():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train/255.0
    X_test = X_test/255.0

    return X_train, y_train, X_test, y_test


def resnet50(X_train):
    input_im = Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))  # cifar 10 images size
    x = ZeroPadding2D(padding=(3, 3))(input_im)

    # 1st stage
    # here we perform maxpooling, see the figure above

    x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 2nd stage
    # frm here on only conv block and identity block, no pooling

    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage

    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    # ends with average pooling and dense connection

    x = AveragePooling2D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(len(class_types), activation='softmax', kernel_initializer='he_normal')(x)  # multi-class

    # define the model

    model = Model(inputs=input_im, outputs=x, name='Resnet50')

    return model


def run(X_train, y_train, X_test, y_test):
    model = resnet50(X_train)
    model.summary()
    #fine_adam = Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
    batch_size = 64# test with 64, 128, 256
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    resnet_train = model.fit(X_train, y_train,
                                      epochs=160,
                                      steps_per_epoch=X_train.shape[0]/batch_size,
                                      validation_steps=X_test.shape[0]/batch_size,
                                      validation_data=(X_test, y_test),
                                      callbacks=[tensorboard_callback])


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = gen_dataset()
    run(X_train, y_train, X_test, y_test)
