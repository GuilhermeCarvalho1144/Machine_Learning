##Import
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow import keras
import os
import cv2
import random
import pickle
import time


##cuDNN compatible issues
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

##

DATADIR = "PetImages/"
CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50
train_X = []
train_Y = []




def createTrainData():
    '''

    :return: train_data, train_labels (np.array)
    '''
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array_gray = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                # cv2.imshow("Debug", img_array_gray)
                # cv2.waitKey(0)
                nimg_array_gray = cv2.resize(img_array_gray, (IMG_SIZE, IMG_SIZE))
                # cv2.imshow("Debug", nimg_array_gray)
                # cv2.waitKey(0)
                training_data.append([nimg_array_gray, class_num])
            except Exception as e:
                pass
    random.shuffle(training_data)
    train_data = []
    train_labels = []
    for features, labels in training_data:
        train_data.append(features)
        train_labels.append(labels)

    train_data = np.array(train_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    train_labels = np.array(train_labels)
    return train_data, train_labels


def saveTrainData(fileTrainData, fileTrainLabels):
    '''
    :param fileTrainData:
    :param fileTrainLabels:
    :return: void
    '''
    pickle_out_train_data = open("train_data.pickle", "wb")
    pickle.dump(fileTrainData, pickle_out_train_data)
    pickle_out_train_data.close()

    pickle_out_train_labels = open("train_labels.pickle", "wb")
    pickle.dump(fileTrainLabels, pickle_out_train_labels)
    pickle_out_train_labels.close()


def loadTrainData():
    xtrain = []
    pickle_in_train_data = open("train_data.pickle", "rb")
    xtrain = pickle.load(pickle_in_train_data)
    pickle_in_train_data.close()

    ytrain = []
    pickle_in_labels_data = open("train_labels.pickle", "rb")
    ytrain = pickle.load(pickle_in_labels_data)
    pickle_in_labels_data.close()

    return xtrain, ytrain


def trainCNN(train_X, train_Y):
    '''
    :param train_X:
    :param train_Y:
    :return: VOID
    '''

    # testing some different parameters
    # change dense layers 0-1-2
    # change con layers 1-2-3
    # change conv layers size 32-64-128

    dense_layers = [1]#[0, 1, 2]
    conv_layers = [3]#[1, 2, 3]
    layer_sizes = [32]#[32, 64, 128]

    ##Creating the CNNs
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, (time.time()))
                print("\n\n\n {} \n\n\n".format(NAME))
                model = keras.Sequential()
                ##Input Layer
                model.add(keras.layers.Conv2D(layer_size, (3, 3), input_shape=train_X.shape[1:], activation="relu"))
                model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                ##Hiden Conv2D Layers
                for l in range(conv_layer-1):
                    model.add(keras.layers.Conv2D(layer_size, (3, 3), activation="relu"))
                    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
                ##Hiden Flatten Layer
                model.add(keras.layers.Flatten())
                for l in range(dense_layer):
                    model.add(keras.layers.Dense(int(layer_size/2), activation="relu"))
                    keras.layers.Dropout(0.2)
                ##Output Layer
                model.add(keras.layers.Dense(1, activation="sigmoid"))

                ##TensorBoard
                tensorboard = keras.callbacks.TensorBoard(log_dir="logs/{}  ".format(NAME))

                ##Compiling the model
                model.compile(optimizere="adam", loss="binary_crossentropy", metrics=["accuracy"])

                ##Training
                model.fit(train_X, train_Y, batch_size=32, validation_split=0.1, epochs=10, callbacks=[tensorboard], verbose=2)

                ##Salving the model
                model.save("CNN_example.h5")

    return model


def prepareData(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    nimg = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    nimg = np.array(nimg).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    nimg = nimg/255.0
    return nimg



def main():


    try:
        fileTrainData = open("train_data.pickle")
        fileTrainData.close()
        fileTrainLabels = open("train_labels.pickle")
        fileTrainLabels.close()
        print("Carregando imagens....")
        train_X, train_Y = loadTrainData()
        print("Imagens Carregadas")
    except FileNotFoundError:
        print("Criando dataset")
        train_X, train_Y = createTrainData()
        saveTrainData(train_X, train_Y)
        print("Dataset criado")

    ##Normalazing the data
    train_X = train_X/255.0
    print(train_X.shape[1:])
    # model = keras.models.load_model("CNN_example.h5")
    # print("Modelo carregado")

    print("Treinado o modelo")
    model = trainCNN(train_X, train_Y)

    prediction = model.predict(prepareData("test/cat_teste.jpg"))
    print(prediction)
    if prediction > 0.5:
        prediction = 1
    else:
        prediction = 0
    # Plotting the image and the prediction
    image = cv2.imread("test/cat_teste.jpg")
    cv2.imshow("Predction {}".format(CATEGORIES[prediction]), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()

##DEBUG
# plt.imshow(train_data[0], cmap=plt.cm.binary)
# print("label {}".format(train_labels[0]))
# plt.show()