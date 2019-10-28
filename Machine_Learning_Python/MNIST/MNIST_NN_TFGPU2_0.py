##Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

##Geting the dataset
data = keras.datasets.fashion_mnist

##Definig train and test datasets
(train_images, train_labels), (test_images, test_labels) = data.load_data()

##Definig the possibles classes 0 -> 9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# DEBUG
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()
# print(train_images.shape)


##Normalazing the dataset
train_images = train_images/255.0
test_images = test_images/255.0


##Creating a Model with Keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#Input layer with the size of the images
    keras.layers.Dense(128, activation="relu"),#Hiden layer 17% of the inout layer
    keras.layers.Dense(10, activation="softmax")#Output layer has a probabilistc activation 0 -> 1
])

#Compiling the model using Keras
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Train the model using keras
model.fit(train_images, train_labels, epochs=10)

#
test_loss, teste_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", teste_acc)



