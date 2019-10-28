##Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

##Geting the dataset
data = keras.datasets.fashion_mnist

##Definig train and test datasets
(train_images, train_labels), (test_images, teste_labels) = data.load_data()

##Definig the possibles classes 0 -> 9
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# DEBUG
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()
# print(train_images.shape)


##Normalazing the dataset
train_images = train_images/255.0
test_images = teste_labels/255.0




