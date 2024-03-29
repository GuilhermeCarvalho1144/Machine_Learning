##Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
IMG_SIZE = 28
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
train_images = np.array(train_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_images = test_images/255.0
test_images_plot = test_images
test_images = np.array(test_images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
print(train_images.shape)

##Creating a Model with Keras
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),#Input layer with the size of the images
    keras.layers.Dense(128, activation="relu"),#Hiden layer 17% of the inout layer
    keras.layers.Dense(10, activation="softmax")#Output layer has a probabilistc activation 0 -> 1
])
'''
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',input_shape=train_images.shape[1:]))
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

#Compiling the model using Keras
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#Train the model using keras
model.fit(train_images, train_labels, epochs=20)

#Print te accuracy of the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)


##Using the model to make prediction
predictions = model.predict(test_images)

##printing some of the predictions using matplotlib
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols



'''
Function to prent the image and the label predicted in %...case prediction is correct text is blue 
else text is red 
'''
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array),
                                          class_names[true_label]), color=color)



def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images_plot)
    plt.subplot(num_rows, num_cols*2, 2*i+2)
    plot_value_array(i, predictions, test_labels)

plt.show()
