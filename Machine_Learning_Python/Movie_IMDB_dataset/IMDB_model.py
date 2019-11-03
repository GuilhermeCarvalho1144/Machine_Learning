##Imports
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

##Get the datset from keras
data = keras.datasets.imdb

##Creating the train and teste datasets
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)##words used > 10000

##DEBUG
# print(train_data[0])

##Preprocesing the data


##mapint the words
word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0     ##make the same len
word_index["<START>"] = 1   ##start of the text
word_index["<UNK>"] = 2     ##UNKNOWN
word_index["<UNUSED>"] = 3  ##NOT USED

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])##int map to words
##making the data the same len
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=350)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=350)
##DEBUG
# print(len(train_data[0]), len(train_data[1]))


'''
Decoding the text
'''

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

    ##DEBUG
    #print(decode_review(train_data[0]))

model = keras.Sequential()
##Defing the model with keras
def modelKeras(train_data, train_labels, test_data, test_labels):
    model.add(keras.layers.Embedding(88000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(optimizer="adam", loss="binary_crossentropy", metric=["accuracy"])

    ##Doing crossvalidation
    x_val = train_data[:10000]
    x_train = train_data[10000:]

    y_val = train_labels[:10000]
    y_train = train_labels[10000:]

    ##Fit the model with keras
    fitModel = model.fit(x_train, y_train, epochs=80, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    ##Geting the results
    results = model.evaluate(test_data, test_labels)
    print(results)

    ##Printing some prediction
    # test_review = test_data[0]
    # predict = model.predict([test_review])
    # print("Review: ")
    # print(decode_review(test_review))
    # print("Prediction: "+str(predict[0]))
    # print("Actual: "+str(test_labels[0]))

    model.save("IMDB_model_test.h5")


##Train the model
try:
    trainedModel = open("IMDB_model_test.h5")
    trainedModel.close()
    model = keras.models.load_model("IMDB_model_test.h5")
    print("Model trained")
except FileNotFoundError:
    print("Model needs to be train")
    modelKeras(train_data, train_labels, test_data, test_labels)

##Predicting with a review that is not in the dataset

##Encode the new review
def reviewEncode(fileTXT):
    encoded = [1]   ##Start
    for word in fileTXT:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)   ##UNKNOWN
    return encoded

##Open the review

with open("LoTR_IMDB_REVIEW.txt", encoding="utf-8") as f:
	for line in f.readlines():
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
		encode = reviewEncode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])