"""
A simple code to detect anomalies in ECGs following the example:
https://www.tensorflow.org/tutorials/generative/autoencoder
by: Guilherme Carvalho Pereira
"""
# Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split

class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(140, activation='sigmoid')
        ])
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded= self.decoder(encoded)
        return decoded
        


def load_data():
    '''
    Load data from Google servers
    :return:
    data: ECG values
    label: Ground truth
    '''
    df = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
    df = df.values
    data = df[:,0:-1]
    labels = df[:,-1]
    return data, labels


def data_split(data, labels):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=21)

    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    train_labels = train_labels.astype(bool)
    test_labels = test_labels.astype(bool)

    normal_train_data = train_data[train_labels]
    normal_test_data = test_data[test_labels]

    anomalous_train_data = train_data[~train_labels]
    anomalous_test_data = test_data[~test_labels]
    
    return normal_train_data, normal_test_data, anomalous_train_data, anomalous_test_data, test_data


if __name__ == '__main__':
    print('Start')
    data, labels = load_data()
    normal_train_data, normal_test_data, anomalous_train_data, anomalous_test_data, test_data = data_split(data, labels)
    '''
    #DEBUG
    plt.plot(np.arange(140), normal_train_data[0])
    plt.title("Normal ECG")
    plt.grid('on')
    plt.show()
    plt.plot(np.arange(140), anomalous_test_data[0])
    plt.title("An anomalous ECG")
    plt.grid('on')
    plt.show()
    '''
    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer='adam',loss='mse')

    history = autoencoder.fit(normal_train_data, normal_train_data,
                              epochs=20, batch_size=512,
                              validation_data=(test_data, test_data), shuffle=True)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training Metrics')
    plt.grid('on')
    plt.legend()
    plt.show()

    #test normal
    encoded_data = autoencoder.encoder(normal_test_data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()

    plt.grid()
    plt.plot(normal_test_data[0], 'b')
    plt.plot(decoded_data[0], 'r')
    plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
    plt.legend(['Input', 'Recontruction', 'Error'])
    plt.show()

    # test anomalous
    encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()

    plt.grid()
    plt.plot(anomalous_test_data[0], 'b')
    plt.plot(decoded_data[0], 'r')
    plt.fill_between(np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
    plt.legend(['Input', 'Recontruction', 'Error'])
    plt.show()


    print('End')