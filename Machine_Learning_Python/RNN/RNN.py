import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow import keras
from sklearn import preprocessing
from collections import deque
import random
import time

##cuDNN compatible issues
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

##TARGETS
SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
RATIO_TO_PREDICT = "LTC-USD"

#CONSTANTS
EPOCH = 10
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0


def preprocess_df(df):
    df = df.drop("future", 1)

    for col in df.columns:
        if col != "target":
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)

    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    
    
    # take the last 60 min and try to predict the target
    for i in df.values:
        prev_days.append([n for n in i[:-1]])##no target
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        else:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))

    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys+sells

    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)


def main():
    main_df = pd.DataFrame()
    ratios = ["BCH-USD", "BTC-USD", "ETH-USD", "LTC-USD"]
    for ratio in ratios:
        dataset = f"crypto_data/{ratio}.csv"
        df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])
        df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
        df.set_index("time", inplace=True)
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]

        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)
            
    main_df["future"] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
    main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))
    
    times = sorted(main_df.index.values)
    last_5pct = times[-int(0.05*len(times))]

    validation_main_df = main_df[main_df.index >= last_5pct]
    main_df = main_df[main_df.index >= last_5pct]

    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)

    # DEBUG
    # print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    # print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
    # print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")

    model = keras.Sequential()
    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())

    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.BatchNormalization())

    model.add(tf.compat.v1.keras.layers.CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(32, activation="relu"))
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(2, activation="softmax"))

    opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

    model.compile(loss="sparse_categorical_crossentropy", opttimizer=opt, metrics=["accuracy"])

    tensorboard = keras.callbacks.TensorBoard(log_dir=f"logs/{NAME}")

    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = keras.callbacks.ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                                          mode='max'))  # saves only the best ones

    hist = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCH, validation_data=(validation_x, validation_y), callbacks=[tensorboard])


    score = model.evaluate(validation_x, validation_y, verbose=0)
    print(f"Test lost: {score[0]}")
    print(f"Test accuracy: {score[1]}")

    model.save(f"models/RNN_{NAME}.h5")

if __name__ == "__main__":
    main()