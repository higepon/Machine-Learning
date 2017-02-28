# Based on sin curve fitting in http://qiita.com/yukiB/items/5d5b202af86e3c587843

import sys
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from matplotlib.pyplot import show, plot
import pandas as pd
import numpy as np
import math
import random
random.seed(0)

version = 3
weights_dir = "/Users/higepon/Desktop/{0}-{1}".format(sys.argv[0], version)

def train_test_split(df, test_size=0.1, n_prev = 100):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)

def _load_data(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train():
    model, _, _, X_train, y_train = create_model()
    os.makedirs(weights_dir, exist_ok=True)
    filepath = weights_dir + "/{loss:.4f}"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.fit(X_train, y_train, batch_size=600, nb_epoch=300, validation_split=0.05, callbacks=callbacks_list)

def create_model():    
    df = pd.read_csv("../../dont_remove_data/nikkei.csv", header=None)
    df[1] = df[1].apply(lambda x: x / 10000)
    df[1] = df[1].astype('float32')
    X = df[[1]]
    length_of_sequences = 200
    (X_train, y_train), (X_test, y_test) = train_test_split(X, n_prev=length_of_sequences)
    
    in_out_neurons = 1
    hidden_neurons = 300
    model = Sequential()  
    model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))  
    model.add(Dense(in_out_neurons))  
    model.add(Activation("linear"))  
#    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    model.compile(loss="mean_squared_error", optimizer="adam")
    return (model, X_test, y_test, X_train, y_train)

def best_model_path():
    files = os.listdir(weights_dir)
    files.sort()
    return "{0}/{1}".format(weights_dir, files[0])

def upOrDown(df):
    ret = []
    for i in range(len(df)):
        if i > 0:
            ret.append(1 if (df[i] - df[i - 1]) > 0 else 0)
    return ret

def upOrDownLoss(a, b):
    ret = 0
    for i in range(len(a)):
        ret += pow(a[i] - b[i], 2)
    return ret / len(a)

def predict():
    model, X_test, y_test, _, _ = create_model()
    model.load_weights(best_model_path())    
    predicted = model.predict(X_test)
    dataf =  pd.DataFrame(predicted)
    dataf.columns = ["predict"]
    dataf["input"] = y_test
    dataf.plot(figsize=(15, 5))
    print("loss:", upOrDownLoss(upOrDown(predicted), upOrDown(y_test)))
    show()    
    
if len(sys.argv) == 2:
    if sys.argv[1] == "--train":
        train()
    elif sys.argv[1] == "--predict":
        predict()
    else:
        print("specify --train or --predict")
else:
        print("specify --train or --predict")

exit(-1)
