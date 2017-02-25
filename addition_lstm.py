# Porting http://peterroelants.github.io/posts/rnn_implementation_part02/ using Keras
import sys
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

version = 1
weights_dir = "/Users/higepon/Desktop/{0}".format(version)
nb_timestamps = 7
nb_variables = 2

def create_dataset(nb_samples, sequence_len):
    """Create a dataset for binary addition and return as input, targets."""
    max_int = 2**(sequence_len-1) # Maximum integer that can be added
    format_str = '{:0' + str(sequence_len) + 'b}' # Transform integer in binary format
    nb_inputs = 2  # Add 2 binary numbers
    nb_outputs = 1  # Result is 1 binary number
    X = np.zeros((nb_samples, sequence_len, nb_inputs))  # Input samples
    T = np.zeros((nb_samples, sequence_len, nb_outputs))  # Target samples
    # Fill up the input and target matrix
    for i in range(nb_samples):
        # Generate random numbers to add
        nb1 = np.random.randint(0, max_int)
        nb2 = np.random.randint(0, max_int)
        # Fill current input and target row.
        # Note that binary numbers are added from right to left, but our RNN reads 
        #  from left to right, so reverse the sequence.
        X[i,:,0] = list(reversed([int(b) for b in format_str.format(nb1)]))
        X[i,:,1] = list(reversed([int(b) for b in format_str.format(nb2)]))
        T[i,:,0] = list(reversed([int(b) for b in format_str.format(nb1+nb2)]))
    return X, T

def train():
    # X shape: (2000, 7, 2)
    #  2000: train samples
    #  7: bits
    #  2: x1 and x2
    X_train, T_train = create_dataset(2000, nb_timestamps)
    print('X_train shape: {0}'.format(X_train.shape))
    print('T_train shape: {0}'.format(T_train.shape))
    
    T_train = np.reshape(T_train, (2000, nb_timestamps))
    model = create_model()

    os.makedirs(weights_dir, exist_ok=True)
    
#    filepath="/Users/higepon/Desktop/hage/additions-{epoch:02d}-{loss:.4f}.hdf5"
    filepath = weights_dir + "/{loss:.4f}"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    # fit the model
    model.fit(X_train, T_train, nb_epoch=8000, batch_size=128, callbacks=callbacks_list)

def create_model():
    model = Sequential()
    model.add(LSTM(10, input_shape=(nb_timestamps, nb_variables)))
    model.add(Dropout(0.2))
    model.add(Dense(nb_timestamps, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def best_model_path():
    files = os.listdir(weights_dir)
    files.sort()
    return "{0}/{1}".format(weights_dir, files[0])
    
def predict():
    model = create_model()
    model.load_weights(best_model_path())
    
    # x1:   1010010   37
    # x2: + 1101010   43
    #      -------   --
    # t:  = 0000101   80
    x = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0]).reshape(1, 7, 2)
    prediction = model.predict(x, verbose=0)
    print(np.around(prediction))

best_model_path()
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
