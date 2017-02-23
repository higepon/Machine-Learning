# Porting http://peterroelants.github.io/posts/rnn_implementation_part02/ using Keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

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

# X shape: (2000, 7, 2)
#  2000: train samples
#  7: bits
#  2: x1 and x2
X_train, T_train = create_dataset(2000, 7)
print('X_train shape: {0}'.format(X_train.shape))
print('T_train shape: {0}'.format(T_train.shape))

T_train = np.reshape(T_train, (2000, 7))

model = Sequential()
model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(T_train.shape[1] ,activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

filepath="/Users/higepon/Desktop/additions-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
# fit the model
model.fit(X_train, T_train, nb_epoch=20, batch_size=128, callbacks=callbacks_list)
