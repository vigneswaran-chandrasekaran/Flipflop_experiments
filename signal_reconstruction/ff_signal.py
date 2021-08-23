from matplotlib import pyplot as plt
import numpy as np

ts = 200
def return_labels(label):
  label_1 = np.array(label).reshape(1, 1, 3)
  label_1 = np.repeat(label_1, ts, axis=1)
  label_1 = np.repeat(label_1, 32, axis=0)
  return label_1

label_1 = return_labels([0, 0, 1])
label_2 = return_labels([0, 1, 0])
label_3 = return_labels([1, 0, 0])

label = np.vstack((label_1, label_2, label_3))
print(label.shape)

data = np.load('signal_data.npy')
print(data.shape)

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, RNN
from keras.activations import tanh
from keras.layers.wrappers import TimeDistributed

class FF(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super(FF, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.j_h = keras.layers.Dense(self.units)
        self.j_x = keras.layers.Dense(self.units)
        self.k_h = keras.layers.Dense(self.units)
        self.k_x = keras.layers.Dense(self.units)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        j = tf.sigmoid(self.j_x(inputs) + self.j_h(prev_output))
        k = tf.sigmoid(self.k_x(inputs) + self.k_h(prev_output))
        output = j * (1 - prev_output) + (1 - k) * prev_output
        return output, [output]

#X shape: (bs, seq, inp_dim)
#Y shape: (bs, seq, out_dim)
X = label
Y = data
model = Sequential()
model.add(RNN(FF(50), input_shape=(200, 3), return_sequences=True))
model.add(RNN(FF(50), return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())

model.fit(X,Y,epochs=500, verbose=2, batch_size=32, shuffle=True)
