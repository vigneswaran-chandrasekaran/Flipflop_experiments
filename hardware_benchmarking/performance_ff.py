import tensorflow as tf
import numpy as np
from memory_profiler import profile

class FF(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        super(FF, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.j_h = tf.keras.layers.Dense(self.units)
        self.j_x = tf.keras.layers.Dense(self.units)
        self.k_h = tf.keras.layers.Dense(self.units)
        self.k_x = tf.keras.layers.Dense(self.units)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        j = tf.sigmoid(self.j_x(inputs) + self.j_h(prev_output))
        k = tf.sigmoid(self.k_x(inputs) + self.k_h(prev_output))
        output = j * (1 - prev_output) + (1 - k) * prev_output
        return output, [output]

@profile
def test():
    batch_size = 256
    seq_len = 300
    input_dim = 50
    output_dim = 10

    input = np.random.rand(batch_size, seq_len, input_dim)
    output = np.random.rand(batch_size, output_dim)
    model = tf.keras.Sequential(tf.keras.layers.RNN(FF(output_dim), input_shape=(seq_len, input_dim)))
    model.compile('SGD', 'mse')
    model.fit(input, output, epochs=2, batch_size=batch_size)

test()
