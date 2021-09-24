import tensorflow as tf
import numpy as np
from memory_profiler import profile

@profile
def test():
    batch_size = 256
    seq_len = 300
    input_dim = 50
    output_dim = 10

    input = np.random.rand(batch_size, seq_len, input_dim)
    output = np.random.rand(batch_size, output_dim)
    model = tf.keras.Sequential(tf.keras.layers.LSTM(output_dim, input_shape=(seq_len, input_dim)))
    model.compile('SGD', 'mse')
    model.fit(input, output, epochs=2, batch_size=batch_size)

test()
