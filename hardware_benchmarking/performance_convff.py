import tensorflow as tf
import numpy as np
from memory_profiler import profile
from convff import ConvFF2D

@profile
def test():
    batch_size = 64
    seq_len = 300
    input_dim = (16, 16, 3)
    output_dim = 30

    input = np.random.rand(batch_size, seq_len, 16, 16, 3)
    output = np.random.rand(batch_size, output_dim)
    model = tf.keras.Sequential()
    model.add(ConvFF2D(8, (3, 3), input_shape=(seq_len, 16, 16, 3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(output_dim))
    model.compile('SGD', 'mse')
    model.fit(input, output, epochs=2, batch_size=batch_size)
test()
