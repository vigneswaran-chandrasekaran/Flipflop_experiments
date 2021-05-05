import tensorflow as tf
from convff import ConvFF2D
from data_loader import *
import time

class Model(tf.keras.Model):

    def __init__(self, dim=250):
        super(Model, self).__init__()

        self.convrnn1 = ConvFF2D(256, (11, 11), return_sequences=True)
        self.convrnn1 = ConvFF2D(256, (5, 5))

        self.linear_1 = tf.keras.layers.Dense(256)
        self.linear_2 = tf.keras.layers.Dense(11)
    
    def call(self, x):
        out1 = self.convrnn1(x)
        out2 = self.convrnn2(out1)

        l_out1 = self.linear_1(out2)
        l_out1 = tf.keras.activations.relu(l_out1)

        l_out2 = self.linear_2(l_out1)
        predictions = tf.keras.activations.softmax(l_out2)

        return predictions

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss_fn = tf.keras.losses.CategoricalCrossentropy()

num_epochs = 10
dataloader = DataLoader()

model = Model()

for e in range(num_epochs):

    print("epoch %d" % e)
    dataloader.reset_batch_pointer()
    while not dataloader.finished:
        tic = time.time()
        with tf.GradientTape() as tape:

            x, y = data_loader.spit_data()
            
            out = model(x)
            loss_value = loss_fn(y, out)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if dataloader.idx % 100 == 0:
            print('batches %d, loss %g' % (b, loss_value))
            print("Time elapsed {} s".format(str(round(time.time() - tic, 2))))

    model.save_weights('ff_saved/checkpoint')
