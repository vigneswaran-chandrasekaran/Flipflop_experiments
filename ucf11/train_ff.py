import tensorflow as tf
from data_loader import *
from convff import ConvFF2D
import time

class Model(tf.keras.Model):

    def __init__(self, dim=250):

        """
        model.add(ConvLSTM2D(filters = 64, kernel_size = (3, 3), return_sequences = False, data_format = "channels_last", input_shape = (seq_len, img_height, img_width, 3)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        """
        super(Model, self).__init__()

        self.convrnn = ConvFF2D(64, (3, 3))
        self.flatten = tf.keras.layers.Flatten()
        self.linear_1 = tf.keras.layers.Dense(256, activation='relu')
        self.linear_2 = tf.keras.layers.Dense(11, activation='softmax')
    
    def call(self, x):

        out1 = self.convrnn(x)
        out1 = self.flatten(out1)        
        l_out1 = self.linear_1(out1)        
        predictions = self.linear_2(l_out1)

        return predictions

optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.CategoricalCrossentropy()
acc_fn = tf.keras.metrics.CategoricalAccuracy()
num_epochs = 10
dataloader = DataLoader(30)


model = Model()
model.load_weights('ff_saved/checkpoint')
loss_monitor = []
acc_monitor = []
for e in range(num_epochs):
    loss_epoch = 0
    acc_epoch = 0
    print("epoch %d" % e)
    dataloader.reset_batch_pointer()
    batch_counter = 0
    while not dataloader.finished:
        tic = time.time()
        with tf.GradientTape() as tape:

            x, y = dataloader.spit_data()
            
            out = model(x)
            loss_value = loss_fn(y, out)
            acc_value = acc_fn(y, out)

            del(x)
            del(y)
            del(out)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
       
        loss_value = round(float(loss_value.numpy()), 3)
        acc_value = round(float(acc_value.numpy()), 3)
        loss_epoch += loss_value
        acc_epoch += acc_value
        
        if batch_counter % 5 == 0:
            
            ta = str(round(time.time() - tic, 2))
            print('B:{} -> L:{}, A:{}, T:{} s'.format(batch_counter,
                                                  loss_value,
                                                  acc_value,
                                                  ta))
        batch_counter += 1
    loss_epoch = loss_epoch / batch_counter
    acc_epoch = acc_epoch / batch_counter
    loss_monitor.append(loss_epoch)
    acc_monitor.append(acc_epoch)

    model.save_weights('ff_saved/checkpoint')
