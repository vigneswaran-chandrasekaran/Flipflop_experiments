import tensorflow as tf
import numpy as np
import model as m
from config import *
import time

from utils import DataLoader, draw_strokes_random_color

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)

args.U = data_loader.max_U
args.c_dimension = len(data_loader.chars) + 1

model = m.Model()
optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
loss_fn = m.compute_custom_loss
#model.load_weights('lstm_validator/checkpoint')

def train_step(x, c_vec, y):
    with tf.GradientTape() as tape:
        out = model([x, c_vec])
        loss_value = loss_fn(y, out)

    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value

for e in range(args.num_epochs):
    print("epoch %d" % e)
    data_loader.reset_batch_pointer()
    for b in range(data_loader.num_batches):
        tic = time.time()
        x, y, c_vec, c = data_loader.next_batch()
        loss_value = train_step(x, c_vec, y)
        del(x)
        del(y)
        del(c_vec)
        del(c)
        toc = str(round(time.time() - tic, 2))
        if b % 50 == 0:
            print('batches %d, loss %g' % (b, loss_value))
            print("Time elpased {} s".format(toc))
    model.save_weights('lstm_validator/checkpoint')
