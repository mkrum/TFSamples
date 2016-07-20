#! /usr/bin/env python2.7

import numpy as np
import tensorflow as tf
from math import sin, cos
import matplotlib.pyplot as plt

n_hidden =  128
n_rows =  5
n_vals = 2
n_outputs = 2

x = tf.placeholder(tf.float32, [None, n_rows, n_vals])
y = tf.placeholder(tf.float32, [None, n_outputs])

weights =  tf.Variable(tf.truncated_normal([n_hidden, n_outputs], stddev=0.1))
biases  =  tf.Variable(tf.constant(0.1, shape=[n_outputs]))

x_m =  tf.transpose(x, [1, 0, 2])
x_m =  tf.reshape(x_m, [-1, n_vals])
x_m =  tf.split(0, n_rows, x_m)

lstm_cell       =  tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
outputs, states =  tf.nn.rnn(lstm_cell, x_m, dtype=tf.float32)
y_              =  tf.nn.tanh(tf.matmul(outputs[-1], weights) + biases)


cost = tf.reduce_mean(tf.square(y - y_))
optimizer =  tf.train.AdamOptimizer(.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    lin      =  np.linspace(0, 100, 2000)
    sin_vals =  np.array([ sin(l) for l in lin ])
    cos_vals =  np.array([ cos(l) for l in lin ])
    combined_vals = zip(sin_vals, cos_vals)
    vals     =  []

    for i in range(len(sin_vals) - 5):
        row = []
        for j in range(5):
            row.append([cos_vals[i + j], sin_vals[i + j]])
        vals.append(row)

    vals = np.array(vals)
    targets =  [ v[-1] for v in vals[1:] ]
    vals    =  np.array(vals[:-1])
    targets =  np.reshape(targets, (-1, n_vals))
    test_vals = vals[-100:]
    train_vals = vals[:-100]
    test_targets = targets[-100:]
    train_targets = targets[:-100]

    for i in range(0, 1900):
        sess.run(optimizer, feed_dict={x: train_vals[i:i + 10], y: train_targets[i: i + 10]})

    projected_y = sess.run(y_, feed_dict={x: test_vals})
    plt.plot(test_targets, 'ro')
    plt.plot(projected_y, 'bo')
    plt.show()
