#! /usr/bin/env python2.7

import tensorflow as tf
import numpy as np
import sys
from createData import *


class RNN:
    def __init__(self, inputs, steps, hidden, classes):
        self.inputs     = inputs
        self.steps      = steps
        self.hidden     = hidden
        self.classes    = classes

    def run(self):

        x = tf.placeholder(tf.float32, [None, self.steps, self.inputs])
        y = tf.placeholder(tf.float32, [None, self.classes])
        weights = tf.Variable(tf.truncated_normal([self.hidden, self.classes], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[self.classes]))
        x_m = tf.transpose(x, [1, 0, 2])
        x_m = tf.reshape(x_m, [-1, self.inputs])
        x_m = tf.split(0, self.steps, x_m)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden, forget_bias=1.0, state_is_tuple=True)
        outputs, states = tf.nn.rnn(lstm_cell, x_m, dtype=tf.float32)
        y_ = tf.matmul(outputs[-1], weights) + biases
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
        optimizer = tf.train.AdamOptimizer(.001).minimize(cost)
        correct_pred = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for i in range(2000):
                trainImage, trainLabels = get_n_vals(50)
                trainImage = trainImage.reshape((-1, self.steps, self.inputs))
                sess.run(optimizer, feed_dict={x: trainImage, y:trainLabels})
                if i % 1000 == 0:
                    acc = sess.run(accuracy, feed_dict={x: trainImage, y: trainLabels})
                    loss = sess.run(cost, feed_dict={x: trainImage, y: trainLabels})
                    print "Iter " + str(i) + ", Minibatch Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(acc)
            print "Optimization Finished!"

            testImages, testLabels = get_n_vals(50)
            testImages = testImages.reshape((-1 , self.steps, self.inputs))
            print "Testing Accuracy:", \
                sess.run(accuracy, feed_dict={x: testImages, y: testLabels})

            while True:
                string = raw_input('Test: ')
                if string == 'exit':
                    exit()
                testar = test_ar(string)
                test = testar.reshape((-1, self.steps, self.inputs))
                pred = sess.run(y_, feed_dict={x:test})
                ar_to_text(pred)

if __name__ == '__main__':
    test = RNN(14, 4, 128, 14)
    test.run()
