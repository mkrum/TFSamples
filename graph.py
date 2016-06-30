#!/usr/bin/env python2.7
import math

#Normal Python solution

def quadratic_solver(a, b, c):
    q1 = (-b + math.sqrt(b ** 2.0 - 4.0 * a * c))/(2*a)
    q2 = (-b - math.sqrt(b ** 2.0 - 4.0 * a * c))/(2*a)
    return (q1, q2)

print quadratic_solver(1, 3, 2)

# TensorFlow Solution

import tensorflow as tf
#define the inputs for the graph
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.placeholder(tf.float32)

#define the operations in the graph
discriminant = tf.sub(tf.square(b), tf.mul(4.0, tf.mul(a, c)))
q1 = tf.div(tf.add(tf.mul(-1.0, b), tf.sqrt(discriminant)), tf.mul(2.0, a))
q2 = tf.div(tf.sub(tf.mul(-1.0, b), tf.sqrt(discriminant)), tf.mul(2.0, a))

#create a session
with tf.Session() as sess:
    #Initialize the placeholders
    sess.run(tf.initialize_all_variables())
    #Feed values into the graph and print the output
    print sess.run([q1, q2], feed_dict={a: 1.0, b:3.0, c:2.0})
