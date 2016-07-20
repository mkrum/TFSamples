#Building a LSTM in TensorFlow
##Example Walk Through

In this tutorial, I am going to be going through how to build a simple LSTM in TensorFlow. In this LSTM, we are going to train an LSTM to predict a sin and cosine curve concurrently. First, we will begin with the necessary imports.

	import numpy as np
	import tensorflow as tf
	from math import sin, cos
	import matplotlib.pyplot as plt

We also need to define some constants that we are going to use later on.

	n_hidden =  128
	n_points =  5
	n_vals   =  2

n_hidden is the number of hidden layers, n_points is the number of points that we are going to have in each input, n_out is the number of points that we are going to project. Next we need to define some tensor variables.

	x = tf.placeholder(tf.float32, [None, n_points, n_vals])
	y = tf.placeholder(tf.float32, [None, n_vals])

	weights =  tf.Variable(tf.truncated_normal([n_hidden, n_vals], stddev=0.1))
	biases  =  tf.Variable(tf.constant(0.1, shape=[n_vals]))

x is the variable that will hold our inputs. This inputs will consist of five sets of two values, one for sin and one for cosine. The y value will hold the target. The target will be the next two values for the functions. 

	x_m =  tf.transpose(x, [1, 0, 2])
	x_m =  tf.reshape(x_m, [-1, n_vals])
	x_m =  tf.split(0, n_points, x_m)

This data needs to be slightly formatted before it is used with some of the tensorflow functions we will be using. Namely, the data can only have rank 2, meaning that it can only have tow different dimensions. This means that since our data is currently in the form (batch_size, 5, 2), we need to condense this to just two dimensions.

	
	lstm_cell       =  tf.nn.rnn_cell.BasicLSTMCell(n_hidden, state_is_tuple=True)
	outputs, states =  tf.nn.rnn(lstm_cell, x_m, dtype=tf.float32)
	y_              =  tf.nn.tanh(tf.matmul(outputs[-1], weights) + biases)

These are the main functions that drive the actual LSTM functionality. We create the LSTM cell using a builtin function called BasicLSTMCell. We then pass this cell along with our inputs into a rnn function.
	
	y_m = tf.reshape(y, [-1, n_vals])

We need to reshape our targets now, to make sure that they match the dimensions of our output.	
	
	cost = tf.reduce_mean(tf.square(y_m - y_))
	optimizer =  tf.train.AdamOptimizer(.001).minimize(cost)

Here we are just defining the standard cost and optimizer functions.
	
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
	
	    in_x = sess.run(x_m, feed_dict={x: test_vals})
	    projected_y = sess.run(y_, feed_dict={x: test_vals})
	    target_y = sess.run(y_m, feed_dict={y: test_targets})
	    plt.plot(target_y, 'ro')
	    plt.plot(projected_y, 'bo')
	    plt.show()

Now we actually run the model. The majority of this code is specific to the dataset, the important part is that we are feeding a numpy array of shape (1900, 5, 2) and (1900, 2) into the model to train and then (100, 5, 2) and (100, 2) to test. If you run the code, you should see that the plot of the projected variables are very similar to the actual values.

##Assignment

 
