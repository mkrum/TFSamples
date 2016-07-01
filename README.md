#TensorFlow 
This repository is for a small intro tutorial into Tensorflow to be used in a neural network class at the University of Notre Dame.

##1.	Installation

###1.1.	Local Machine
These instructions assume that you are using python 2.7 on a 64-bit CPU only machine.

Mac: 
<pre><code>
$ sudo easy_install –upgrade six
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/tensorflow-0.9.0-py2-none-any.whl 
$ sudo pip install --upgrade $TF_BINARY_URL 
</code></pre>
		
Linux:
<pre><code>
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.9.0-cp27-none-linux_x86_64.whl 
$ sudo pip install --upgrade $TF_BINARY_URL 
</code></pre>
Methods for using virtualenv, anaconda, and docker can be found [here] (https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html)

###1.2.	CRC Machines

The TensorFlow setup on the CRC machines is optimized for its GPU, making it much faster. To run a TensorFlow program on a CRC machine, use the following template for your job script:

[job.script] (https://github.com/mkrum/TFSamples/blob/master/job.script):

<pre><code>

#!/bin/csh  
#$ -q gpu@qa-titanx-001  
#$ -M <your email>
#$ -m abe  
#$ -N <name of your job>
   
module load python/2.7.11  
module load tensorflow/0.8  
module load cuda/7.5  
module load cudnn/v4    
setenv CUDA_VISIBLE_DEVICES 0 

./tfscript.py


</code></pre>



If you want to run two scripts at once, make sure to have the CUDA_VISIBLE_DEVICES set to different values  (0 or 1).

You submit job scripts by using the command: qsub job.script

###1.3.	Test Installation

To ensure your setup works, attempt to run the following code:

[helloworld.py:] (https://github.com/mkrum/TFSamples/blob/master/helloworld.py)

<pre><code>

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

</code></pre>

##2.	What is TensorFlow
From https://www.tensorflow.org/:
TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them. The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. TensorFlow was originally developed by researchers and engineers working on the Google Brain Team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research, but the system is general enough to be applicable in a wide variety of other domains as well.

##3.	TensorFlow Tutorial
###3.1.	Writing code in TensorFlow isn’t like other python programs. Here is a small example using the quadratic equation:

[quadratic.py:] (https://github.com/mkrum/TFSamples/blob/master/quadratic.py)

Classical Python Solution:
<pre><code>
		import math
   	
   		def quadratic_solver(a, b, c):
       			q1 = (-b + math.sqrt(b ** 2.0 - 4.0 * a * c))/(2*a)
       			q2 = (-b - math.sqrt(b ** 2.0 - 4.0 * a * c))/(2*a)
       		  return (q1, q2)
  
 		print quadratic_solver(1, 3, 2)
</code></pre>
	TensorFlow Solution:
<pre><code>
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
      			print sess.run([q1, q2], feed_dict={a: 1.0, b: 3.0, c: 2.0})
</code></pre>

TensorFlow works in two main steps. First, you have to define the graph. Consider this part of the program completely separate from everything else. Everything that occurs in this graph needs to use the Tensorflow specific operations in order for the to be added to the graph. Even things like variables need to be defined through TensorFlow. Once the full graph is created, you need to create a session to run this graph. In the session, you define all of you placeholders through a feed_dict and a dictionary. If you want to think of the graph as a function, these are the input parameters. Instead of defining specific returns from the graph, you pick exactly what values you want to run. For example, I ran sess.run for both q1 and q2. The graph realized that q1 needs discriminant and that the discriminant needs the placeholders. You do not need to specify that any intermediate steps be ran, just define what output values you want to view. To learn more, read through the (FAQ) [https://www.tensorflow.org/versions/r0.9/resources/faq.html]

##4.	MNIST Expert Example Walk Through 
###4.1.	I will be going through the tutorial available [here] (https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html)
You may want to also consider reading through the [beginner version] (https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html)
This tutorial assumes the reader understands at some level the functionality of a convolutional neural network. If this is not the case, I recommend the following introductory reasources:
[here] (http://deeplearning.net/tutorial/lenet.html)
and [here] (http://cs231n.github.io/convolutional-networks/)

The full code can be found [here] (https://github.com/mkrum/TFSamples/blob/master/mnistexample.py)


###4.2.	Loading the Data
Correctly loading the data into the network will be one of the most time consuming parts of building a network. This example uses a pre-loaded version that drastically simplifies things. The dataset is called MNIST and it is a series of 28 by 28 greyscale images of hand written digits. The goal of the network will be to classify these handwritten digits to their actual values.

<pre><code>
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
</code></pre>

When you build your own network, you will need to convert the images into numpy arrays of floats scaled between 0 and 1. The next step is relatively straightforward, declaring the placeholders. 

<pre><code>
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
</code></pre>

x is the image data. It is of shape [None, 784], because it is not specifying the number of images, but it is specifying the they will be of size 784 (total number of pixels in a 28 by 28 greyscale image). y_ is the labels. It is of shape [None, 10] because again, it is not specifying the number of images, but it is specifying the size of the label, which is ten. The label is of size ten because it is being store as a one-hot representation of each digit. This means that 0 is 1000000000, 1 is 0100000000, 2 is 0010000000, and so on. Next, we declare some useful functions for later in the program.

<pre><code>
def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
 
def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
</code></pre>

These functions are just shortcuts for the declaration of weights and bias variables that will be used in the network. Both functions return variables of the requested shape. Weights are randomized using the tf.truncated_normal function, which generates random values. Biases are all initialized at a constant value of 0.1. Next we declare the functions that while define the behavior of this network.

<pre><code>
def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  
def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           	  strides=[1, 2, 2, 1], padding='SAME')
                           	  
</code></pre>
	
Since TensorFlow is a deep learning package, it has many built in functions that are incredibly helpful. Here we see two examples of this. Building a 2-d convolutional layer is accomplished in a single line. This function, conv2d, takes in two parameters, x and W. x is the input to the convolutional layer and W is the weights. The strides are the distance the kernel moves in each direction. SAME means that padding is added along to the edges to make the shapes match as needed. VALID means no padding. The max-pooling layer only takes in the input value, and then returns the modified value. Remember that this value is now of size (width/2.0, height/2.0).  The ksize parameter modifies the size of the values incorporated in the pool. The strides in this case will usually be the same size of the kernel, so each maximum value depends on a wholly independent 2 by 2 square. 

<pre><code>
x_image = tf.reshape(x, [-1,28,28,1])
</code></pre>

First step in the network is to make sure that the image is in the right shape. These Images are fed into the network as a flat array. Reshape puts them back into the shape of the image. The first dimension, –1, is a placeholder, because again we are not specifying the total number of images that we are putting into the network. The next two dimensions are the width and height of the image, 28 by 28. The last dimension is the number of channels in the image. Since these images are grey-scale, this value is 1. For a color image, this value will be 3 for the red, green, and blue values. The next step will be defining the first convolutional layer.

<pre><code>
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
</code></pre>

First we can use the functions we defined earlier to create the weights and bias variables for this layer. The first two dimensions, 5 by 5, is the dimension of the kernel for this convolution.  The next dimension is the number of channels in the image, which again is one. The final dimension is the number of feature maps you would like to produce, in this case 32. The bias needs to only be one dimension, the size of the feature map. The output of the layer is h_conv1, uses the activation function RELU, or rectifier linear unit.  Then we pool the output. 

<pre><code>
h_pool1 = max_pool_2x2(h_conv1)

	This pattern will be repeated again. 

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
 
h_pool2 = max_pool_2x2(h_conv2)
</code></pre>

The only structural difference is the dimensions on the weights and bias. For the weights, the kernel size remains the same, 5 by 5. Now, the second dimension needs to be the size of the feature maps declared in the last layer, 32. The last dimension again is the number of feature maps you want to generate from this layer, in this case 64. Now, once we have the output of this second layer, we need to send this into a fully connected layer. Currently the output is still two dimensional, so we need to flatten it to fit into the network.

<pre><code>
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
</code></pre>

The first dimension is -1, a placeholder for the number of images. The second dimension is the total number of values in the last feature map. We’ve pooled twice, so the map is of size width/4 by height/4, in this case 7 by 7. So the total number of values is the area of an individual map, 7 multiplied by 7, multiplied by the number of maps, 64. 

<pre><code>
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
</code></pre>

The weight variable takes the dimensions of the input, and the number of neurons you want in this layer. The bias variable has a single dimension, the number of neurons. Before the final softmax layer, we add in a drop out layer.

<pre><code>
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
</code></pre>

This layer is relatively straightforward. The keep_prob is left as a placeholder because we will want to be able to control this easily later. 

<pre><code>
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
</code></pre>

The final outputs, y_conv, will be ten different values, one for each digit, between zero and one. Now we need to tell the network how to handle these outputs. First, we need to define how we are going to measure the accuracy of the network.

<pre><code>
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
</code></pre>

The correct_prediction statement is basically a Boolean defined as whether the maximum values in the predicated value and the actual value are at the same index for every value in y_conv and y_. To compute the accuracy, the Booleans are all converted to 1.0’s and 0.0’s. Then it calculates the average of these values, effectively calculating the success rate. 

<pre><code>
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
</code></pre>

The error is calculated using categorical cross entropy. The train_step defines how the network trains itself to minimize the measure of error. 1e-4 is the step size of the optimizer. Now all we need to do is actually run the model.
<pre><code>
with tf.Session() as sess:
     sess.run(tf.initialize_all_variables())
     for i in range(20000):
         batch = mnist.train.next_batch(50)
         if i%100 == 0:
             train_accuracy = accuracy.eval(feed_dict={
                 x: batch[0], y_: batch[1], keep_prob: 1.0})
             print("step %d, training accuracy %g"%(i, train_accuracy))
         train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                           
     print("test accuracy %g"%accuracy.eval(feed_dict={
             x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
</code></pre>
This will print the accuracy every 100 steps and the test accuracy after 20,000 steps. 
##5 Project
Project

1)	Create a data handing program that will accomplish the following things

	a)	Load the images and labels from a file

	b)	Split the data into randomized training and testing sets

	c)	Convert the images into a numpy array

	d)	Easily obtain batches of specific size for both the train and test

2)	Create a network with three convolutional layers and one fully connected layer that can handle images from your dataset. Feel free to use the example above as a base. Pick some arbitrary, but reasonable values for your variables

3)	Record your results for the initial network. Begin to start to experiment with various different network structures and variable values. Record your results from all of the experiments and visualize them. 

4)	Write a short report about what you found to be the optimal structure along with the effects you found various changes had on your network. Try to explain the reasons behind any differences you observed. Include graphs to give a visual representation of the changes.

