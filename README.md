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
