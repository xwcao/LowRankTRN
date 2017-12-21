#!/usr/bin/python
import sys, os

sys.path.append('/home/xcao')

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from TRL import *
from util import TNSR

mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)


def mode_dot(tensor, matrix, mode):
    new_shape = tensor.get_shape().as_list()

    if matrix.get_shape().as_list()[1] != tensor.get_shape().as_list()[mode]:
        raise ValueError("Shape error. {0}(matrix's 2nd dimension) is not as same as {1} (dimension of the tensor)".format(matrix.get_shape().as_list()[1], tensor.get_shape().as_list()[mode]))

    new_shape[mode] = matrix.get_shape().as_list()[0]

    res = tf.matmul(matrix, TNSR.unfold(tensor, mode))

    return TNSR.fold(res, mode, new_shape)

def main(rank = None, file = None):

	print(rank)


	#training = []

	#x = tf.placeholder(tf.float32, shape=[None,3,4,5])

	#out, core, factors, b = trl(x, [6,7,8,9], 10)
	

	with tf.Session() as sess:

		# weight initialzier
		def weight_variable(shape):
			initial = tf.truncated_normal(shape, stddev = 0.1)
			return tf.Variable(initial)
		
		# bias initializer
		def bias_variable(shape):
			initial = tf.constant(0.1, shape = shape)
			return tf.Variable(initial)
		
		# Computes a 2-D convolution with stride of 1
		def conv2d(x, W):
			return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')
		
		def max_pool_2x2(x):
			return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
		
		x = tf.placeholder(tf.float32, shape = [None, 784])
		
		# the correct answer y_
		y_ = tf.placeholder(tf.float32, [None, 10])
		
		# The first layer. Size of 5 by 5 with input channel 1 and output channel of 32
		W_conv1 = weight_variable([5,5,1,16])
		b_conv1 = bias_variable([16])
		
		# reshape the image with some * 28 * 28
		# -1 will be 1
		x_image = tf.reshape(x, [-1,28,28,1])
		
		# the relu layer (activation layer) with input x_image to W_conv1 then add b_conv1
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		
		# pooling layer here the size of the tensor will be 14 * 14 * 32
		h_pool1 = max_pool_2x2(h_conv1)
		
		# Second convolutional layer. Size of 5 by 5 with input 32 to output 64
		W_conv2 = weight_variable([5,5,16,32])
		b_conv2 = bias_variable([32])
		
		# relu layer. input of h_pool to W_conv2 then add bias
		h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)

		#print(h_conv2.get_shape().as_list()) [None, 14, 14, 32]
		
		# second pooling layer then the size will be 7*7*32
		h_pool2 = max_pool_2x2(h_conv2)
		#print(h_pool2.get_shape().as_list())
		

		#x = tf.reduce_mean(h_pool2, reduction_indices=[1, 2], name="avg_pool")
		#convshape = h_pool2.get_shape().as_list()[1:]

		# assumption that the order of output conv is 3
		#weight_initializer = tf.contrib.layers.xavier_initializer()
		#u1 = tf.get_variable("tcl_gap_{}".format(1), shape=[1,convshape[0]],
		#    initializer = weight_initializer)
		#u2 = tf.get_variable("tcl_gap_{}".format(2), shape=[1,convshape[1]],
		#    initializer = weight_initializer)
		#u3 = tf.get_variable("tcl_gap_{}".format(3), shape=[32,convshape[2]],
		#    initializer = weight_initializer)

		#h_pool2 = mode_dot(h_pool2,u1,1)
		#h_pool2 = mode_dot(h_pool2,u2,2)
		#h_pool2 = mode_dot(h_pool2,u3,3)
		#h_pool2 = tf.reshape(h_pool2, [-1, np.prod(h_pool2.get_shape().as_list()[1:])])

		#print(h_pool2.get_shape().as_list())


		# weight and bias variable
		#W_fc1 = weight_variable([32,10])
		W_fc1_gap = weight_variable([32,10])
		b_fc1 = bias_variable([10])
		
		# flatten h_pool2
		#h_pool2_flat = tf.reshape(h_pool2, [-1, 32])

		# gap h_pool2
		conv_gap = tf.reduce_mean(h_pool2, reduction_indices=[1, 2], name="avg_pool")
		
		# flat result will be relu layer after matmul of [1,7*7*64] x [7*7*64,1024] + [1,1024]
		#out = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
		out = tf.matmul(conv_gap, W_fc1_gap) + b_fc1

		#h_fc1 = tf.matmul(tf.reshape(h_conv2, [-1, 14*14*32]), W_fc1) + b_fc1
		#out = tf.nn.softmax(h_fc1)
		
		# place holder
		keep_prob = tf.placeholder(tf.float32)
		
		# dropout of flattened h_fc1 with keep_prob
		#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		
		# initialized variables of [1024, 10] and bias [10]
		#W_fc2 = weight_variable([1024, 10])
		#b_fc2 = bias_variable([10])


		# h_pool2.shape -> (50, 7, 7, 32)
		#out1 = ttrl(h_conv2, rank, [2,2,2,2,2,2]) #[1,75,100,75,1]
		#out = ttrl(tf.nn.relu(out1), [1,5,5,5,5,5,10,1], 10)
		#out1 = trl(h_conv2, rank, [2,2,2,2,2,2]) # [60,30,30,60]
		#out = trl(tf.nn.relu(out1), rank, 10)
		#out1 = cprl(h_conv2, rank, [2,2,2,2,2,2])
		#out = cprl(tf.nn.relu(out1), rank, 10)

		#out = ttrl(tf.nn.relu(h_pool2), rank, 10)
		#out = cprl(tf.nn.relu(h_pool2), rank, 10)
		#out = trl(tf.nn.relu(h_pool2), rank, 10)


		# softmax activation function after multing flat_drop with weight varibale w_fc2 add bias
		#y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

		# cross entropy comparing y_ and y_conv
		cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out))
		
		# train step with adam optimizer
		train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
		
		# check if they are same
		correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y_,1))
		
		# accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		sess.run(tf.initialize_all_variables())

		for i in range(5000):
			batch = mnist.train.next_batch(50)
			if i%100 == 0:
				train_accuracy = accuracy.eval(feed_dict={
					x:batch[0], y_: batch[1], keep_prob: 1.0})
				file.write("step %d, training accuracy %g\n"%(i, train_accuracy))
				#training.append(train_accuracy)
				file.write("step %d, test accuracy %g\n"%(i, accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
		
			a = sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

	
			# h_pool2.shape -> (50, 7, 7, 32)

		file.write("Final test accuracy %g\n"%accuracy.eval(feed_dict={
			x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

	#f.write(training)
	#f.close()

def run(outfilepath, rank, iter):
	with open(outfilepath,"w+") as f:
		for i in range(iter):
			main(rank = rank, file = f)
			tf.reset_default_graph()
		f.write("\n")

if __name__ == '__main__':

	#20877 T
	#20878 CP
	#20879 TT

	#run("./out/fc_gap_tcl_free.txt", 0, 5)
	#run("./out/fc_gap_tcl_fix.txt", 0, 5)

	
	#run("./out/TT[1,1,1,2,1].txt", 		[1,1,1,2,1]			, 5)
	#run("./out/TT[1,1,1,5,1].txt", 		[1,1,1,5,1]			, 5)
	#run("./out/TT[1,1,1,10,1].txt", 	[1,1,1,10,1]		, 5)

	"""
	for j in [1,5,10,20]
		for i in range(2, 8):
			run("./out/TT[1,{0},{1},{2},1].txt".format(i,i,j), 	[1,i,i,j,1]		, 5)
	"""

	#for i in range(8, 33):
	#	run("./out/TT[1,7,{0},10,1].txt".format(i), 	[1,7,i,10,1]		, 5)
	
	
	"""
	for i in range(1,8):
		run("./out/T[{0},{1},{2},10].txt".format(i,i,i), 	[i,i,i,10]		, 5)

	for i in range(8,33):
		run("./out/T[7,7,{},10].txt".format(i), 	[7,7,i,10]		, 5)
	"""


	
	"""
	for i in range(1, 51):
		run("./out/C{}.txt".format(i), 	i		, 5)

	for i in range(60,211,10):
		run("./out/C{}.txt".format(i), 	i		, 5)
	"""
	run("./out/fc_gap.txt", 0, 5)
