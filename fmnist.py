#!/usr/bin/python
import sys, os

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from TRL import *
from util import TNSR

#mnist = input_data.read_data_sets('./data/MNIST_data', one_hot=True)
mnist = input_data.read_data_sets('data/fminst')


def mode_dot(tensor, matrix, mode):
    new_shape = tensor.get_shape().as_list()

    if matrix.get_shape().as_list()[1] != tensor.get_shape().as_list()[mode]:
        raise ValueError("Shape error. {0}(matrix's 2nd dimension) is not as same as {1} (dimension of the tensor)".format(matrix.get_shape().as_list()[1], tensor.get_shape().as_list()[mode]))

    new_shape[mode] = matrix.get_shape().as_list()[0]

    res = tf.matmul(matrix, TNSR.unfold(tensor, mode))

    return TNSR.fold(res, mode, new_shape)

def main(rank = None, file = None):
	

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
		
		# second pooling layer then the size will be 7*7*32
		h_pool2 = max_pool_2x2(h_conv2)


		"""
		Low rank Tensor Regression Layer
		ttrl : Tensor Train Regression Layer
		trl  : Tucker Regression Layer
		cprl : CP Regression Layer
		"""
		out = ttrl(tf.nn.relu(h_pool2), rank, 10)
		#out = cprl(tf.nn.relu(h_pool2), rank, 10)
		#out = trl(tf.nn.relu(h_pool2), rank, 10)

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
					x:batch[0], y_: batch[1]})
				file.write("step %d, training accuracy %g\n"%(i, train_accuracy))
		
			a = sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

		file.write("Final test accuracy %g\n"%accuracy.eval(feed_dict={
			x: mnist.test.images, y_: mnist.test.labels}))


def run(outfilepath, rank, iter):
	with open(outfilepath,"w+") as f:
		for i in range(iter):
			main(rank = rank, file = f)
			tf.reset_default_graph()
		f.write("\n")

if __name__ == '__main__':

	run("./out/TT[1,1,1,10,1].txt", 	[1,1,1,10,1]		, 1)

