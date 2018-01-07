#!/usr/bin/python
import sys, os

import tensorflow as tf
import numpy as np
from util import TNSR

"""
Tucker regression layer (trl)

Input:
	x 		: the input tensor
	rank 	: hyperparameter. Must be a list that specifies the Tucker rank of the regression tensor W.
	n_outputs : the number of outputs. A scalar. For exmaple, if one's doing MNIST this is 10.
"""
def trl(x, ranks, n_outputs):
	weight_initializer = tf.contrib.layers.xavier_initializer()
	input_shape = x.get_shape().as_list()[1:]

	core,factors = None, None

	core = tf.get_variable("core_last", shape=ranks, initializer = weight_initializer)
	factors = [	tf.get_variable("basic_factor_{0}_{1}".format(i,e),
				shape=(input_shape[i],ranks[i]),
				initializer = weight_initializer)
				for (i, e) in enumerate(input_shape)
				]

	bias = tf.get_variable("bias_trl", shape=(1, n_outputs))

	factors.append(tf.get_variable("factor_{}".format(len(ranks)-1),
			shape=(n_outputs, ranks[-1]),
			initializer = weight_initializer))

	regression_weights = TNSR.tucker_to_tensor(core, factors)

	x_0 = tf.reshape(x, [-1, np.prod(input_shape)])

	w_minus1 = tf.reshape(regression_weights, [np.prod(input_shape),n_outputs])
	return tf.add(tf.matmul(x_0, w_minus1) ,bias)

"""
Tensor Train regression layer (ttrl)

Input:
	x 		: the input tensor
	ranks 	: hyperparameter. Must be a list that specifies the TT rank of the regression tensor W.
	n_outputs : the number of outputs. A scalar. For exmaple, if one's doing MNIST this is 10.
"""
def ttrl(x, ranks, n_outputs):
	weight_initializer = tf.contrib.layers.xavier_initializer()

	suffix = n_outputs
	input_shape = x.get_shape().as_list()[1:]
	bias = tf.get_variable("bias_{}".format(np.prod(n_outputs)), shape=(1, np.prod(n_outputs)))

	cores = []

	for i in range(1, len(ranks)-1):
		cores.append(tf.get_variable("core_{0}_output_{1}".format(i,suffix), 
			shape = (ranks[i-1], input_shape[i-1], ranks[i]), 
			initializer = weight_initializer))

	cores.append(tf.get_variable("core_{0}_last_output_{1}".format(i,suffix), 
		shape=(ranks[-2],n_outputs,ranks[-1]),
		initializer = weight_initializer))

	regression_weights = TNSR.tt_to_tensor(cores)
	return regression(x, regression_weights, input_shape, bias, n_outputs)


"""
CP regression layer (cprl)

Input:
	x 		: the input tensor
	rank 	: hyperparameter. Must be a scalar that specifies the CP rank of the regression tensor W.
	n_outputs : the number of outputs. A scalar. For exmaple, if one's doing MNIST this is 10.
"""
def cprl(x, rank, n_outputs):
	weight_initializer = tf.contrib.layers.xavier_initializer()
	input_shape = x.get_shape().as_list()[1:]
	
	bias = tf.get_variable("bias_{}".format(np.prod(n_outputs)), shape=(1, np.prod(n_outputs)))

	rank1_tnsrs = []

	for i in range(rank):
		rank1_tnsr = []

		for j in range(len(input_shape)):
			rank1_tnsr.append(tf.get_variable("rank1_tnsr_{0}_{1}_{2}".format(i,j,np.prod(n_outputs)), 
				shape = (input_shape[j]), 
				initializer = weight_initializer))

		rank1_tnsr.append(tf.get_variable("rank1_tnsr_{0}_output_{1}".format(i,np.prod(n_outputs)), 
			shape = (n_outputs), 
			initializer = weight_initializer))

		rank1_tnsrs.append(rank1_tnsr)

	regression_weights = TNSR.cp_to_tensor(rank1_tnsrs)
	
	return regression(x, regression_weights, input_shape, bias, n_outputs)

def regression(x, regression_weights, input_shape, bias, n_outputs):

	x_0 = tf.reshape(x, [-1, np.prod(input_shape)])

	w_minus1 = None
	if type(n_outputs) == type([]):
		w_minus1 = tf.reshape(regression_weights, [np.prod(input_shape), np.prod(n_outputs)])
		return tf.reshape(tf.add(tf.matmul(x_0, w_minus1) ,bias), [-1]+n_outputs)
	else:
		w_minus1 = tf.reshape(regression_weights, [np.prod(input_shape),n_outputs])
		return tf.add(tf.matmul(x_0, w_minus1) ,bias)






