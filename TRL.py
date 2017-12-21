#!/usr/bin/python
import sys, os

sys.path.append('/home/xcao')

import tensorflow as tf
import numpy as np
from util import TNSR


def trl(x, ranks, n_outputs):

	# n_outputs = [2,2,2,2,2,2]
	# ranks 	= [1,2,2,1]
	#FIXED_RANK = 3

	weight_initializer = tf.contrib.layers.xavier_initializer()
	input_shape = x.get_shape().as_list()[1:]

	core,factors = None, None
	"""
	if(type(n_outputs) == type([])):
		core = tf.get_variable("core_higher_{}".format(np.prod(n_outputs)), shape=ranks[1:]+[FIXED_RANK]*len(n_outputs),
			initializer = weight_initializer)
		factors = [tf.get_variable("basic_factor_higher_map_{0}_{1}".format(i,e),
		shape=(input_shape[i],e),
		initializer = weight_initializer)
		for (i, e) in enumerate(ranks[1:])
		]
	else:
		"""

	core = tf.get_variable("core_last", shape=ranks, initializer = weight_initializer)
	factors = [	tf.get_variable("basic_factor_{0}_{1}".format(i,e),
				shape=(input_shape[i],ranks[i]),
				initializer = weight_initializer)
				for (i, e) in enumerate(input_shape)
				]

	bias = tf.get_variable("bias_{}".format(np.prod(n_outputs)), shape=(1, np.prod(n_outputs)))

	#bias = tf.get_variable("bias", shape=(1, n_outputs))
	
	"""
	if(type(n_outputs) == type([])):
		for i in range(len(n_outputs)):
			factors.append(tf.get_variable("higher_factor_{0}_{1}".format(i,np.prod(n_outputs)),
				shape=(n_outputs[i], FIXED_RANK),
				initializer = weight_initializer
				))
	else:
	"""
	# append the last N+1 factor matrix to the list
	factors.append(tf.get_variable("factor_{}".format(len(ranks)-1),
			shape=(n_outputs, ranks[-1]),
			initializer = weight_initializer))

	regression_weights = TNSR.tucker_to_tensor(core, factors)

	x_0 = tf.reshape(x, [-1, np.prod(input_shape)])

	w_minus1 = None

	"""
	if type(n_outputs) == type([]):
		w_minus1 = tf.reshape(regression_weights, [np.prod(input_shape), np.prod(n_outputs)])
		return tf.reshape(tf.add(tf.matmul(x_0, w_minus1) ,bias), [-1]+n_outputs)
	else:
	"""
	
	print(core.get_shape().as_list())
	for f in factors:
		print(f.get_shape().as_list())
	

	w_minus1 = tf.reshape(regression_weights, [np.prod(input_shape),n_outputs])
	return tf.add(tf.matmul(x_0, w_minus1) ,bias)
	#return regression(x, regression_weights, input_shape, bias, n_outputs)

def ttrl(x, ranks, n_outputs):
	weight_initializer = tf.contrib.layers.xavier_initializer()

	if(type(n_outputs) == type([])):
		#n_outputs 	[2,2,2,2,2,2]
		#ranks     	[1,1,2,2,3,4,3,2,2,1,1]
		#x 		 	[50,14,14,32]
		#input_shape[14,14,32,2,2,2,2,2,2]
		suffix = np.prod(n_outputs)
		input_shape = x.get_shape().as_list()[1:]+n_outputs
		bias = tf.get_variable("bias_{}".format(np.prod(n_outputs)), shape=(1, np.prod(n_outputs)))

		cores = []

		for i in range(len(input_shape)-1):
			cores.append(tf.get_variable("core_{0}_output_{1}".format(i,suffix), 
				shape = (ranks[i], input_shape[i], ranks[i+1]), 
				initializer = weight_initializer))

		cores.append(tf.get_variable("core_{0}_last_output_{1}".format(i,suffix), 
			shape = (ranks[-2], input_shape[-1], ranks[-1]), 
			initializer = weight_initializer))

		#for c in cores:
		#	print(c.get_shape().as_list())

		regression_weights = TNSR.tt_to_tensor(cores)
		w_minus1 = tf.reshape(regression_weights, [np.prod(input_shape[:3]), np.prod(n_outputs)])
		x_0 = tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])
		return tf.reshape(tf.add(tf.matmul(x_0, w_minus1) ,bias), [-1]+n_outputs)

	else:
		suffix = n_outputs
		input_shape = x.get_shape().as_list()[1:]
		#bias = tf.get_variable("bias", shape=(1, n_outputs))
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

	#return regression(x, regression_weights, input_shape, bias, n_outputs)

def cprl(x, rank, n_outputs):
	weight_initializer = tf.contrib.layers.xavier_initializer()
	input_shape = x.get_shape().as_list()[1:]
	
	bias = tf.get_variable("bias_{}".format(np.prod(n_outputs)), shape=(1, np.prod(n_outputs)))

	rank1_tnsrs = []

	print(rank)

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

	print(input_shape)
	print(x.get_shape().as_list())
	print(regression_weights.get_shape().as_list())
	x_0 = tf.reshape(x, [-1, np.prod(input_shape)])

	w_minus1 = None
	if type(n_outputs) == type([]):
		w_minus1 = tf.reshape(regression_weights, [np.prod(input_shape), np.prod(n_outputs)])
		return tf.reshape(tf.add(tf.matmul(x_0, w_minus1) ,bias), [-1]+n_outputs)
	else:
		w_minus1 = tf.reshape(regression_weights, [np.prod(input_shape),n_outputs])
		return tf.add(tf.matmul(x_0, w_minus1) ,bias)

def next_batch(num, data, labels):
	'''
	Return a total of `num` random samples and labels. 
	'''
	idx = np.arange(0 , len(data))
	np.random.shuffle(idx)
	idx = idx[:num]
	data_shuffle = [data[ i] for i in idx]
	labels_shuffle = [labels[ i] for i in idx]

	return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def one_hot(lst,num_classes):
	ret = []
	for elem in lst:
		v = [0]*num_classes
		v[elem-1] = 1
		ret.append(v)
	return ret






