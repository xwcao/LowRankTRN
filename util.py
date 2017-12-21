#!/usr/bin/python

import numpy as np
import tensorflow as tf

class TNSR:

	# TF
	@staticmethod
	def unfold(tensor, mode):
		lst = range(0, len(tensor.get_shape().as_list()))
		return tf.reshape(tensor = tf.transpose(tensor, [mode] + lst[:mode] + lst[mode+1:]), shape = [tensor.get_shape().as_list()[mode],-1])


	@staticmethod
	def fold(tensor, mode, shape):
		full_shape = list(shape)
		mode_dim = full_shape.pop(mode)
		full_shape.insert(0, mode_dim)

		if None in full_shape:
			full_shape[full_shape.index(None)] = -1

		lst = range(1, len(full_shape))
		lst.insert(mode, 0)

		return tf.transpose(tf.reshape(tensor = tensor, shape = full_shape), lst)

	@staticmethod
	def mode_dot(tensor, matrix, mode):
		new_shape = tensor.get_shape().as_list()

		if matrix.get_shape().as_list()[1] != tensor.get_shape().as_list()[mode]:
			raise ValueError("Shape error. {0}(matrix's 2nd dimension) is not as same as {1} (dimension of the tensor)".format(matrix.get_shape().as_list()[1], tensor.get_shape().as_list()[mode]))

		new_shape[mode] = matrix.get_shape().as_list()[0]

		res = tf.matmul(matrix, TNSR.unfold(tensor, mode))

		return TNSR.fold(res, mode, new_shape)

	@staticmethod
	def tucker_to_tensor(core, factors):
		#tnsr = tf.identity(core)
		for i,factor in enumerate(factors):
			core = TNSR.mode_dot(core, factor, i)
		return core	

	@staticmethod
	def tt_to_tensor(cores):
		tensor_size = []
		for c in cores:
			tensor_size.append(c.get_shape().as_list()[1])

		md = 2
		new_shape = cores[0].get_shape().as_list()[:-1] + cores[1].get_shape().as_list()[1:]
		t = tf.reshape(TNSR.mode_dot(cores[0],tf.transpose(TNSR.unfold(cores[1],0)),md), new_shape)

		for i in range(1,len(tensor_size)-1):
			md = md + 1
			new_shape = t.get_shape().as_list()[:-1] + cores[i+1].get_shape().as_list()[1:]

			t = tf.reshape(TNSR.mode_dot(t, tf.transpose(TNSR.unfold(cores[i+1],0)),md), new_shape)

		return tf.reshape(t, tensor_size)

	@staticmethod
	def cp_to_tensor(rank1_tnsrs):
		tnsr = rank1_tnsrs[0][0]
		for i in range(1, len(rank1_tnsrs[0])):
			if(tf.__version__ == "0.11.0rc2"):
				tnsr = tf.mul(tf.reshape(tnsr,[-1,1]),tf.reshape(rank1_tnsrs[0][i], [1,-1]))
			else:
				tnsr = tf.multiply(tf.reshape(tnsr,[-1,1]),tf.reshape(rank1_tnsrs[0][i], [1,-1]))

		for j in range(1,len(rank1_tnsrs)):
			t = rank1_tnsrs[j][0]
			for k in range(1,len(rank1_tnsrs[j])):
				if(tf.__version__ == "0.11.0rc2"):
					t = tf.mul(tf.reshape(t,[-1,1]), tf.reshape(rank1_tnsrs[j][k],[1,-1]))
				else:
					t = tf.multiply(tf.reshape(t,[-1,1]), tf.reshape(rank1_tnsrs[j][k],[1,-1]))

			tnsr = tf.add(tnsr,t)

		return tnsr

	# NUMPY
	@staticmethod
	def np_unfold(tensor, mode):
		return np.moveaxis(tensor, mode, 0).reshape((tensor.shape[mode], -1))


	@staticmethod
	def np_fold(unfolded_tensor, mode, shape):
		full_shape = list(shape)
		mode_dim = full_shape.pop(mode)
		full_shape.insert(0, mode_dim)
		return np.moveaxis(unfolded_tensor.reshape(full_shape), 0, mode)

	@staticmethod
	def np_mode_dot(tensor, matrix, mode):
		new_shape = list(tensor.shape)

		if matrix.ndim == 2:  # Tensor times matrix
			# Test for the validity of the operation
			if matrix.shape[1] != tensor.shape[mode]:
				raise ValueError(
					'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
						tensor.shape, matrix.shape, mode, tensor.shape[mode], matrix.shape[1]
					))
			new_shape[mode] = matrix.shape[0]

		res = np.dot(matrix, TNSR.np_unfold(tensor, mode))

		return TNSR.np_fold(res, mode, new_shape)

		




