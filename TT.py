#!/usr/bin/python

import numpy as np
from util import TNSR


class TensorTrain():

	def __init__(self, ranks, tensor_size, initializer = np.random.normal()):

		self.ranks = ranks
		self.cores = []
		self.tensor_size = tensor_size

		self.initializer = initializer

		for i in range(1, len(ranks)):
			self.cores.append(np.arange(np.prod((ranks[i-1],tensor_size[i-1],ranks[i]))).reshape((ranks[i-1],tensor_size[i-1],ranks[i])))
			#self.cores.append(np.random.normal(loc = 0.0, scale = 0.5, size = (ranks[i-1],tensor_size[i-1],ranks[i])))

	def sliceAt(self, core_at, slice_at):
		return self.cores[core_at][:,slice_at,:]

	def entryAt(self, index):
		e = self.sliceAt(0 , index[0])
		for i in range(1,len(index)):
			e = np.matmul(e, self.sliceAt(i,index[i]))
		return e

	def restore(self):
		t = []

		for index, value in np.ndenumerate(np.zeros(self.tensor_size)):
			t.append(self.entryAt(index))

		return np.asarray(t).reshape(self.tensor_size)

	def restore_2(self):
		md = 2
		t = TNSR.np_mode_dot(self.cores[0],np.transpose(TNSR.np_unfold(self.cores[1],0)),md).reshape(list(self.cores[0].shape)[:-1]+list(self.cores[1].shape)[1:])
		for i in range(1,len(self.tensor_size)-1):
			print(t.shape)
			print(i)
			md = md + 1
			t = TNSR.np_mode_dot(t,np.transpose(TNSR.np_unfold(self.cores[i+1],0)),md).reshape(list(t.shape)[:-1]+list(self.cores[i+1].shape)[1:])

		return t.reshape(self.tensor_size)


		

if __name__ == '__main__':
	print("Testing TT.")

	a = TensorTrain(ranks = [1,2,3,2,1], tensor_size = [2,3,4,5])

	print(a.restore()-a.restore_2())




	



