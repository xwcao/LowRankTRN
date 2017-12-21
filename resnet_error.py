import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from operator import add, sub


MAXITER = 500000
STRIDE = 100
ITER = MAXITER/STRIDE

PLOT_LABELS = ["CP(10)", "Tucker(16,10,10,80)", "TT(1,10,15,10,1)"]
FILENAMES = ["resnet_test.py.o20603","resnet_test.py.o20605","resnet_test.py.o20606"]

_steps = []
_losses = []
_val_top1 = []
_steps_val_top1 = []

for i in range(len(FILENAMES)):
	with open('./out/resnet_cifar/'+FILENAMES[i]) as f:

		steps = []
		losses = []
		val_top1 = []
		steps_val_top1 = []
		LINES = 21

		for j in range(ITER):
			for k in range(LINES):
				# step 85, loss = 2.09 (346.0 examples/sec; 0.046 sec/batch)
				l = f.readline().strip().split(" ")
				if(int(l[1][:-1]) % 5000 == 0):
					steps.append(int(l[1][:-1]))
					losses.append(float(l[4]))

			if(LINES == 21):
				LINES = 20

			# Validation top1 error 0.94
			l = f.readline().strip().split(" ")
			if(STRIDE*(j+1) % 5000 == 0):
				val_top1.append(float(l[-1]))
				steps_val_top1.append(STRIDE*(j+1))

		_steps.append(steps)
		_losses.append(losses)
		_val_top1.append(val_top1)
		_steps_val_top1.append(steps_val_top1)

fig, ax = plt.subplots()

for i in range(len(_steps)):
	s = _steps[i]
	ax.plot(s, _losses[i], linestyle='solid', label = PLOT_LABELS[i])

plt.title("Loss of Tensor Regression Layers with ResNet on CIFAR-10")
ax.legend(loc='upper right')
ax.set_xlabel('Steps')
ax.set_ylabel('Cross Entropy Loss')
plt.savefig("./graphs/ResNetCifarLoss.png")

fig, ax = plt.subplots()
for i in range(len(_steps_val_top1)):
	s = _steps_val_top1[i]
	ax.plot(s, _val_top1[i], linestyle='solid', label = "Validation Error of "+PLOT_LABELS[i])

plt.title("Validation Error of Tensor Regression Layers with ResNet on CIFAR-10")
ax.legend(loc='upper right')
ax.set_xlabel('Steps')
ax.set_ylabel('Validation Error')
plt.savefig("./graphs/ResNetCifarValError.png")