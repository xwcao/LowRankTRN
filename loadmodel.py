#!/usr/bin/python

import sys
import tensorflow as tf
import numpy as np

sys.path.append('/home/xcao')
sys.path.append('./tensorflow-resnet-master/')

from util import TNSR
from TRL import *
from input_cifar import *
from train_cifar import *
from resnet_train import *

tf.app.flags.DEFINE_integer('input_size', 32, "input image size")
tf.app.flags.DEFINE_integer('num_classes', 10, "input image size")
tf.app.flags.DEFINE_string('data_dir', './data/cifar-10-batches-bin','where to store the dataset')
tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
tf.app.flags.DEFINE_integer('test_size', 10000, 'size of the test dataset')
tf.app.flags.DEFINE_integer('test_stride', 2000, 'the iteration to do test on')

tf.app.flags.DEFINE_string('train_dir', './resnet_train/exp42',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 128, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 64000, "max steps")
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')

from resnet import *

# CIFAR 10 Data
#tr_data, tr_labels, te_data, te_labels, label_names = get_cifar10('./data/cifar-10-batches-py/')

#tr_data = tr_data.reshape(50000, 3*32*32)
#tr_labels = one_hot(tr_labels,10)
#te_data = te_data.reshape(10000, 3*32*32)
#te_labels = one_hot(te_labels,10)
result = []

expnum = "exp46"

filepath = "./resnet_train/{}/".format(expnum)

#rng = range(2000,62001,6000)
#rng = [52000,56000,60000,64000]

#42 C1
#43 C5
#44 C10
#45 C25
#46 C50

rng = range(52000,64001,4000)

for elem in rng:
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	
	lst = []
	
	with tf.Session(config = config) as sess:
		flags = tf.app.flags.FLAGS
	
		#input_x = tf.placeholder(tf.float32, shape = [None, 32* 32* 3])
		#labels = tf.placeholder(tf.int32, [None])
		#images = tf.reshape(input_x, [-1, 32, 32, 3], name='images')
	
		saver = tf.train.import_meta_graph(filepath+'model.ckpt-{}.meta'.format(elem+1))
		saver.restore(sess, filepath+'model.ckpt-{}'.format(elem+1))
	
	
		tvars = tf.trainable_variables()
		tvars_vals = sess.run(tvars)
	
		for var, val in zip(tvars, tvars_vals):
			lst.append(val)
			print(var)
	
	#print(vars)
	
	# key u'scale1/block1/B/weights:0'
	# val nparray
	
	tf.reset_default_graph()
	
	with tf.Session() as sess:
		flags = tf.app.flags.FLAGS
		images_val, labels_val = inputs(True, FLAGS.data_dir, FLAGS.test_size)
		is_training = tf.placeholder('bool', [], name='is_training')
	
		logits = inference_small(images_val, num_classes=10, is_training=is_training, use_bias=(not FLAGS.use_bn), num_blocks=3
								,trl_type = 'cp', rank = 50)
	
	
	
		global_step = tf.get_variable('global_step', [],
			initializer=tf.constant_initializer(0),
			trainable=False)
		val_step = tf.get_variable('val_step', [],
			initializer=tf.constant_initializer(0),
			trainable=False)
	
		loss_ = loss(logits, labels_val)
		predictions = tf.nn.softmax(logits)
	
		top1_error, top5_error = top_1_and_5(predictions, labels_val)
	
		# loss_avg
		ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
		tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
		#tf.scalar_summary('loss_avg', ema.average(loss_))
	
		# validation stats
		ema = tf.train.ExponentialMovingAverage(0.9, val_step)
		val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
		top1_error_avg = ema.average(top1_error)
		#tf.scalar_summary('val_top1_error_avg', top1_error_avg)
	
		#tf.scalar_summary('learning_rate', FLAGS.learning_rate)
	
		opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
		grads = opt.compute_gradients(loss_)
		#for grad, var in grads:
		    #if grad is not None and not FLAGS.minimal_summaries:
		        #tf.histogram_summary(var.op.name + '/gradients', grad)
		apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	
		#if not FLAGS.minimal_summaries:
		    # Display the training images in the visualizer.
		    #tf.image_summary('images', images)
	
		    #for var in tf.trainable_variables():
		        #tf.histogram_summary(var.op.name, var)
	
		batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
		batchnorm_updates_op = tf.group(*batchnorm_updates)
		train_op = tf.group(apply_gradient_op, batchnorm_updates_op, name = 'train_op')
	
		#saver = tf.train.Saver(tf.all_variables(), max_to_keep=(FLAGS.max_steps/FLAGS.test_stride)+1)
	
		#testSteps = range(0,FLAGS.max_steps+1,FLAGS.test_stride)[1:]
	
		#summary_op = tf.merge_all_summaries()
		sess.run(tf.initialize_all_variables())
		#sess.graph.finalize()
		tf.train.start_queue_runners(sess=sess)
	
		tvars = tf.trainable_variables()
		#print(len(lst))
		#print(len(tvars))
	
		ops = []
		i = 0
		for v in tvars:
			assign_op = v.assign(lst[i])
			ops.append(assign_op)
			i += 1
	
	
		for operation in ops:
			sess.run(operation)
	
		_, t1, t5 = sess.run([val_op, top1_error, top5_error], { is_training: True })
	
		result.append([elem,t1,t5])
	
	tf.reset_default_graph()

with open('./resnet_train/test_error/{}.txt'.format(expnum), 'w+') as f:
	for e in result:
		f.write("At:{0} Top 1:{1} Top 5:{2}\n".format(e[0],e[1],e[2]))

"""
graph = tf.get_default_graph()	
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict ={w1:13.0,w2:17.0}

#Now, access the op that you want to run. 
op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

print sess.run(op_to_restore,feed_dict)
"""

#g = [n.name for n in tf.get_default_graph().as_graph_def().node]
#print(g)

"""
graph = tf.get_default_graph()
input_x = graph.get_tensor_by_name("Placeholder:0")
labels = graph.get_tensor_by_name("Placeholder_1:0")
images = graph.get_tensor_by_name("images:0")
is_training = graph.get_tensor_by_name('is_training:0')

op_top1 = graph.get_tensor_by_name("ToFloat:0")
op_top5 = graph.get_tensor_by_name("ToFloat_1:0")

t1, t5 = sess.run([op_top1,op_top5], feed_dict = {input_x:te_data, labels: te_labels, is_training:True})

print(sum(t1))
print(len(t1))
print(sum(t5))
print(len(t5))
"""

