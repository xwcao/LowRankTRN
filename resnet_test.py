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

tf.app.flags.DEFINE_integer('input_size', 32, "input image size")
tf.app.flags.DEFINE_integer('num_classes', 10, "input image size")
tf.app.flags.DEFINE_string('data_dir', './data/cifar-10-batches-bin','where to store the dataset')
tf.app.flags.DEFINE_boolean('use_bn', True, 'use batch normalization. otherwise use biases')
tf.app.flags.DEFINE_integer('test_size', 10000, 'size of the test dataset')
tf.app.flags.DEFINE_integer('test_stride', 4000, 'the iteration to do test on')

tf.app.flags.DEFINE_string('train_dir', './resnet_train/exp46',
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

#42 C1
#43 C5
#44 C10
#45 C25
#46 C50

# CIFAR 10 Data
tr_data, tr_labels, te_data, te_labels, label_names = get_cifar10('./data/cifar-10-batches-py/')

tr_data = tr_data.reshape(50000, 3*32*32)
#tr_labels = one_hot(tr_labels,10)
te_data = te_data.reshape(10000, 3*32*32)
#te_labels = one_hot(te_labels,10)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config = config) as sess:
	flags = tf.app.flags.FLAGS

	#maybe_download_and_extract()

	images_train, labels_train = distorted_inputs(FLAGS.data_dir, FLAGS.batch_size)
	#images_val, labels_val = inputs(True, FLAGS.data_dir, FLAGS.test_size)

	is_training = tf.placeholder('bool', [], name='is_training')


	#imgs, lbls = tf.cond(is_training,
	#	lambda: (float_images, labels),
	#	lambda: (images, labels))

	#images, labels = tf.cond(is_training,
	#	lambda: (images_train, labels_train),
	#		lambda: (images_val, labels_val))

	#20603 cp 10
	#20605 t [16,10,10,80]
	#20606 tt [1,10,15,10,1]
	#FIXED THE BATCH SIZE TO BE 10000
	#20616 tt [1,10,15,10,1]
	#20617 t [16,10,10,80]
	#20618 cp 10
	# Three above didnt work within 24 hours
	#20754 cp 10
	# It's not finishing So changed a code
	#20828 cp 10
	# Just saving the models at each step

	# 8, 8 ,64, 10
	logits = inference_small(images_train, num_classes=10, is_training=is_training, use_bias=(not FLAGS.use_bn), num_blocks=3
							,trl_type = 'cp', rank = 50) #[1,8,64,10,1]

	#exp9: gap_tcl without fixing anything
	#exp10: gap_tcl with fixing the last contraction

	#exp11: 2,4,8

	#train(is_training, logits, input_x, lbls, sess, tr_data, tr_labels)
	train(is_training, logits, images_train, labels_train, sess, tr_data, tr_labels)

	#x = tf.placeholder(tf.float32, shape = [None, flags.input_size * flags.input_size * 3])
	#y = tf.placeholder(tf.float32, shape = [None, flags.num_classes])

