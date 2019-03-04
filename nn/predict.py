import tensorflow as tf, sys
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import sklearn
'''
'''
def get_files(file_dir):
	m0 = []
	m1 = []
	m2 = []
	l0 = []
	l1 = []
	l2 = []
	i = 0
	for folder in os.listdir(file_dir):
		for file in os.listdir(file_dir + '/' + folder):
			if i == 0:
				m0.append(file_dir + '/' + folder + '/' + file)
				l0.append(0)
			elif i == 1:
				m1.append(file_dir + '/' + folder + '/' + file)
				l1.append(1)
			else:
				m2.append(file_dir + '/' + folder + '/' + file)
				l2.append(2)
		i+=1
	image_list = np.hstack((m0,m1))
	image_list = np.hstack((m2,image_list))
	label_list = np.hstack((l0,l1))
	label_list = np.hstack((l2,label_list))
	temp = np.array([image_list, label_list])
	temp = temp.transpose()
	np.random.shuffle(temp)
	image_list = list(temp[:,0])
	label_list = list(temp[:,1])
	label_list = [int(i) for i in label_list]

	return image_list, label_list

from PIL import Image

def get_one_image(train):
	n = len(train)
	ind = np.random.randint(0, n)
	img_dir = train[ind]

	image = Image.open(img_dir)
	plt.imshow(image)
	plt.show()
	image = image.resize([208, 208])
	image = np.array(image)
	return image

def inference(images, batch_size, n_classes) :
	'''Build


	'''
	with tf.variable_scope('conv1') as scope:
		weights = tf.get_variable('weights',
									shape = [3,3,3, 16],
									dtype = tf.float32,
									initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))
		biases = tf.get_variable('biases',
									shape = [16],
									dtype = tf.float32,
									initializer = tf.constant_initializer(0.1) )
									
		conv = tf.nn.conv2d(images,weights,strides =[1,1,1,1], padding = "SAME")
		pre_activation = tf.nn.bias_add(conv,biases)
		conv1 = tf.nn.relu(pre_activation, name = scope.name)

	with tf.variable_scope('pooling1_lrn') as scope:
		pool1 = tf.nn.max_pool(conv1, ksize =[1,3,3,1], strides =[1,2,2,1],
											padding = "SAME", name = 'pooling1')
		norm1 = tf.nn.lrn(pool1, depth_radius = 4, bias = 1.0, alpha = 0.001/9.0,
				beta = 0.75, name = 'norm1')

	with tf.variable_scope('conv2') as scope:
		weights = tf.get_variable('weights',
									shape = [3,3,16, 16],
									dtype = tf.float32,
									initializer = tf.truncated_normal_initializer(stddev = 0.1, dtype = tf.float32))
		biases = tf.get_variable('biases',
									shape = [16],
									dtype = tf.float32,
									initializer = tf.constant_initializer(0.1) )
									
		conv = tf.nn.conv2d(norm1,weights,strides =[1,1,1,1], padding = "SAME")
		pre_activation = tf.nn.bias_add(conv,biases)
		conv2 = tf.nn.relu(pre_activation, name = 'conv2')

	with tf.variable_scope('pooling2_lrn') as scope:
		norm2 = tf.nn.lrn(conv2, depth_radius = 4, bias = 1.0, alpha = 0.001/9.0,
				beta = 0.75, name = 'norm2')
		pool2 = tf.nn.max_pool(norm2, ksize =[1,3,3,1], strides =[1,1,1,1],
											padding = "SAME", name = 'pooling2')

	with tf.variable_scope('local3') as scope:
		reshape = tf.reshape(pool2, shape =[batch_size, -1])
		dim = reshape.get_shape()[1].value
		weights = tf.get_variable('weights',
									shape =[dim,128],
									dtype = tf.float32,
									initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))
		biases = tf.get_variable('biases',
									shape =[128],
									dtype = tf.float32,
									initializer = tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name = scope.name)

	with tf.variable_scope('local4') as scope:
		weights = tf.get_variable('weights',
									shape =[128,128],
									dtype = tf.float32,
									initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))
		biases = tf.get_variable('biases',
									shape =[128],
									dtype = tf.float32,
									initializer = tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name = 'local4')	

	with tf.variable_scope('softmax_linear') as scope:
		weights = tf.get_variable("softmax_linear",
			shape = [128, n_classes],
			dtype = tf.float32,
			initializer = tf.truncated_normal_initializer(stddev = 0.005, dtype = tf.float32))
		biases = tf.get_variable('biases',
									shape =[n_classes],
									dtype = tf.float32,
									initializer = tf.constant_initializer(0.1))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name = 'softmax_linear')

	return softmax_linear

def evaluated_one_image():
		test_dir = 'testing'
		test, test_label = get_files(test_dir)
		image_array = get_one_image(test)
		with tf.Graph().as_default():
			BATCH_SIZE = 1
			N_CLASSES = 3

			image = tf.cast(image_array,tf.float32)
			image = tf.reshape(image, [1, 208, 208, 3])
			logit = inference(image, BATCH_SIZE, N_CLASSES)
			logit = tf.nn.softmax(logit)

			x = tf.placeholder(tf.float32, shape =[208, 208, 3])
			logs_test_dir = 'bogs'
			saver= tf.train.Saver()

			with tf.Session() as sess:

				ckpt = tf.train.get_checkpoint_state(logs_test_dir)
				if ckpt and ckpt.model_checkpoint_path:
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					saver.restore(sess, ckpt.model_checkpoint_path)
					print('Loading success, global_step is %s' % global_step)
				else:
					print('No checkpoint file found')

				prediction = sess.run(logit, feed_dict = {x:image_array})
				max_index = np.argmax(prediction)

				if max_index == 0:
					print("this is a black with a probability of %.6f" %prediction[:,0])
				elif max_index == 1: 
					print("this is a white with a probability of %.6f" %prediction[:,1])
				elif max_index == 2: 
					print("this is a red with a probability of %.6f" %prediction[:,2])	

evaluated_one_image()