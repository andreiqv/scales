#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path
import sys
import argparse
from PIL import Image, ImageDraw
import _pickle as pickle
import gzip
import random
from random import randint
import math
import numpy as np

import tensorflow as tf
import network

DO_MIX = False

if os.path.exists('.notebook'):
	data_dir = 'data'
	module = network.conv_network_224
else:
	data_dir = '/home/chichivica/Data/Datasets/Scales/data'
	import tensorflow_hub as hub
	module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1")		

np.set_printoptions(precision=4, suppress=True)


#import tensorflow_hub as hub

"""
def load_data(in_dir, image_size):	
	# each image has form [height, width, 3]
	

	data = dict()
	data['images'], data['labels'], data['filenames'] = [], [], []

	files = os.listdir(in_dir)
	random.shuffle(files)

	for file_name in files:

		file_path = in_dir + '/' + file_name

		#img_gray = Image.open(file_path).convert('L')
		#img = img_gray.resize(img_size, Image.ANTIALIAS)
		img = Image.open(file_path)
		img = img.resize(image_size, Image.ANTIALIAS)
		arr = np.array(img, dtype=np.float32) / 256

		name = ''.join(file_name.split('.')[:-1])
		angle = name.split('_')[-1]
		lable = np.array([float(angle) / 360.0], dtype=np.float64)

		if type(lable[0]) != np.float64:
			print(lable[0])
			print(type(lable[0]))
			print('type(lable)!=float')
			raise Exception('lable type is not float')
			
		print('{0}: [{1:.3f}, {2}]' .format(angle, lable[0], file_name))
		data['images'].append(arr)
		data['labels'].append(lable)
		data['filenames'].append(file_name)

	# mix data
	print('mix data')
	zip3 = list(zip(data['images'], data['labels'], data['filenames']))
	random.shuffle(zip3)
	data['images']    = [x[0] for x in zip3]
	data['labels']    = [x[1] for x in zip3]
	data['filenames'] = [x[2] for x in zip3]

	for i in range(data['labels']):
		print('{0}' - '{1}'.format(data['labels'], data['filenames']))

	return data
	#return train, valid, test
"""

"""
def covert_data_to_feature_vector(data):

	module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
	height, width = hub.get_expected_image_size(module)
	assert (height, width) == (299, 299)
	image_feature_vector = module(data['images'])
	data['images'] = image_feature_vector
	return data
"""

#--------------------


def create_bootleneck_data(dir_path, shape):
	""" Calculate feature vectors for input images using TF.
	Returns:
	data : dict {'images':  [list of feature_vectors], 
				 'labels':  [list of labels], 
				 'filenames': [list of file names]}
		where 
	"""
	image_size = (shape[0], shape[1])
	feature_vectors, labels, filenames = [], [], []
	class_index_set = set()
	num_classes = 412

	files = os.listdir(dir_path)
	random.shuffle(files)
	num_files = len(files)

	# Calculate in TF
	height, width, color =  shape
	x = tf.placeholder(tf.float32, [None, height, width, 3], name='Placeholder-x')
	resized_input_tensor = tf.reshape(x, [-1, height, width, 3])
	#module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/classification/1")		
	
	# num_features = 2048, height x width = 224 x 224 pixels
	assert height, width == hub.get_expected_image_size(module)	
	bottleneck_tensor = module(resized_input_tensor)  # Features with shape [batch_size, num_features]
	print('bottleneck_tensor:', bottleneck_tensor)	

	with tf.Session() as sess:  # Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)	# Randomly initialize weights.
		
		for index_file, filename in enumerate(files):

			#print(filename)
			base = os.path.splitext(filename)[0]
			ext = os.path.splitext(filename)[1]
			if not ext in {'.jpg', ".png"} : continue			

			try:
				class_index = int(base.split('_')[-1])
			except:
				continue

			if class_index >= num_classes:
				logging.error('class_index >= num_classes')	
				sys.exit(1)

			label = [0]*num_classes
			label[class_index] = 1	

			class_index_set.add(class_index)	

			file_path = dir_path + '/' + filename
			im = Image.open(file_path)		
			#sx, sy = im.size
			im = im.resize(image_size, Image.ANTIALIAS)
			arr = np.array(im, dtype=np.float32) / 256

				
			#feature_vector = bottleneck_tensor.eval(feed_dict={ x : [arr] })
			feature_vector = None

			feature_vectors.append(feature_vector)
			labels.append(label)
			filenames.append(filename) # or file_path

			im.close()

	print('Number of feature_vectors: {0}'.format(len(feature_vectors)))
	
	data = {'images': feature_vectors, 'labels': labels, 'filenames':filenames}

	# mix data	
	if DO_MIX:
		print('start mix data')
		zip3 = list(zip(data['images'], data['labels'], data['filenames']))
		random.shuffle(zip3)
		print('mix ok')
		data['images']    = [x[0] for x in zip3]
		data['labels']    = [x[1] for x in zip3]
		data['filenames'] = [x[2] for x in zip3]
		print('data ready')

	#for i in range(len(data['labels'])):
	#	print('{0} - {1}'.format(data['labels'][i], data['filenames'][i]))

	class_list = list(class_index_set)
	class_list.sort()
	print('Number of classes: {0}'.format(len(class_list)))
	print('Min class index: {0}'.format(min(class_list)))
	print('Max class index: {0}'.format(max(class_list)))

	return data


def make_bottleneck_dump(in_dir, shape):

	bottleneck_data = dict()
	parts = ['valid', 'test', 'train']
	
	for part in parts:
		print('\nProcessing {0} data'.format(part))
		part_dir = in_dir + '/' + part
		bottleneck_data[part] = create_bootleneck_data(part_dir, shape)

	assert len(bottleneck_data['train']['labels']) == len(bottleneck_data['train']['images'])
	print('Size of train data:', len(bottleneck_data['train']['labels']))
	print('Size of valid data:', len(bottleneck_data['valid']['labels']))
	print('Size of test data:',  len(bottleneck_data['test']['labels']))

	return bottleneck_data


def save_data_dump(data, out_file):
	
	# save the data on a disk
	dump = pickle.dumps(bottleneck_data)
	print('dump done')
	f = gzip.open(out_file, 'wb')
	print('gzip done')
	f.write(dump)
	print('dump was written')
	f.close()



def createParser ():
	"""
	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--in_dir', default=None, type=str,\
		help='input dir')
	parser.add_argument('-o', '--out_file', default=None, type=str,\
		help='output file')
	parser.add_argument('-m', '--mix', dest='mix', action='store_true')

	parser.add_argument('-n', '--num', default=100, type=int,\
		help='num_angles for a single picture')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	NUM_ANGLES 	 = arguments.num
	DO_MIX 		 = arguments.mix
	print('NUM_ANGLES =', 	NUM_ANGLES)
	print('DO_MIX =',		DO_MIX)

	if not arguments.in_dir:
		in_dir = data_dir
	if not arguments.out_file:		
		out_file = 'dump.gz'

	shape = 224, 224, 3	
	bottleneck_data = make_bottleneck_dump(in_dir=in_dir, shape=shape)
	save_data_dump(bottleneck_data, out_file=out_file)
