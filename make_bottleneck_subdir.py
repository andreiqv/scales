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

import split_data

DO_MIX = True

if os.path.exists('.notebook'):
	#data_dir = '../data'
	data_dir = '../separated'
	module = network.conv_network_224
	SHAPE = 224, 224, 3
else:
	#data_dir = '/home/chichivica/Data/Datasets/Scales/data'
	data_dir = '/home/chichivica/Data/Datasets/Scales/separated'
	import tensorflow_hub as hub

	model_number = 3
	type_model = 'feature_vector'
	#type_model = 'classification'
	
	if model_number == 1:		
		module = hub.Module("https://tfhub.dev/google/imagenet/resnet_v2_152/{0}/1".format(type_model))
		SHAPE = 224, 224, 3
	elif model_number == 2:
		module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/{0}/1".format(type_model))
		SHAPE = 299, 299, 3
	elif model_number == 3:
		module = hub.Module("https://tfhub.dev/google/imagenet/inception_resnet_v2/{0}/1".format(type_model))
		SHAPE = 299, 299, 3
	else:
		raise Exception('Bad n_model')
		# https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1

np.set_printoptions(precision=4, suppress=True)


#---------------------------------


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





#---------------------------------

def data_analysis(dir_path):

	class_id_set = set()
	#num_classes = 412
	files = os.listdir(dir_path)
	random.shuffle(files)
	num_files = len(files)	

	for index_file, filename in enumerate(files):

		#print(filename)
		base = os.path.splitext(filename)[0]
		ext = os.path.splitext(filename)[1]
		if not ext in {'.jpg', ".png"} : continue			

		try:
			class_id = int(base.split('_')[-1])
		except:
			continue
	
		class_id_set.add(class_id)

	return class_id_set 	



def create_bootleneck_data(dir_path, shape, maps):
	""" Calculate feature vectors for input images using TF.
	Returns:
	data : dict {'images':  [list of feature_vectors], 
				 'labels':  [list of labels], 
				 'filenames': [list of file names]}
		where 
	"""
	image_size = (shape[0], shape[1])
	feature_vectors, labels, filenames = [], [], []
	#class_index_set = set()
	map_id_label = maps['id_label']
	num_classes = len(map_id_label)

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

			base = os.path.splitext(filename)[0]
			ext = os.path.splitext(filename)[1]
			if not ext in {'.jpg', ".png"} : continue			

			try:
				class_id = int(base.split('_')[-1])
			except:
				continue

			class_index = map_id_label[class_id]				

			if class_index >= num_classes:
				logging.error('class_index >= num_classes')	
				sys.exit(1)

			label = [0]*num_classes
			label[class_index] = 1	

			#class_index_set.add(class_index)	

			file_path = dir_path + '/' + filename
			im = Image.open(file_path)		
			#sx, sy = im.size
			im = im.resize(image_size, Image.ANTIALIAS)
			arr = np.array(im, dtype=np.float32) / 256
			feature_vector = bottleneck_tensor.eval(feed_dict={ x : [arr] })			

			feature_vectors.append(feature_vector)
			labels.append(label)
			filenames.append(filename) # or file_path

			im.close()

			print("{0}/{1} : {2:3d} -> {3:3d} : {4}".format(index_file, num_files, class_id, class_index, filename))


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

	return data



def make_bottleneck_dump(src_dir, shape):

	bottleneck_data = dict()
	parts = ['valid', 'test', 'train']

	class_id_set = set()
	for part in parts:
		part_dir = in_dir + '/' + part
		class_id_set |= data_analysis(part_dir)
	
	id_list = list(class_id_set)
	id_list.sort()
	print('Number of classes in the sample: {0}'.format(len(id_list)))
	print('Min class id: {0}'.format(min(id_list)))
	print('Max class id: {0}'.format(max(id_list)))
	map_id_label = {class_id : index for index, class_id in enumerate(id_list)}
	map_label_id = {index : class_id for index, class_id in enumerate(id_list)}
	maps = {'id_label' : map_id_label, 'label_id' : map_label_id}
	bottleneck_data['id_label'] = map_id_label
	bottleneck_data['label_id'] = map_label_id
	print(map_id_label)
	#sys.exit(0)

	for part in parts:
		print('\nProcessing {0} data'.format(part))
		part_dir = in_dir + '/' + part
		bottleneck_data[part] = create_bootleneck_data(part_dir, shape, maps)

	assert len(bottleneck_data['train']['labels']) == len(bottleneck_data['train']['images'])
	print('Size of train data:', len(bottleneck_data['train']['labels']))
	print('Size of valid data:', len(bottleneck_data['valid']['labels']))
	print('Size of test data:',  len(bottleneck_data['test']['labels']))

	return bottleneck_data


#------------------------------------------------

def make_bottleneck_dump_subdir(src_dir, shape, ratio):
	"""
	Use names of subdirs as a id.
	And then calculate class_index from id.
	"""

	class_id_set = set()
	#bottleneck_data = dict()
	feature_vectors, labels, filenames = [], [], []

	image_size = (shape[0], shape[1])
	listdir = os.listdir(src_dir)
	
	# 1) findout number of classes
	for class_id in listdir:
		
		subdir = src_dir + '/' + class_id
		if not os.path.isdir(subdir): continue

		if len(os.listdir(subdir)) == 0: 
			continue
		else: 
			try:
				class_id_int = int(class_id)
				class_id_set.add(class_id_int)
			except:
				continue	
			
	# 2) maps class_id to class_index			
	id_list = list(class_id_set)
	id_list.sort()
	print('Number of classes in the sample: {0}'.format(len(id_list)))
	print('Min class id: {0}'.format(min(id_list)))
	print('Max class id: {0}'.format(max(id_list)))
	map_id_label = {class_id : index for index, class_id in enumerate(id_list)}
	map_label_id = {index : class_id for index, class_id in enumerate(id_list)}
	maps = {'id_label' : map_id_label, 'label_id' : map_label_id}
	num_classes = len(map_id_label)		

	# 3) Calculate bottleneck in TF
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
		
		for class_id in class_id_set:

			subdir = src_dir + '/' + str(class_id)
			print(subdir)
			files = os.listdir(subdir)
			num_files = len(files)
		
			for index_file, filename in enumerate(files):

				base = os.path.splitext(filename)[0]
				ext = os.path.splitext(filename)[1]
				if not ext in {'.jpg', ".png"} : continue			

				class_index = map_id_label[class_id]
				#print(class_index)

				label = [0]*num_classes
				label[class_index] = 1	

				#class_index_set.add(class_index)	

				file_path = subdir + '/' + filename
				im = Image.open(file_path)		
				im = im.resize(image_size, Image.ANTIALIAS)
				arr = np.array(im, dtype=np.float32) / 256
				feature_vector = bottleneck_tensor.eval(feed_dict={ x : [arr] })			

				feature_vectors.append(feature_vector)
				labels.append(label)
				filenames.append(filename) # or file_path

				im.close()

				print("dir={0}, class={1}: {2}/{3}: {4}".format(class_id, class_index, index_file, num_files, filename))
	
	print('----')
	print('Number of classes: {0}'.format(num_classes))	
	print('Number of feature vectors: {0}'.format(len(feature_vectors)))	

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

	print('Split data')
	data = split_data.split_data(data, ratio=ratio)
	data['id_label'] = map_id_label
	data['label_id'] = map_label_id

	return data


#------------------------------------------------

def save_data_dump(data, dst_file):
	
	# save the data on a disk
	dump = pickle.dumps(bottleneck_data)
	print('dump done')
	f = gzip.open(dst_file, 'wb')
	print('gzip done')
	f.write(dump)
	print('dump was written')
	f.close()



def createParser ():
	"""
	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--src_dir', default=None, type=str,\
		help='input dir')
	parser.add_argument('-o', '--dst_file', default=None, type=str,\
		help='output file')
	parser.add_argument('-m', '--mix', dest='mix', action='store_true')

	parser.add_argument('-n', '--num', default=100, type=int,\
		help='num_angles for a single picture')
	return parser


if __name__ == '__main__':

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])			
	NUM_ANGLES 	 = arguments.num
	#DO_MIX 		 = arguments.mix
	print('NUM_ANGLES =', 	NUM_ANGLES)
	#print('DO_MIX =',		DO_MIX)

	if not arguments.src_dir:
		src_dir = data_dir
	if not arguments.dst_file:		
		dst_file = 'dump.gz'

	#bottleneck_data = make_bottleneck_dump(in_dir=in_dir, shape=shape)

	bottleneck_data = make_bottleneck_dump_subdir(
		src_dir=src_dir, shape=SHAPE, ratio=[9,1,1])	

	save_data_dump(bottleneck_data, dst_file=dst_file)
