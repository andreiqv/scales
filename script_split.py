#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division  
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import argparse
import math
import logging
from collections import namedtuple
import operator # for min
#import numpy as np
import random


def data_split_by_id(src_dir, dst_dir):

	files = os.listdir(src_dir)
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
	
		sub_dir = dst_dir + '/' + src(class_id)
		src_path = src_dir + '/' + filename
		dst_path = sub_dir + '/' + filename

		cmd = 'mkdir -p {0}'.format(sub_dir)
		os.system(cmd)
		cmd = 'cp {0} {1}'.format(src_path, dst_path)
		os.system(cmd)
 	



#---------------

def createParser ():
	"""	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--part', default="", type=str,\
		help='part')
	#parser.add_argument('-th', '--threshold', default=0.05, type=float,\
	#	help='threshold value (default 0.05)')
	#parser.add_argument('-df', '--diff', dest='diff', action='store_true')

	return parser


if __name__ == '__main__':	

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])
	part = arguments.part

	src_dir = '/home/chichivica/Data/Datasets/Scales/data/{0}'.format(part)
	dst_dir = '/home/chichivica/Data/Datasets/Scales/split'
	src_dir = src_dir.rstrip('/')
	dst_dir = dst_dir.rstrip('/')

	#src_dir = '/w/WORK/ineru/06_scales/data/train/'
	#dst_dir = '/w/WORK/ineru/06_scales/data/_diff/'
	#file_list = '/w/WORK/ineru/06_scales/git_scales/diff_train.txt'

	data_split_by_id(src_dir, dst_dir)