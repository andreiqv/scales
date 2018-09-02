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

def copy_files_by_list(src_dir, dst_dir, file_list):

	with open(file_list) as f:
		for line in f:
			filename = line.split()[3]
			#print(filename)
			path = src_dir + '/' + filename
			cmd = "cp {0} {1}".format(path, dst_dir)
			print(cmd)




#---------------

def createParser ():
	"""	ArgumentParser
	"""
	parser = argparse.ArgumentParser()
	#parser.add_argument('-th', '--threshold', default=0.05, type=float,\
	#	help='threshold value (default 0.05)')
	#parser.add_argument('-df', '--diff', dest='diff', action='store_true')

	return parser


if __name__ == '__main__':	

	parser = createParser()
	arguments = parser.parse_args(sys.argv[1:])	

	src_dir = '/home/chichivica/Data/Datasets/Scales/data_all/train'
	dst_dir = '/home/chichivica/Data/Datasets/Scales/diff'
	file_list = '/home/chichivica/Data/Datasets/Scales/diff_train.txt'

	#src_dir = '/w/WORK/ineru/06_scales/data/train/'
	#dst_dir = '/w/WORK/ineru/06_scales/data/_diff/'
	#file_list = '/w/WORK/ineru/06_scales/git_scales/diff_train.txt'


	src_dir = src_dir.rstrip('/')
	dst_dir = dst_dir.rstrip('/')

	copy_files_by_list(src_dir, dst_dir, file_list)