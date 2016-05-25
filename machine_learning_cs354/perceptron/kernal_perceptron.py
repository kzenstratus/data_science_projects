# -*- coding: utf-8 -*-
"""
Created on Tue May 24 01:26:10 2016

@author: kevinzen

"""

import math
import numpy as np
from numpy import linalg as LA
from utils import *
from random import random

def rbf_kernel(x_i, x_j, gamma) :
	return np.exp( -gamma * (LA.norm(x_i - x_j) ** 2))

def train(train_data,label, gamma, bias):
	instances = train_data.shape[0]
	c         = np.zeros(instances)

	# iterate through all instances
	for t in range(instances) :
		temp = 0
		local_error = 0

		# calcualte local error and update c from 0 to curr instance
		for i in range(0, t) :
			x = train_data[i]
			# create output
			temp += c[i] * rbf_kernel(x[i],x[t], gamma) + bias
			# convert output to true or false
			predict = 1 if temp >= 0 else -1

			c[t] = 0

			# Update c according to if output was right or wrong
			if predict == label[i]:
				continue
			else:
				local_error += 1
				a[t] += 1
				if (predict == -1) and (label[i] == 1):
	            	c[t] = 1
	        	if (predict == 1) and (label[i] == -1):
	            	c[t] = -1

        total_error += local_error
        print total_error
    return c

def grid_search(train_data, label, k, gammas, biases ):
	''' Iterate through all permutations of gamma and bias
		Finds the best gamma and bias
	'''
	rv = []

	for gamma in gammas:
		for bias in biases:
			error = k_cross(train_data, label, k, gamma, bias)
			rv.append(error, gamma, bias)

	return min(rv, key = lambda x: x[0])

def k_cross(train_data, label, k, gamma):
	''' k-fold cross validation for gamma and bias
	'''
	length = train_data.shape[0]
	total_error = 0
	for i in range(0,length):
		k_data, k_label, valid_data, valid_label = partition(train_data, label, i, k)
		c = train(k_data, k_label, gamma, bias)
		labels, local_error = test(valid_data, valid_label, c)
		total_error += local_error

	return total_error

def test(test_data, label, c, bias) :
	instances = test_data.shape[0]
	total_error = 0
	for t in range(0,instances):
		x = train_data[t]
		temp += c[i] * rbf_kernel(x[i],x[t], gamma) + bias
		predict = 1 if temp >= 0 else -1

		if not predict == label[i]:
			total_error += 1

	return total_error/instances



def main(size):

	train_data  = read('data/train2k.databw.35', size)
    train_label = read('data/train2k.label.35', size)
    test_data   = read('data/test200.databw.35',200)
    test_label  = read('data/test200.label.35',200)

    # k-fold cross-validation parameters
    # bias and gamma are common ranges, need to find optimal pair

    k           = 10
    biases        = np.logspace(-5, 15, num = 21, base = 2)
    gammas       = np.logspace(-15, 3, num = 19, base = 2)

    error, gamma, bias = grid_search(train_data, label, k , gammas, biases ):
    
    train(train_data, train_label, sigma )

	



