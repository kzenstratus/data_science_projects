# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:59:19 2016

@author: kevinzen
"""
import scipy
import random
import numpy as np
import math
import k_means, utils
import matplotlib.pyplot as plt

from scipy import stats

def d_squared(centroids,data_pnt):
	''' Given a point, calculate D(x)^2
	'''
	return math.pow((k_means.assign_centroid_helper(centroids,data_pnt)[1]),2)

def new_prob(centroids,data_pnts):
	''' 
		Need to map each point with a probability
	    Return a list of probabilities, mapped to coords by indices.
	'''
	denom = 0.0
	num   = []
	for pnt in data_pnts:
		distance = d_squared(centroids,pnt)
		num.append(distance)
		denom += distance

	return np.array(num)/denom


def new_center(centroids,data_pnts):
	''' use discrete to generate random indices based on distribution
	    which map to a datapoint.
	'''
	prob_array = new_prob(centroids,data_pnts)
	indices = np.arange(len(data_pnts))
	# 
	custm_gen = stats.rv_discrete(values=(indices, prob_array))
	index = custm_gen.rvs()

	return data_pnts[index]

def get_centroids(data_pnts,k):
	centroids = (utils.init_centroids(data_pnts,1).tolist())

	# rest of the centroids are determined by how close they are to existing centroids
	for i in range(1,k):
		centroids.append(new_center(centroids,data_pnts))

	# Finished initializing, continue with regular k means
	# print centroids
	centroids = np.array(centroids)
	return centroids

def main():
	'''
	1. choose center uniformly
	2. compute distance between x and nearest center D(x)
	3. Choose 1 new data point at random as new center
	using weighted probability distribution where x is 
	chosen with prob proportion D(x)^2
	4. Repeat 2-3 until k centers have been chosen
	5. continue using k-means
	'''
	k = 3
	datafile = "toydata.txt"
	data_pnts = utils.read(datafile)
	#initialize first centroid

	centroids = get_centroids(data_pnts,k)
	# stupid way of running plots 20 times, too tired to be more elegant
	rv = k_means.run(centroids,data_pnts,k,"cost++.png")
	for j in range(19):
		centroids = get_centroids(data_pnts,k)
		temp = k_means.run(centroids,data_pnts,k,"cost++.png")
		rv = np.concatenate((rv,temp),axis = 0)

	plt.figure()

	for k in rv:
		k_means.plot_cost(k[0],k[1])
	plt.savefig("cost++.png")





if __name__ == "__main__":
    main()