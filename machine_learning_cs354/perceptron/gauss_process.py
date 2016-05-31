# -*- coding: utf-8 -*-
"""
Created on Thu May 26 15:00:10 2016

@author: kevinzen

"""

import math
import numpy as np
from numpy import linalg as LA
from utils import *
from random import random
from scipy.spatial import distance
import matplotlib.pyplot as plt


def kernal(x, x_p, t_sq):
    ''' x_p = x'
        k(x,x') = exp( (-(x - x')^2)/ (2t^2) )
        t_sq = t^2
    '''
    return math.exp( -((x - x_p) * (x - x_p) / (2 * t_sq)) )

def covar(arr, t_sq) :
    ''' Calculate the covariance matrix
        using rbf kernels
        Input: 
            arr = array of x's
    '''
    n = arr.shape[0]
    cov = np.zeros(( n, n))
    for i in range(n):
        for j in range(n):
            cov[i][j] = kernal(arr[i], arr[j], t_sq)
    return cov

def p_a():
    ''' Code for problem (a)'''
    mean        = 0
    t_sq        = 0.12
    num_samples = 20
    N           = 100
    x           = np.linspace(0 ,1 , N)
    cov         = covar(x, t_sq)
    means       = np.zeros(N)
    outfile     = "output/pa.png"  
    # np.random.seed(29)

    # credit to 
    # http://stackoverflow.com/questions/4873665/joining-two-2d-numpy-arrays-into-a-single-2d-array-of-2-tuples
    # best = np.vstack(([x.T], [y.T])).T
    plt.figure()

    for i in range(num_samples):
        y = np.random.multivariate_normal(means,cov)
        plt.plot(x, y, '.r-')
    plt.savefig(outfile)

def k_matrix(xtrain, t_sq):
    size = xtrain.shape[0]
    matrix = np.zeros( (size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = kernal(xtrain[i], xtrain[j], t_sq)
    return matrix

def p_b():

    outfile  = "output/pb.png"
    data     = read('data/gp.dat')
    N        = 100 # num points within a line 
    x        = np.linspace(0 ,1 , N)
    t_sq     = 0.12

    xtrain   = (data[:,0]).reshape((data.shape[0],1))
    ytrain   = (data[:,1]).reshape((data.shape[0],1))
    sigma_sq = 1;

    

    means = np.zeros(N) 
    var   = np.zeros(N)
    ci    = np.zeros((N,2))
    for i in range(N):

        kx = np.zeros((data.shape[0],1))
        for j in range(data.shape[0]):

            kx[j] = kernal(x[i], xtrain[j], t_sq)

        k_corner = kernal(x[i], x[i], t_sq)

        k_mat    = k_matrix(xtrain,t_sq)
        means[i]  = kx.T.dot( LA.inv((k_mat + sigma_sq * np.eye(k_mat.shape[0])))).dot(ytrain)
        var[i]   = k_corner - kx.T.dot(LA.inv(k_mat + sigma_sq * np.eye(k_mat.shape[0]))).dot(kx)
        ci[i][0] = means[i] - 2 * math.pow(var[i],0.5) # store lower CI
        ci[i][1] = means[i] + 2 * math.pow(var[i],0.5) # store upper CI


    plt.figure()

    plt.scatter(xtrain,ytrain)
    plt.plot(x, means, 'r')
    plt.plot(x, ci[:,0],'g')
    plt.plot(x, ci[:,1],'g')

    plt.savefig(outfile)
if __name__ == "__main__":
    p_a()
    p_b()

