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
import colorsys

def color_gradient(n = 20):
    hsv_tuples = [(0, 1, x*1.0/n) for x in range(n)]
    rgb_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples)
    return rgb_tuples

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
    colors      = color_gradient(20)
    # np.random.seed(29)

    plt.figure()

    for i in range(num_samples):
        y = np.random.multivariate_normal(means,cov)
        plt.plot(x, y, c =  colors[i])
    plt.savefig(outfile)

def k_matrix(xtrain, t_sq):
    size = xtrain.shape[0]
    matrix = np.zeros( (size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = kernal(xtrain[i], xtrain[j], t_sq)
    return matrix

def p_b():
    print "Starting p_b ..."
    outfile  = "output/pb.png"
    data     = read('data/gp.dat')
    N        = 100 # num points within a line 
    x        = np.linspace(0 ,1 , N)
    t_sq     = 0.12

    xtrain   = (data[:,0]).reshape((data.shape[0],1))
    ytrain   = (data[:,1]).reshape((data.shape[0],1))
    sigma_sq = 1;   

    means = np.zeros((N,1)) 
    var   = np.zeros((N,1))
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
    return means, var, ci, xtrain, ytrain

def p_c(means,var,ci, xtrain, ytrain):
    print "Starting p_c ..."
    num_samples = 20
    N           = 100
    x           = np.linspace(0 ,1 , N)
    colors      = color_gradient(20)
    outfile     = "output/pc.png"
    # 200 x 20 sized randomized matrix
    rand_norm   = np.random.randn(N,num_samples) 
    # xsim        = np.zeros(num_samples)
    xsim        = []
    plt.figure()
    for i in range(num_samples):
        xsim.append((np.diag( np.sqrt(var))).dot(rand_norm[:,i].reshape(1,100)).reshape(100,1) + means)
        plt.plot( x, xsim[i], c = colors[i], alpha = 0.5)

    plt.scatter(xtrain,ytrain, marker = "x")
    plt.plot(x, means, 'b')
    plt.plot(x, ci[:,0],'g')
    plt.plot(x, ci[:,1],'g')

    plt.savefig(outfile)

if __name__ == "__main__":
    # p_a()
    means, var, ci, xtrain, ytrain = p_b()
    p_c(means,var, ci, xtrain, ytrain)

