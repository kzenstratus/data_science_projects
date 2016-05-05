# -*- coding: utf-8 -*-
"""
Created on Sun Apr  17 12:16:19 2016

@author: kevinzen
credit :https://github.com/daisukekobayashi/drpy/blob/master/laplacian_eig.py

"""
import numpy as np
import math
from scipy.linalg import norm, eig
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.neighbors import KNeighborsClassifier
from pca import read,eigen_decomp
from numpy.linalg import inv

from l2_distance import l2_distance
from isomap import knn,plot_data

COLOR_DIC = {1:'green',2:'yellow',3:'blue',4:'red'}

def set_weights(matrix,k,heat):
    '''container : adjacency matrix, where all
       non neighbors are zero, and all neighbors
       have a heated weight in place of the euclidean
       distance.
       Heat Equation =  exp -{||x_i-x_j||^2/4t}

       '''
    infinity = float("inf")
    temp_matrix = matrix.copy()
    container = np.zeros(matrix.shape)

    # Find the minimum values across each row of 
    # adjacency matrix.
    # This creates a container to map
    # future heated values.

    for i in range(matrix.shape[0]):
        matrix[i, i] = infinity
        for j in range(k):
            idx = matrix[i].argmin()
            container[i, idx] = 1.0
            container[idx, i] = container[i,idx]
            matrix[i, idx] = infinity

    # Prevents divide by 0
    if not heat == 0:
        # find all non zero entries in your graph
        # in your nearest neighbor matrix.
        # indices[0] = x values, indices[1] = y
        indices = np.nonzero(container)
        # Use the heat equation
        heated = np.exp(-temp_matrix[indices] / heat)

        # map the heats to the container
        container[indices] = container[indices] * heated
    return container

def get_laplacian(matrix):
    '''Laplacian is just the difference between
        the diagonal matrix and the weighted graph.
    '''
    weight = np.diag(matrix.sum(1))
    laplacian = weight - matrix
    laplacian[np.isinf(laplacian)] = 0
    laplacian[np.isnan(laplacian)] = 0

    return (laplacian,weight)

def eigen_decomp(laplacian,weight, d):
    '''Return a list of n points in d dimensions'''
    eig_val, eig_vect = eig(laplacian, weight)
    # sort and find the smallest two eigen vectors
    eigen_indices = np.real(eig_val).argsort()

    # pick the first d dimensions, ignore the first one
    # at index 0
    indices = eigen_indices[1:d+1]
    points  = np.real(eig_vect)[:,indices]

    return points

def plot_points(points,style_data,outfile):

    for i in range(len(style_data)):
        col = COLOR_DIC[style_data[i]]
        x = points[i][0]
        y = points[i][1]

        plt.plot(x,y,'o', markersize=7, color=col, alpha=0.5)
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        # plt.title('Transformed samples with class labels')
     
    plt.savefig(outfile)
    plt.show()

def lap(k,data,style_data,dim,outfile):
    ''' 1. Construct your k nearest neighbors adjacency matrix
        2. Use Heat Equation exp -{||x_i-x_j||^2/4t}
    '''    
    adjacency = knn(data,data.shape[1])
    heat = 1.0
    print "--- Seeting Weights on Adjacency Matrix ---\n"
    heat_matrix = set_weights(adjacency,k,heat)

    print "--- Creating Laplacian Matrix ---\n"

    laplacian, weight = get_laplacian(heat_matrix)

    print "--- Eigendecomp -> Points ---\n"

    points = eigen_decomp( laplacian,weight,dim)

    print "--- Plotting Points ---\n"

    plot_points(points,style_data, outfile)

def main():
    k = 10
    dim = 2

    datafile = "3Ddata.txt"
    outfile = "lap.png"
    temp = read(datafile)
    data = temp[0]
    style_data = temp[1]
    
    lap(k,data,style_data,dim,outfile)


if __name__ == "__main__":
    main()
