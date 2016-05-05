# -*- coding: utf-8 -*-
"""
Created on Sun Apr  17 12:16:19 2016

@author: kevinzen

"""
import scipy
import random
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from pca import read,eigen_decomp
from sklearn.neighbors import KNeighborsClassifier
COLOR_DIC = {1:'green',2:'yellow',3:'blue',4:'red'}


def get_neighbors(data,center,k):
    '''Given a point, return the k nearest neighbors
       Return as an array, from closest to farthest
    '''
    rv = []
    for i in range(data.shape[1]):
        xi = data[:,i].reshape(3,1)
        # xi is now in 3x1 form
        dist = distance.euclidean(center,xi)
        # append the index of the point
        rv.append((dist,i))
    rv.sort(key=(lambda x: x[0]))
    return rv[:k]

def insert_neighbor(matrix,row,neighbors):
    for cost,index in neighbors:
        if row ==index: matrix[row][index] = 0.0
        matrix[row][index] = cost
    return matrix

def knn(data,k):
    '''k nearest neighbors
       Create a 500x500 adjacency matrix, filled with cost (distance)
       The index corresponds to the first point in data
       i.e. the 3x1 reshaped point in data.
       The k nearest negihbors of x = find the row with x, all
       values on that row are the nearest neighbors.
    '''
    print "--- Calculatin k Nearest Neighbors ---\n"
    inf = float("inf")
    length = data.shape[1]
    matrix = np.empty((length, length))
    matrix.fill(inf)

    # equivalent of going down each row of the matrix.
    for i in range(length):
        xi = data[:,i].reshape(3,1)
        neighbors = get_neighbors(data,xi,k)
        matrix = insert_neighbor(matrix,i,neighbors)
        

    return matrix

def shortest_path(data,matrix):
    '''Use Floyd Warshall Algorithm for shortest path
       Between 2 nodes.
       1. Create a v by v matrix.
       2. Set all diagonal entries = 0
       3. Set all other entries to euclidean distance
       4. For everything in your adjacency matrix, replace 

    '''
    n = data.shape[1]
    print " --- Starting Floyd Washall ---\n"
    for k in range(n):
        print k
        for i in range(n):
            for j in range(n):
                if matrix[i][j] > (matrix[i][k] + matrix[k][j]):
                    matrix[i][j] = matrix[i][k] + matrix[k][j]

    return np.add(matrix,matrix.T)

# MDS
def gram_matrix(matrix,n):
    '''1. G = -1/2(PDP)
       G = centered gram matrix, 500x500 semi-definite matrix
       i.e. positive, with some 0s inside.? not necessarily p.s.d
       P = 500x500 centering matrix, P = I- 1/n(<1,1,1..n>T<1,1,1..n>) 
       D = squared distance matrix

    '''
    print "--- Calculating Gram Matrix ---\n"
    matrix = np.square(matrix)
    identity = np.identity(n)
    one = np.zeros((1,n))
    one.fill(1)
    centering = identity - (1/float(n))*np.dot(one.T,one)
    gram = -0.5*centering.dot(matrix).dot(centering)
    return gram

def mds(matrix,n,outfile):
    '''Code for multi-dimensional scaling
       Take the eigendecomp of the gram matrix.
       Take the p largest eigenvectors where
       rank(G)< p, 
       ? p should be bounded by the number of attributes?
       ? so p < 3?
    '''
    print "--- Starting Multi-Dimensional Scaling ---\n"
    gram = gram_matrix(matrix,n)
    print "--- Starting Eigen Decomp --- \n"
    matrix_w = eigen_decomp(gram,dim= n)

    return matrix_w
    
def plot_data(eigenvectors,outfile,style_data,data):
    ''' Map each point using matrix_w, hard code 2x3 dot 3x1 = 2x1
        Now in 2d space.
    '''    
    for i in range(len(style_data)):
        col = COLOR_DIC[style_data[i]]
        plt.plot(eigenvectors[0,i], eigenvectors[1,i], 'o', markersize=7, color=col, alpha=0.5)
        # plt.xlim([-4,4])
        # plt.ylim([-4,4])
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        # plt.title('Transformed samples with class labels')
     
    plt.savefig(outfile)
    plt.show()
    

def isomap(k,data,style_data,outfile):
    '''Send 500x500 geodesic matrix into mds
       Just take your first two smallest eigenvalues and plot those.
    '''
    n = data.shape[1]
    dist_matrix = knn(data,k)
    shortest_matrix = shortest_path(data,dist_matrix)
    matrix_w = mds(shortest_matrix,n,outfile)

    plot_data(matrix_w.T,outfile,style_data,data)


def main():
    k = 10
    datafile = "3Ddata.txt"
    outfile = "isomap.png"
    temp = read(datafile)
    data = temp[0]
    style_data = temp[1]
    
    isomap(k,data,style_data,outfile)


if __name__ == "__main__":
    main()
