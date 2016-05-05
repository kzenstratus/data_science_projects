# -*- coding: utf-8 -*-
"""
Created on Sun Apr  17 12:16:19 2016

@author: kevinzen


"""
import numpy as np
from scipy.linalg import norm, eig,solve
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from pca import read,eigen_decomp
from numpy.linalg import inv

from isomap import knn
from lap import plot_points

COLOR_DIC = {1:'green',2:'yellow',3:'blue',4:'red'}


def make_gram(xi,matrix,k,src):
    ''' For each point, calculate a gram matrix.
    '''
    # subtract each neighbor by the xi point
    # Since our matrix only has cost, we run down each valid
    # column (non zero/inf), and calculate a gram matrix for each
    # matrix is a 1x500
    neighbors = []
    indices = [] # for use in mapping weights to big matrix
    for col in range(len(matrix)):
        # if we have a neighbor
        if not matrix[col] == 0:
            neighbor = src[col] # 3x1 point
            neighbors.append(neighbor)
            indices.append(col)

    # k-1 because nearest neighbor with itself is ignored
#    print len(neighbors)
#    assert(len(neighbors) == k)

    neighbors = np.array(neighbors)
    # subtracting each point in neighbors by xi.T
    # k x 3 - 3x1.T = kx3 - 1x3 = kx3
    neighbors = neighbors - xi.T
    # gram is now kxk
    gram = np.dot(neighbors,neighbors.T)
    return (gram,indices)

def set_weights(matrix,data,k,dim,src,n):
    ''' Given a xi, calculate the weight of each nearest neighbor
        such that, the linear combination of the nearest neighbors
        is approximately equal to xi.

        minimizing the cost function with constraints, take the langrangian,
        and the optimal weight wi = gram matrices
        
    '''

    # for each point, calculate the weights of 
    # it's nearest 10 nearest neighbors
    eps = 1e-3
    length = data.shape[1]

    # turn all the infinities into 0
    matrix[np.isinf(matrix)] = 0.0
    ones = np.ones((k,1))

    weighted = np.zeros((n,n))
    for i in range(length):
        xi = data[:,i].reshape(3,1)
        # xi is a 3x1 point
        gram,indices = make_gram(xi,matrix[i],k,src)
        # gram is a kxk matrix.
        regular = 0
        if k > 3:
        # regularization is kxk I* eps
            regular = np.eye(k)*eps

        weight = np.dot(inv(gram + regular),ones)
        weight = np.transpose(weight)
        # weight is now 1x10

        weighted[i,indices] = (weight/weight.sum())
    return weighted


def embedd_y(weighted,n):
    '''Embedd the y coordinates with weights W
    '''
    # nxn - nxn
    temp = np.eye(n) - weighted
    return np.dot(temp.T,temp)

def get_points(matrix,dim,src):

    eig_val, eig_vec = eig(matrix)

    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=(lambda x: x[0]))

    # reverse the order
#    index = index[::-1]
    top = eig_pairs[1:dim+1]
    points = []
    for e in top:
        points.append(e[1])
        
    return points


def lle(k,data,style_data,src,outfile,dim):
    '''1. Start with knn
    '''
    n = data.shape[1]
    # k+1 because nearest neighbor includes self
    matrix = knn(data,k+1)

    print "--- Seeting Weights on Adjacency Matrix ---\n"
    # weighted is a 500x500 matrix
    weighted = set_weights(matrix,data,k,dim,src,n)

    print "--- Embedding Coordinates ---\n"
    # matrix is nxn
    matrix = embedd_y(weighted,n)

    print "--- Eigendecomp -> Points ---\n"

    # return a 500x2 points
    points = get_points(matrix,dim,src)

    print "--- Plotting Points ---\n"

    plot_points(np.transpose(points),style_data,outfile)


def main():
    k = 10
    dim = 2
    datafile = "3Ddata.txt"
    outfile = "lle.png"
    temp = read(datafile)
    data = temp[0]
    style_data = temp[1]
    src = data.T

    
    lle(k,data,style_data,src,outfile,dim)


if __name__ == "__main__":
    main()
