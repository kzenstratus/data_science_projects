# -*- coding: utf-8 -*-
"""
Created on Sun Apr  17 12:16:19 2016

@author: kevinzen
Credit: PCA: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#drop_labels

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
    
COLOR_DIC = {1:'green',2:'yellow',3:'blue',4:'red'}


def read(datafile):
    ''' Reads in data as a 2d list
        Points used for styling are stripped and returned as 1x500
        array in second tuple
    '''
    data_pnts = []
    with open(datafile) as infile:
        for line in infile:
            data_pnts.append([float(i) for i in line.split()])
    pnts = np.transpose(np.array(data_pnts))
    return (pnts[:3],pnts[3])

def eigen_decomp(covar_matrix, dim):
    '''Compute eigenvectors and eigenvalues
       Do this with an eigenvector decomposition
    '''
    eig_val, eig_vect = np.linalg.eig(covar_matrix)
    
    # transform them into tuples (eigenvalue,eigenvector)
    eig_pairs = [(np.abs(eig_val[i]), eig_vect[:,i]) for i in range(len(eig_val))]
    eig_pairs.sort(key=(lambda x: x[0]))
    eig_pairs.reverse()

    # Since we are mapping to R2, we pick the 2 biggest eigen values
    # Will form a 3x2 matrix.
    matrix_w = np.hstack((eig_pairs[0][1].reshape(dim,1), eig_pairs[1][1].reshape(dim,1)))
    return matrix_w

def plot_data(transformed,outfile,style_data,data):
    ''' Map each point using matrix_w, hard code 2x3 dot 3x1 = 2x1
        Now in 2d space.
    '''

    for i in range(len(style_data)):
        col = COLOR_DIC[style_data[i]]
        plt.plot(transformed[0,i], transformed[1,i], 'o', markersize=7, color=col, alpha=0.5)
        plt.xlim([-4,4])
        plt.ylim([-4,4])
        plt.xlabel('x_values')
        plt.ylabel('y_values')
        plt.title('Transformed samples with class labels')
     
    plt.savefig(outfile)
    plt.show()
    


def main():

    datafile = "3Ddata.txt"
    outfile = "pca.png"
    temp = read(datafile)
    data = temp[0]
    style_data = temp[1]

    # Find the center of the data:
    x_mean = np.mean(data[0])
    y_mean = np.mean(data[1])
    z_mean = np.mean(data[2])
    mean_vector = np.array([[x_mean],[y_mean],[z_mean]])

    # Find the principle components:
    # Find the covariance matrix 
    covar_matrix = np.zeros((3,3))
    for i in range(data.shape[1]):
        xi = data[:,i].reshape(3,1) - mean_vector
        covar_matrix += xi.dot(xi.T)

    
    # mapping to R2, form 3x2 matrix to map datta to
    matrix_w = eigen_decomp(covar_matrix,dim = 3)
    # print len(matrix_w), len(matrix_w[0])
    transformed = matrix_w.T.dot(data)
    print len(transformed[0])
    plot_data(transformed,outfile,style_data,data)




    





if __name__ == "__main__":
    main()