# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:16:19 2016

@author: kevinzen

"""
import math
import numpy as np
import scipy
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt

def read(filename, size = -1):
    ''' Import data as a 2d array, each row = 1 pic
        size used if you want to use smaller data set
    '''
    data_pnts = []

    with open(filename) as infile:
        if size == -1:
            for line in infile:
                data_pnts.append([float(i) for i in line.split()])
        else:
            for line in infile:
                if size <= 0 :
                    break
                else:
                    data_pnts.append([int(i) for i in line.split()])
                    size -= 1
    return np.array(data_pnts)

def plot_final(best,outfile):

    plt.figure()

    # for error, x in best:
    x_pnts = [x for (x,y) in best]
    y_pnts = [y for (x,y) in best]

    # plt.scatter(x_pnts,y_pnts, c = "#1a75ff",linewidths = 0, alpha = 0.8)
    plt.plot(x_pnts, y_pnts, '.r-')
    plt.savefig(outfile)

def normalize(train_data):
    return train_data * np.average(train_data)

def partition(train_data, label, index, k):
    ''' Input:
            k     = total number of partitions
            index = int delineating one partition
                ie. if k = 10, slice = 2, validation = second chunk
        Output:
            training   set    = k - 1 size
            training labels   = k - 1 size
            validation set    = size 1 partition
            validation labels = size 1 partition
    '''
    length    = train_data.shape[0]
    part_size = length/k
    start     = index * part_size
    end       = (index + 1) * part_size

    k_data  = np.concatenate((train_data[:start],train_data[end:]))
    k_label = np.concatenate((label[:start],label[end:]))
    valid_data  = k_data[start : end]
    valid_label = label[start : end] 

    return k_data, k_label, valid_data, valid_label