# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:59:19 2016

@author: kevinzen
"""

import scipy
import random
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math

BLUE     = "#1a75ff"
RED      = "#ff5050"
GREEN    = "#98FB98"

def init_centroids(data_pnts,k):
    ''' Initializes random centroids
        data_pnts = 2d list
        return list of centroids as 2d list, uses with replacement
    '''
    pnts = np.array(data_pnts)
    return pnts[np.random.choice(pnts.shape[0],size = k,replace = False),:]

def read(datafile):
    ''' Reads in data as 2d list'''
    data_pnts = []
    with open(datafile) as infile:
        for line in infile:
            data_pnts.append([float(i) for i in line.split()])
    return data_pnts

def plot_final(dic,outfile):
    ''' Dic maps centroid coordinate with list of points
        Hard code in number of clusters
    '''
    colors = [BLUE,RED,GREEN]
    plt.figure()

    # stupid way of mapping colors with keys in dictionary
    count = 0
    for key, data_pnts in dic.iteritems():
        x_pnts = [x for [x,y] in data_pnts]
        y_pnts = [y for [x,y] in data_pnts]
        x_cent = [key[0]]
        y_cent = [key[1]]
        plt.scatter(x_pnts,y_pnts, c = colors[count],linewidths = 0, alpha = 0.8)
        plt.scatter(x_cent,y_cent, s = 40, c = colors[count], linewidths = 2, alpha = 0.8)
        count +=1
    plt.savefig(outfile)