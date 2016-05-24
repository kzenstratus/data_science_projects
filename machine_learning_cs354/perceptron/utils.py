# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:16:19 2016

@author: kevinzen

"""
import math
import numpy as np


def read(filename, size):
    ''' Import data as a 2d array, each row = 1 pic
        size used if you want to use smaller data set
    '''
    data_pnts = []
    with open(filename) as infile:
        for line in infile:
            if size <= 0 :
                break
            else:
                data_pnts.append([int(i) for i in line.split()])
                size -= 1
    return np.array(data_pnts)