# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:16:19 2016

@author: kevinzen

"""
import math
import numpy as np
from utils import *
from random import random

def calc_output( bias, weights, pic):
    # returns 1 if true, else 0
    return (1) if sum(weights * pic) + bias >= 0 else (-1)

def train(train_data,label, max_iter,err):

    # Use max_iter to determine if online or batched
    is_online = True if max_iter == 1 else False


    dim = train_data.shape[1] # num dimensions, ~750 
    num_pics = train_data.shape[0] # num pics, ~2k

    weights = np.zeros(dim) # size = dim
    tot_error = float("inf")
    err_array = []# used purely for plotting errors
    
    while ((max_iter > 0) and (tot_error > err)) :
        tot_error = 0.0

        for i in range(0,num_pics) :
            x = normalize(train_data[i])

            # calculate dot product
            predict = 1 if weights.dot(x.reshape(dim,1)) >= 0 else -1

            if (predict == -1) and (label[i] == 1):
                tot_error += 1
                weights += x
            if (predict == 1) and (label[i] == -1):
                tot_error += 1
                weights -= x
            err_array.append(tot_error)
        print "total error is ", max_iter, tot_error
        max_iter -= 1

    if is_online: plot_errors(err_array, 'output/lin_online_err.png')
    return weights



def test(weights,bias,label, test_data):

    tot_error = 0
    pred_labels = []

    for i in range(0,test_data.shape[0]):

        output = calc_output( bias, weights, test_data[i])

        pred_labels.append(output)

        local_error = label[i] - output

        # print "local error:", local_error
        tot_error += abs(local_error/2)

    return tot_error, pred_labels

def k_cross(train_data, train_label, k, max_iter):
    ''' k-fold cross validation for gamma and bias
    '''
    rv = []
    for i in xrange(0,int(0.01 * train_data.shape[0]),2):
        tot_error = 0
        print "iteration ",i
        for j in range(0,k):
            k_data, k_label, valid_data, valid_label = partition(train_data, train_label, j, k)

            weights = train(k_data, k_label, max_iter ,float(i))
            batch_error, pred_labels = test(weights, 0, valid_label, valid_data)

            tot_error += float(batch_error)/(float(valid_label.shape[0]))
            print "Iteration ",i,"Partition",j, "Error", batch_error, "Total Error: ", tot_error

        # divide by k to normalize on 0-1 scale (purely for graph plotting)
        rv.append((float(i),tot_error))

    plot_final(rv, "output/linear_cross_valid.png")

    return min(rv, key = lambda x: x[1])


def main(size,is_online, find_thresh = False):
    ''' 1  -> 3
        -1 -> 5
    '''
    
    train_data  = read('data/train2k.databw.35', size)
    train_label = read('data/train2k.label.35', size)
    test_data   = read('data/test200.databw.35',200)
    test_label  = read('data/test200.label.35',200)
    bias        = 0 # set to 0 cause we centralized data
    max_iter    = 200
    k           = 10  # how many partitions to break up grid search cross-valid
    thresh      = 6.0 # already computed best threshold is 6.0
    outfile     = "output/batch_linear"

    if find_thresh :
        best = k_cross(train_data, train_label, k, max_iter)
        print "THRESH IS ", best
        thresh = best[0]

    if is_online:
        max_iter = 1
        outfile = "output/online_linear"

    weights = train(train_data,train_label,max_iter, thresh)
    pred_error, pred_labels  = test(weights, bias , test_label, test_data)
    np.save(outfile , np.array(pred_labels))

    print "Final Error is : ", pred_error

def test_np():
    b_kern  = np.load('output/batch_kernal.npy')
    b_lin   = np.load('output/batch_linear.npy')
    on_kern = np.load('output/online_kernal.npy')
    on_lin  = np.load('output/online_linear.npy')
    real  = read('data/test200.label.35',200)

    correct = [0,0,0,0]
    size = len(pred)
    for i in range(size):
        if b_kern[i] == pred[i]:
            correct[0] += 1
        if b_lin[i] == pred[i]:
            correct[1] += 1
        if on_kern[i] == pred[i]:
            correct[2] += 1
        if on_lin[i] == pred[i]:
            correct[3] += 1

    print "b_kern: ", correct[0]/size, "b_lin: ", correct[1]/size,\
            "on_kern: ", correct[2]/size, "on_lin: ", correct[3]/size 
    return correct


if __name__ == "__main__":
    main(2000, True)






