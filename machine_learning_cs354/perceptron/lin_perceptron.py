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

def update_weights(learn, local_error, pic, weights):
    temp = learn * local_error * pic
    return temp + weights

def normalize(train_data):
    return train_data * np.average(train_data)

def train_online(train_data,label, max_iter,err):
    
    dim = train_data.shape[1]

    weights = np.zeros(train_data.shape[1])
    num_pics = train_data.shape[0]
    tot_error = float("inf")
    while ((max_iter > 0) and (tot_error > err)) :
        tot_error = 0.0

        for i in range(0,num_pics) :
            x = normalize(train_data[i])
            predict = 1 if weights.dot(x.reshape(dim,1)) >= 0 else -1

            if (predict == -1) and (label[i] == 1):
                tot_error += 1
                weights += x
            if (predict == 1) and (label[i] == -1):
                tot_error += 1
                weights -= x

        # print "total error is ", max_iter, tot_error
        max_iter -= 1
    return weights


def train_batch(train_data,label,max_iter):
    # Init parameters
    # weights   = np.random.rand(train_data.shape[1]) # 1x 28^2 random array
    weights   = np.zeros(train_data.shape[1])
    learn     = 0.1 # learning rate
    bias      = random() # random bias
    tot_error = float("inf")

    while ((max_iter > 0) and (tot_error > 0)) :

        tot_error = 0.0
        for i in range(0,train_data.shape[0]):
            output = calc_output( bias, weights, train_data[i])
            local_error = label[i] - output
            weights = update_weights(learn, local_error, train_data[i], weights)

            bias +=  learn * local_error

            tot_error += local_error * local_error

        print "iteration", max_iter, "total_error",tot_error
        max_iter -= 1

    return weights, bias

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

            weights = train_online(k_data, k_label, max_iter ,float(i))
            batch_error, pred_labels = test(weights, 0, valid_label, valid_data)

            tot_error += float(batch_error)/(float(train_data.shape[0]))
            print "Iteration ",i,"Partition",j, "Error", batch_error, "Total Error: ", tot_error

        # divide by k to normalize on 0-1 scale (purely for graph plotting)
        rv.append((float(i),tot_error/k))

    plot_final(rv, "linear_cross_valid.png")

    return min(rv, key = lambda x: x[1])


def main(size,is_online):
    ''' 1  -> 3
        -1 -> 5
    '''
    
    train_data  = read('data/train2k.databw.35', size)
    train_label = read('data/train2k.label.35', size)
    test_data   = read('data/test200.databw.35',200)
    test_label  = read('data/test200.label.35',200)
    max_iter    = 200
    k           = 10 # how many partitions to break up cross-valid

    best = k_cross(train_data, train_label, k, max_iter)
    print "THRESH IS ", best
    thresh = best[0]


    if is_online:
        weights = train_online(train_data, train_label, 1, thresh)
        online_error, pred_labels = test(weights, 0, test_label, test_data)
        np.save("output/online_linear",np.array(pred_labels))
        print "ONLINE ERROR IS :", online_error

    else:

        weights = train_online(train_data,train_label,max_iter, thresh)
        batch_error, pred_labels  = test(weights, 0 , test_label, test_data)
        np.save("output/batch_linear", pred_labels)

        print "BATCH ERROR IS ", batch_error









