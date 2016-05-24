# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:16:19 2016

@author: kevinzen

"""
import math
import numpy as np
from utils import *
from random import random

def calc_output(thresh, bias, weights, pic):
    # returns 1 if true, else 0
    return (1) if sum(weights * pic) + bias >= thresh else (-1)

def update_weights(learn, local_error, pic, weights):
    temp = learn * local_error * pic
    return temp + weights


def train_online(train_data,label,thresh):
    
    dimension = train_data.shape[1]

    weights = np.zeros(train_data.shape[1])
    num_pics = train_data.shape[0]
    tot_error = 0
    for i in range(0,num_pics) :
        x = train_data[i].reshape(dimension,1)
        predict = 1 if weights.dot(x) >= 0 else -1

        if (predict == -1) and (label[i] == 1):
            tot_error += 1
            weights += train_data[i]
        if (predict == 1) and (label[i] == -1):
            tot_error += 1
            weights -= train_data[i]

    print "total error is ", tot_error
    return weights


def train_batch(train_data,label,thresh,max_iter):
    # Init parameters
    weights   = np.random.rand(train_data.shape[1]) # 1x 28^2 random array
    learn     = 0.1 # learning rate
    bias      = random() # random bias
    tot_error = float("inf")

    while ((max_iter > 0) and (tot_error > 0)) :

        tot_error = 0.0
        for i in range(0,train_data.shape[0]):
            output = calc_output(thresh, bias, weights, train_data[i])
            local_error = label[i] - output
            weights = update_weights(learn, local_error, train_data[i], weights)

            bias +=  learn * local_error

            tot_error += local_error * local_error

        print "iteration", max_iter, "total_error",tot_error
        max_iter -= 1

    return weights, bias

def test(weights,bias,label, test_data,thresh):
    tot_error = 0
    for i in range(0,test_data.shape[0]):
        output = calc_output(thresh, bias, weights, test_data[i])
        local_error = label[i] - output
        print "local error:", local_error
        tot_error += abs(local_error/2)
    print tot_error
    return tot_error

def main(size,is_online):
    ''' 1 -> 5
        0 -> 3
    '''
    
    train_data  = read('data/train2k.databw.35', size)
    train_label = read('data/train2k.label.35', size)
    test_data   = read('data/test200.databw.35',200)
    test_label  = read('data/test200.label.35',200)
    thresh      = 0
    max_iter    = 200

    if is_online:
        weights = train_online(train_data, train_label, thresh)
        online_error = test(weights, 1, test_label, test_data, thresh)
    else:
        weights, bias = train_batch(train_data,train_label,thresh,max_iter)
        batch_error  = test(weights, bias, test_label, test_data, thresh)









