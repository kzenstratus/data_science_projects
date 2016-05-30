# -*- coding: utf-8 -*-
"""
Created on Tue May 24 01:26:10 2016

@author: kevinzen

"""

import math
import numpy as np
from numpy import linalg as LA
from utils import *
from random import random
from scipy.spatial import distance

def rbf_kernel(x_i, x_t, gamma) :
    return math.exp(- gamma * distance.sqeuclidean(x_i, x_t) )

def rbf_predict(alpha, instances , train_data, x_t, gamma):
    temp = 0
    # calcualte local error and update c from 0 to curr instance
    for i in range(0, instances) :
        x_i = normalize(train_data[i])
        # x_i = train_data[i]
        # temp += alpha[i] * rbf_kernel(x_i ,x_t, gamma)
        temp += alpha[i] * lin_kernal(x_i ,x_t)

    return 1 if temp >= 0 else -1

def lin_kernal(x_i, x_t):
    ''' Test to see if kernal produces same results as lin_percep'''
    return x_i.dot(x_t)

def train(train_data,label, gamma, max_iter, err = 6):
    instances = label.shape[0]
    alpha     = np.zeros(instances)
    tot_error = float("inf")

    while ((max_iter > 0) and (tot_error > err)) :
        # iterate through all instances
        tot_error = 0

        for t in range(0,instances) :

            x_t = normalize(train_data[t])
            # x_t = train_data[t]
            predict = rbf_predict(alpha, instances, train_data, x_t, gamma)
            # alpha[t] = 0

            # Update c according to if output was right or wrong
            if predict != label[t]:
                tot_error += 1
                alpha[t] += label[t]

        max_iter -= 1
        print "Current iter:", max_iter,"tot_error: ", tot_error
    return alpha

def grid_search(train_data, label, k, gammas):
    ''' Iterate through all permutations of gamma and bias
        Finds the best gamma and bias
    '''
    rv = []
    for gamma in gammas:
        print "GAMMA,", gamma

        error = k_cross(train_data, label, k, gamma)
        rv.append((gamma, error))

    plot_final(rv, "output/rbf_cross_valid.png")

    return min(rv, key = lambda x: x[1])

def k_cross(train_data, label, k, gamma):
    ''' k-fold cross validation for gamma
    '''
    best = []
    for i in xrange(0,k,2):
        k_data, k_label, valid_data, valid_label = partition(train_data, label, i, k)

        # avoid divide by zero error in test func
        if valid_data.shape[0] == 0:
            continue
        alpha = train(k_data, k_label, gamma, 1)

        labels, correct = test(valid_data, k_data, valid_label, alpha, gamma)
        best.append(correct)
 
        print "########### CROSS VALID ", i, "local local best = ", correct
    rv = np.mean(best)
    print "K FOLD - Local Gamma correct: ", rv

    return (1 - rv)

def test(test_data, train_data, test_label, alpha, gamma) :
    instances = train_data.shape[0]
    pics      = test_data.shape[0]
    rv_label  = []

    correct = 0.0
    for t in range(0,pics) :
        predict = rbf_predict(alpha, instances, train_data, test_data[t], gamma)

        rv_label.append(predict)

        if predict == test_label[t]:
            correct += 1.0

    print "Correct IS ", correct, "Pics", (correct/pics)

    return np.array(rv_label), (correct/pics)
            



def main(size, is_online, find_gamma = False):

    train_data  = read('data/train2k.databw.35', size)
    train_label = read('data/train2k.label.35', size)
    test_data   = read('data/test200.databw.35',200)
    test_label  = read('data/test200.label.35',200)
    k           = 10
    gamma       = math.pow(2,(-6))
    max_iter    = 100   
    outfile     = "output/batch_kernal"


    # k-fold cross-validation parameters
    # bias and gamma are common ranges, need to find optimal pair

    # not using bias right now, normalizing instead
    # biases  = np.logspace(-5, 15, num = 21, base = 2)


    if find_gamma:
        gammas  = np.logspace(-5, 5, num = 11, base = 2)
        gamma, error= grid_search(train_data, train_label, k , gammas)
    # print "FINAL RESULT", gamma, error, ^-6

    if is_online:
        max_iter = 1
        outfile = "output/online_kernal"

    print "GAMMA VALUE IS ", gamma
    print "Starting training ..."

    alpha = train(train_data, train_label, gamma, max_iter)

    print "Starting testing ..."
    final_labels, tot_error = test(test_data, train_data, test_label, alpha, gamma)
    np.save( outfile, final_labels)


if __name__ == "__main__":
    main(2000, False)

    



