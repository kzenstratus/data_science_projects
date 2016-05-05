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
import utils

MAX_ITER = 100

 

def assign_centroid_helper(centroids,point):
    ''' calculate euclidean distance
        return min centroid and it's cost
    '''
    distances = []
    for center in centroids:
        distances.append(distance.euclidean(center,point))

    # array of distances, argmin to find index, return corresponding centroid
    index = np.argmin(distances)
    # (centroid,distance to centroid)
    return (centroids[index],distances[index])


def assign_centroids(data,centroids,k):
    '''Given point and centroids, assign a centroid to a point
        dic : key = centroid (as tuple), value = list of points
    '''
    dic = {}
    for c in centroids:
        dic.setdefault(tuple(c),[])
        
    total_cost = 0
    for point in data:
        temp = assign_centroid_helper(centroids,point)
        dic.setdefault(tuple(temp[0]),[]).append(point)
        total_cost += math.pow(temp[1],2) # temp[1] = J_avg^2
    # If a centroid is empty, we should randomly re-initialize it
    # Don't need to worry about that if we initialize our centroid as a point
    return (dic,total_cost)
    
def gen_new_centroids(dic):
    '''Given data point centroid assignment,
       find mean = new centroid
    '''
    new_centroids = []
    
    for key, value in dic.iteritems():
        # Goes through each centroid in dic, average all
        new_centroids.append(np.mean(value,axis = 0))
        # look into using geometric means
        
    return new_centroids

def is_stop(old_centroids,centroids,count):
    '''2 stopping conditions, if count exceeds limit
        or if your centroids are no longer changing
    '''
    if count > MAX_ITER: return True
    else: return np.array_equal(old_centroids, centroids)

def plot_cost(cost,iteration):
    ''' Plots single cost vs iteration'''
    plt.scatter(iteration,cost)





def run(centroids,data_pnts,k,outfile_1,outfile_2 = None):
    old_centroids = np.empty(centroids.shape)
    count = 0
    # plt.figure()

    # Iterate until convergence
    # print centroids
    total_cost = []
    while (not is_stop(old_centroids,centroids,count)):
        old_centroids = centroids
        # dic maps centroids with data points
        temp = assign_centroids(data_pnts,centroids,k)
        dic  = temp[0]
        cost = temp[1]
        centroids = gen_new_centroids(dic)
        total_cost.append([cost,count])

        count +=1

    # outputs iteration graph
    # plt.savefig(outfile_1)

    # outputs final graph, with all scatter points
    if outfile_2:
        utils.plot_final(dic,outfile_2)

    return total_cost

def main():

    datafile = "toydata.txt"
    k = 3

    # Read in file
    data_pnts = utils.read(datafile)
    centroids = utils.init_centroids(data_pnts,k)

    rv = run(centroids,data_pnts,k,"cost.png","k_means.png")
    # run through k_means ~ 19 times to account for random init
    
    for i in range(19):
        centroids = utils.init_centroids(data_pnts,k)

        temp = run(centroids,data_pnts,k,"cost.png")
        rv = np.concatenate((rv,temp),axis = 0)
    plt.figure()


    for j in rv:
        plot_cost(j[0],j[1])
    plt.savefig("cost.png")



if __name__ == "__main__":
    main()
    