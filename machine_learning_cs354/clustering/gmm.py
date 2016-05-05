# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 09:59:19 2016

@author: kevinzen
Collaborated with Malika Aubakirova
"""
import scipy
import random
import numpy as np
import math
import utils
from scipy import stats
from scipy.stats import multivariate_normal
from numpy import linalg as LA

MAX_ITER = 20
THRESHOULD = 0.01



def initialize(data_pnts,k):
    '''Randomly select points to serve as means
       Covariance matrix equals covariance of full training set
       Each cluster has equal prior probability. 
       Dataset is equally divided among clusters
       Each cluster has a mean, a probability of coming from it,
       and a covariance matrix of the zth gaussian
    '''
    # randomly find k gaussians from n datapoints, with equal prob
    means = utils.init_centroids(data_pnts,k)
    # probability is uniform, fill pi
    pi = [1/float(k) for i in range(k)]
    # covar matrix initialized as identity matrix
    # initialize it as a 2x2 matrix
    covar = [np.identity(2) for j in range(k)]


    # covar = [np.dot(means[j] for j in range(k)]

    return (means,pi,covar)

def e_step_helper(data_pnts,k,pdf,pi,means,covar,k_curr):
    ''' Given the pdf, iterate through each point
        and return list of probabilities
    '''
    probs = []
    for i in data_pnts:
        numerator = pi[k_curr]*pdf.pdf(i)

        # here point i is static, already picked i, prob of
        # coming from all k clusters = total = denom
        # Creating the denominator
        denom = 0
        for z in range(k):
            # pdf returns a point
            pdf_z = multivariate_normal(mean = means[z],cov = covar[z])

            denom += pi[z]*pdf_z.pdf(i)

        probs.append(numerator/denom)
    return probs

def e_step(data_pnts,k,package):
    '''Calculate the probability that each data point
       belongs to each cluster.
       Store clusters as keys and list of probabilties as
       values.
    '''
    means = package[0]
    pi = package[1]
    covar = package[2]
    denom = 0
    dic = {}
    for z in range(k):
        var = multivariate_normal(mean = means[z],cov = covar[z])
        probs = e_step_helper(data_pnts,k,var,pi,means,covar,k_curr = z,)
        dic.setdefault(tuple(means[z]),[]).append(probs)

    # dic contains cluster means -> [probabilities for each point]

    return dic

def calc_covar(data_pnts,mu_j,probs,denom):
    '''Calculate covariance for m step'''
    numerator = 0
    for i in range(len(data_pnts)):
        diff = convert_to_2d(data_pnts[i] - np.array(mu_j))
        numerator += (probs[i]*np.dot(diff, np.transpose(diff)))

    return (numerator/denom)

def convert_to_2d(point):
    '''Simple data structure conversion for matrices in calc_covar'''
    x = [[point[0]], [point[1]]]
    return np.array(x)

def calc_mu(data_pnts,probs):
    '''Calculate the mu for update'''
    numerator = 0
    denom = 0
    for i in range(len(data_pnts)):
        numerator += (np.dot(probs[i],data_pnts[i]))
        denom += probs[i]
    return ((numerator/denom),denom)

def m_step(data_pnts,k,dic):
    '''Update pi,'''
    mu= []
    pi = []
    covar= []
    for key, value in dic.iteritems():
        pi.append(np.mean(value[0]))

        # Calculate mu
        mu_temp = calc_mu(data_pnts,value[0])
        mu.append(mu_temp[0])
        denom = mu_temp[1]

        # calculate covar
        covar.append(calc_covar(data_pnts,mu_temp[0],value[0],denom))

    return (mu,pi,covar)

def converges(package,count,data_pnts,k,old_package):
    '''Package refers to the tuple with mu,pi,sigma'''

    if count > MAX_ITER: 
         return True
    if not old_package: return False

    logl_old = loglike(old_package,data_pnts,k)
    logl = loglike(package,data_pnts,k)
    return (np.abs(logl - logl_old) < THRESHOULD)


def loglike(package, data, k):
    ''' Calculate log liklihood for convergence tests'''
    logl = 0.
    for t in range(len(data)):
        p_xt_theta = 0
        for i in range(k):
            p_xt_theta += (package[1][i] *multivariate_normal.pdf(
                data[t], mean=package[0][i], cov=package[2][i]))
        logl += np.log(p_xt_theta)
    return logl

def form_clusters(dic,data_pnts):
    ''' Given package of info, sort the data into 3 clusters'''
    probs = []
    clusters = []
    # stupid way of unpackaging dictionary
    for cluster, prob in dic.iteritems():
        clusters.append(cluster)
        probs.append(prob[0])
    # maximum is a 500x1 array with highest probabilities
    maximum = np.argmax(probs,axis = 0)

    final_dic = {}
    index = 0
    for i in maximum:
        final_dic.setdefault(tuple(clusters[i]),[]).append(data_pnts[index])
        index += 1
    return final_dic
    
def run(data_pnts,k):
    '''Runs the whole program'''
    package = initialize(data_pnts,k)
    count = 0
    old_package = None

    while not converges(package,count,data_pnts,k,old_package):
        old_package = package
        dic = e_step(data_pnts,k,package)
        package = m_step(data_pnts,k,dic)

        count += 1
    final_dic = form_clusters(dic,data_pnts)
    utils.plot_final(final_dic,"gmm.png")

    return (final_dic,dic)

def main():
    '''Instantiates program wide info'''
    k = 3
    datafile = "toydata.txt"
    data_pnts = utils.read(datafile)

    rv = run(data_pnts,k)
    


if __name__ == "__main__":
    main()
