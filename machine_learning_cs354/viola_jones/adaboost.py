# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:16:19 2016

@author: kevinzen

"""
import math
import numpy as np
from violajones import getWeakClassifier, sumRect
from scipy import misc
import glob
from utils import *
from cascade import apply_weak_classifier
from collections import deque
import time

    
def update_weights(weight,h,truth,beta):
  ''' Input: 
	      weight = original weights used 
        h_i = classifier to all validation images
        truth = 800x1 truth array for validation set
        beta = calculated from error in training set

	    Output:
	      new_weight = updated weights
  '''
  for i in range(h.shape[0]):

    e = 1
    if h[i] == truth[i]: e = 0

    weight[i][0] = weight[i][0]*beta**(1-e)

  # normalize the weights
  weight = weight/(np.sum(weight))
  return weight


def calc_error(h,truth,signed = 0):
  '''
    # CHANGED ZERO TO NEG
    Input:
      h = any classified array of 1s, 0s
      truth = actual classified array of 1s, 0s

    Calculates the false positive and false negative rates

    Output:
     returns false_pos, false_negative rates

  '''
  fpos = 0.0
  fneg = 0.0
  pos_tot = 0
  for i in range(len(h)):
    if truth[i] == 1 :
      pos_tot +=1

      if h[i] == signed :
        fneg += 1.0

    elif h[i] == 1 and truth[i] == signed: 
      fpos += 1.0

  neg_tot = len(truth) - pos_tot
  return (fpos/pos_tot), (fneg/neg_tot)


def calc_sign(h,alpha_tot, signed = 0):
  ''' 
    CHANGED ZERO TO NEG
    Converts any numpy array into a binary, 1 or 0
  '''
  for i in range(len(h)):
    if h[i] >= 0.5*alpha_tot: h[i] = 1
    else: h[i] = signed
  return h


def calc_classifiers(h, big_theta = 0):
  ''' Input:
        h = array of many alpha_t and h_t, such that we want to sum
        up alpha_t*h_t 
      Output:
        array of 1,0
  '''
  combined = np.zeros(h[0][1].shape)
  alpha_tot = 0.0
  for t in h:
    alpha_t = t[0]
    h_t     = t[1]
    alpha_tot += alpha_t
    combined += alpha_t * h_t
  return calc_sign(combined + big_theta, alpha_tot)


def boost(h,features,label,weight,Npos):
  '''
	   One Round of boosting
       Calculate new labels by parity*f_i < parity*theta
	   Input:
	     features = 300k x 3200 features from training data
	     weight = weights used determine importance of misclassified images
	     label = describes truth of training set
	     num_faces = number of positive images
	     valid = Dictionary of all the images in grayscale

	   Output:
         new_weights = re-calculated weights after applying h_i(x) to validation set
  '''

  min_error,thresh ,polar,featureIdx, h_t = getWeakClassifier(features,weight,label,Npos)
  print "Got my weak classification!"
  print "MY ERROR IS ", min_error
  halt = False
  try:
    beta_t = min_error/(1.0-min_error)
    alpha_t = math.log(1.0/beta_t)
  except:
    return h, weight, True

  weight = update_weights(weight,h_t,label,beta_t)

  h.append(tuple((alpha_t, h_t, featureIdx , polar, thresh)))

  return h, weight, halt


def apply_validation(h,valid,feat_index,label,truth,big_theta):
  ''' CHANGED ZERO TO NEG
    Input:
        h = (alpha, h_t, featureIdx, polar, thresh) 2d array T entries
        valid = dictionary containing all validation rectangles
        feat_index = lookup table for all features
        label = truth vector for all training images

      Output:
        Current false positive and false negative rates
      Find big theta, and run the validation set and 
      training set on that
  '''
  pos_list = valid["pos"]
  neg_list = valid["neg"]
  combined = pos_list+neg_list # first half is positive, second is negative

  h_valid  = [] # all rounds of boosting
  h_valid_temp = np.zeros(len(combined)) # One round of boosting
  
  # create the shfited classifier
  for t in range(len(h)):
    alpha_t = h[t][0]
    thresh  = h[t][4]
    parity  = h[t][3]
    feature = feat_index[h[t][2]]
    # creates h_i
    print "Applying the weak classificaiton to a validation data"
    for i in range(len(combined)):
      # for each image in combined, apply the weak classifier
      img_gray = combined[i]

      # classified array of 1,0, 800x1
      h_valid_temp[i] = apply_weak_classifier(img_gray,thresh,parity,feature)
    
    h_valid.append(tuple(( alpha_t, h_valid_temp )))

  # print "MY BIG THETA IS :", big_theta

  return test_big_theta(big_theta, h_valid, truth, label, h)


def test_big_theta(big_theta,h_valid,truth,label,h):
  ''' 
    Input:
      big_theta = used to shift the strong classification
    Try a new big_theta and see what our false positive rate is
  '''
  # run the shifted classifier on valid
  signed_valid = calc_classifiers(h_valid,big_theta)

  # calc error rate for validation set
  fpos_v, fneg_v = calc_error(signed_valid,truth)

  # calc shifted classifier on original training set
  
  signed_train = calc_classifiers(h, big_theta)

  # calc error rate for validation set
  fpos_t, fneg_t = calc_error(signed_train,label)
    
  fpos = max(fpos_v,fpos_t)
  fneg = min(fneg_v,fneg_t)

  return fpos, fneg


def find_big_theta(features, mu,feat_index, h, big_theta, curr_fpr,\
                   curr_fnr, weight, last_cases, l, new_weak,\
                   max_fpr, max_fnr , Npos, label ,valid , truth):
  ''' Input:
        Dynamic:
          mu         = How much to increment big_theta by
          big_theta  = what we are trying to find to decrease false neg
          h          = array of clasifiers
          curr_fpr   = Current false positive rate
          curr_fnr   = current false negative rate
          weight     = reweight everytime we get a new feature
          last_cases = Determines trajectory of big_theta is monotonic
                      want to avoid endless loops
          l          = Current layer or window
          new_weak   = determine if we add in new weak_classifier

        Static: 
          max_fpr   = max false pos rate
          max_fnr   = max false neg rate
          Npos      = Number of positive images in label
          label     = truth vector for training
          valid     = dictionary of validation set
          truth     = truth vector for validation

    4 Cases depending on result of fpos and fneg
  '''
  print "LOOP :", len(h)
  print "Current false pos: ", curr_fpr
  print "Current false neg: ", curr_fnr
  
  # h gets updated here, grows larger each weak classifier

  h, weight, halt = boost(h, features, label, weight, Npos)
  curr_fpr, curr_fnr = apply_validation(h,\
                                    valid,feat_index,label,truth,big_theta)
  # emergency stop condition
  if halt:
    print "Emergency stop"
    return tuple((h, curr_fpr, curr_fnr, big_theta, halt))

  # caes 1 -> move to next window:
  if (curr_fpr <= max_fpr) and (curr_fnr <= max_fnr): 
    print "Case 1"
    print "curr_fpr", curr_fpr
    print "curr_fnr,", curr_fnr
    print "big_theta", big_theta
    return tuple((h, curr_fpr, curr_fnr, big_theta,halt))

  # case 2 -> false pos is ok, false neg is bad, increase big_theta
  elif (curr_fpr <= max_fpr) and (curr_fnr > max_fnr):
    print "Case 2"
    big_theta += mu

    if (last_cases[1] == 3) and (last_cases[0] == 2):
      mu /= 2
      big_theta -= mu

    last_cases.append(2)
    last_cases.popleft()
    new_weak = False

    return find_big_theta(features,mu, feat_index, h,big_theta,curr_fpr,\
            curr_fnr,weight,last_cases, l, new_weak,
            max_fpr, max_fnr, Npos, label, valid, truth)

  # case 3 -> false pos too high, false neg ok
  elif (curr_fpr > max_fpr) and (curr_fnr <= max_fnr):
    print "Case 3"
    big_theta -= mu

    if (last_cases[1] == 2) and (last_cases[0] == 3):
      mu /= 2
      big_theta += mu

    last_cases.append(3)
    last_cases.popleft()
    new_weak = False

    return find_big_theta(features, mu,feat_index, h,big_theta,curr_fpr,\
            curr_fnr,weight,last_cases, l, new_weak,
            max_fpr, max_fnr, Npos, label, valid, truth)
  # case 4 
  else:
    print "Case 4"
    last_cases.append(4)
    last_cases.popleft()
    if len(h) > min( 10 * l, 200):
      while curr_fnr > max_fnr:
        curr_fpr, curr_fnr = apply_validation(h,\
                                    valid,feat_index,label,truth,big_theta)

      return tuple((h, curr_fpr, curr_fnr, big_theta,halt))
    else:
      print "Case 5"
      new_weak = True
      return find_big_theta(features, mu, feat_index, h,big_theta,curr_fpr,\
                      curr_fnr,weight,last_cases, l, new_weak,
                      max_fpr, max_fnr, Npos, label, valid, truth)

def get_strong_classifier(features,feat_index,valid,label, weight,Npos, l):
  ''' Input: 
        features = All features ran on training data, 300k x 3200
        feat_index = look up table for a feature given an index, 300k x 1
        valid = validation dictionary of images
      Output:
        strong classifier = tuple for each boosting round
          ith round <- alpha_i, feat_idx, parity, threshold
        bigTheta = constant value
      - ASSUME that # faces = # non_faces
  '''
  max_fpr = 0.3
  max_fnr = 0.01
  mu = 0.01
  truth = form_truth_valid(valid)
  last_cases = deque([0,0])
  # weak classifier is made up of alpha, theta = threshold, polar = parity
  # feature = look up via featureIdx
  # new_weight used to generate new featureIdx
  # false_pos = false positive used to determine stopping condition
  # false_neg = for determining big theta?

  curr_fpr = 1.0 # just so we can enter while loop
  curr_fnr = 1.0
  h = []
  big_theta = 0.0
  new_weak = True
  
  return find_big_theta(features, mu, feat_index, h, big_theta, curr_fpr,\
          curr_fnr, weight, last_cases, l, new_weak,\
          max_fpr, max_fnr , Npos, label ,valid , truth)
          
  
def train_cascade(features, feat_index, valid):
  '''
     Input:
       features = matrix of features from training data
       feat_index = look up table for features
       valid = Dictionary containing validation set

     Output:
      cascade = Array of strong classifiers 

     -Throw away all the images which have on zeros FROM THE TRAINING DATA
     - Validation data is only used to build Big Theta and the number of weak classifiers
     -Set our F_target to 0.3^10
     - Strong is an array of tuples, of length T
     - - each T has an alpha and an 800x1 array (800 being )
  '''
  fpr_target = math.pow(0.3,10)
  curr_fpr = 1.0 #F_i in the paper
  curr_fnr = 1.0

  num_img = features.shape[1]
  Npos = num_img/2 # assume #faces and #non_faces are the same
  # this doesn't change since we never remove any faces

  weight = np.zeros((num_img,1))
  weight[:,0] = 1.0/num_img

  label = np.zeros((num_img,1))
  label[:Npos] = 1.0
  # leave lables as 1 and 0 because getWeakClassifier needs 1 and 0 to be like that
 
  l = 1
  cascade = []
  # stop if we reach below our target f

  while  curr_fpr > fpr_target:
    print "Current window ", l
    if l >= 5: return cascade # solely unique for this training set
    
    strong = get_strong_classifier(features,feat_index,valid,label, weight,Npos, l)
    l += 1
    halt = strong[4]

    curr_fpr *= strong[1]
    features, label, weight = toss_zeros_training(strong,features,label)
    cascade.append(strong)

    if not 0 in label: return cascade
    if halt : return cascade
        

  return cascade

def toss_zeros_training(strong,features,label):
  '''Input:
        strong = (h, curr_fpr, curr_fnr, big_theta, halt)
        h = (alpha_t, h_t, featureIdx , polar, thresh)
        big_theta = constatnt that eliminates false negative rates
         - was chosen as the minimum value below 0, so subtract negative big_theta
        features = 300k x 3200 feature matrix, need to remove all non faces that
          were detected correctly

      Output:
      - calculate h(x) -> 800s
  '''
  h = strong[0]
  big_theta = strong[3]
  signed_train = calc_classifiers(h, big_theta)

  # apply labels, aka convert from value to 1 or 0
  # each our strong has a 0, and our training set has a 0
  remove_idx = []
  for i in range(len(signed_train)):
    if (label[i] == 0) and (signed_train[i] == 0) :
      # if we said face when non face, then remove that from traing data
      remove_idx.append(i)

  # Remove the correctly classified non-faces from training data
  features = np.delete(features,remove_idx, axis = 1)
  # remove the correctly classified non-faces from the truth label

  print "Removing Non Faces: ", len(remove_idx)

  label = np.delete(label, remove_idx, axis = 0)
  print "Number of images left ", len(label)

  weight = np.zeros((len(label),1))
  weight[:,0] = 1.0/len(weight)

  return features,label, weight



def main():
  '''Find the best classifier for the entire data set, do this
     -iteratively, re-weighting each step.
     -weight = 1/imgs = 1/4000
     -theta = subtract normal adaboost alpha * ht(x) - theta, sum all
     -beta = error/(1-error)
     - Stop boosting after your false positive rate drops below 30%
     - Set big Theta so no false positives (false when should be true)

  '''
  start = time.time()
  
  num_img = "2000"
  filename = "./data/features/training_"+num_img+".npy"
  features = np.load(filename)

  feat_index = np.load("./data/features/feat_index.npy")

  valid = read_validation(start_idx = 1600, Npos = 2000,Nneg = 2000)

  cascade = train_cascade(features, feat_index, valid)

  end = time.time()
  print "TIME ELAPSED :", (end - start)
  np.save("./data/cascades/cascade_"+num_img,cascade)
  

if __name__ == "__main__":
    main()