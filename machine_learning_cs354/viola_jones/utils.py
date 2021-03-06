# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:16:19 2016

@author: kevinzen

"""

from violajones import getHaar
import numpy as np
import glob
from scipy import misc
import skimage.io as io

def read_validation(start_idx = 1600, Npos = 2000,Nneg = 2000):
  ''' Input: 
        start_id = which file index to begin reading from, assuming 2k dataset
        Npos = number of faces
        Nneg = number of non-faces

      Output: 
        Dictionary -> keys = faces, nonfaces
  '''
  filepath = "./data/"
  files_pos = glob.glob(filepath+ "faces/*.jpg")
  files_neg = glob.glob(filepath+ "background/*.jpg")

  valid = {}

  for i in xrange(start_idx,Npos):
    gray_pos = misc.imread (files_pos[i], flatten=1) #array of floats, gray scale image
    valid.setdefault("pos", []).append(gray_pos)

  for i in xrange(start_idx,Nneg):
    gray_neg = misc.imread (files_neg[i], flatten=1) #array of floats, gray scale image
    valid.setdefault("neg", []).append(gray_neg)

  return valid



def form_truth_valid(valid):
  ''' Input:store
        valid = validation set as a dict
      Output: 
        array formed by valid set
  '''
  num_pos = len(valid["pos"])
  num_neg = len(valid["neg"])

  truth = np.zeros(num_pos+num_neg)
  truth[:num_pos] = 1.0
  return truth



def store_features(Npos = 2000,Nneg = 2000):
  ''' Save the features x imgs  onto hard drive
  If imgs = 2000, features ~ 300,000, save 300,000 x 2000 array
  Assume that the patches are 64x64
  Split training data into training and validation set
  Training: features * 0.8
  Validation : features* 0.2
  '''
  print "starting storage .."
  row = 64
  col = 64
  root = "./data/"
  
  train_pos = int(Npos*0.8)
  train_neg = int(Nneg*0.8)
  
  training = getHaar(root,row,col,train_pos,train_neg)
  np.save(root+"features/training_"+repr(Npos),training)
  
  return training

def get_all_features(row = 64,col = 64,numFeatures = 295936):
  ''' Return all the features from violajones
    cnt=295936 features, 
    returns cnt x 1 array
    One feature is a tuple of two rectangles
  '''
  feature = np.zeros((numFeatures,2,4))
	
  #extract all features
  cnt = 0 # count the number of features 
  # This function calculates cnt=295937 features.
  # position is at top left corner
  window_h = 1; window_w=2 #window/feature size 
  for h in xrange(1,row/window_h+1): #extend the size of the rectangular feature
    for w in xrange(1,col/window_w+1):
      for i in xrange (1,row+1-h*window_h+1,4): #stride size=4
        for j in xrange(1,col+1-w*window_w+1,4): 
          rect1=np.array([i,j,w,h]) #4x1
          rect2=np.array([i,j+w,w,h])
          feature [cnt]=(rect2,rect1)
          cnt=cnt+1

  window_h = 2; window_w=1 
  for h in xrange(1,row/window_h+1): 
    for w in xrange(1,col/window_w+1):
      for i in xrange (1,row+1-h*window_h+1,4):
        for j in xrange(1,col+1-w*window_w+1,4):
          rect1=np.array([i,j,w,h])
          rect2=np.array([i+h,j,w,h])
          feature[cnt]=(rect1,rect2)
          cnt=cnt+1
          
  np.save("./data/features/feat_index",feature)
  return feature

