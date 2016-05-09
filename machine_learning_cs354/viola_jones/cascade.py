# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:04:53 2016

@author: kevinzen
"""

import numpy as np 
from violajones import sumRect


def apply_weak_classifier(img_gray,thresh,parity,feature):
  ''' Given parity, threshold, and 
        img_file = 64x64 grayscale image
        feature = [rectagnle 1][rectangle[2]
  '''
  f = compute_feature(img_gray,feature)

  # somehow needs to be reversed from all papers
  if (parity * f) > (parity * thresh) : return 1
  else: return 0

def compute_feature(img_gray,feature,row = 64,col = 64):
  '''Given a 64x64 patch and feature,
       calculate pixel values under black and white parts of the
       image
  '''
  rect1 = feature[0]
  rect2 = feature[1]
  # imgGray = misc.imread (img_file, flatten=1) #array of floats, gray scale image
    # convert to integral image
  intImg = np.zeros((row+1,col+1))
  intImg [1:row+1,1:col+1] = np.cumsum(np.cumsum(img_gray,axis=0),axis=1)
  #	print intImg.shape 
  # compute features
  return sumRect(intImg, rect1)- sumRect(intImg, rect2)
