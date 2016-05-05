# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:16:19 2016

@author: kevinzen

"""
import math
import numpy as np
from violajones import getWeakClassifier
import os



def reduce_false_pos(strong,label):
  ''' Set your Big theta based on your last 
        classifier, since that classifier will have a low false negative weight
        Strong = storng classifier
        Label = true classification (first half face, second non face)
  '''
  
  for (alpha,classifier) in strong:
#    weighted = alpha*classifier
    for i in range(len(classifier)):
      if (classifier[i] == 0) and (label[i] == 1): # false negative
        classifier[i] = label[i] # Not sure if this is correct amount to
        # increase by, but it will at least eliminate false negatives
  return (alpha,classifier)
    

def get_strong_classifier(faces,features, images,label,weight):
  ''' Return your strong classifier (alpha,h_t(x))
      Run many weak classifiers until we get a false_positive
      rate below 30 percent.
      - ASSUME that # faces = # non_faces

  '''
  strong_classifier = [] # contains indices of weak features

  base = 0.0 # if strong_classifier is >= 1/2 base, return face

  count = 0 # only used for printing purposes
  false_pos = 1.0 # just so we can enter while loop

  while false_pos > 0.3:
    print "Count ",count
    count +=1
    min_error,theta,polar,featureIdx, h = getWeakClassifier(features,weight,label,faces)

    beta = min_error/(1.0-min_error)
    alpha = math.log(1.0/beta)

    strong_classifier.append(tuple((alpha,h)))

    base += alpha

    false_pos = 0.0
    false_neg = 0.0

    # test for false_pos and false_neg, update weights, and calc big Theta
    print "Starting re-weighting\n"
    for i in range(h.shape[0]):

      pred = h[i][0]
      actual = label[i][0]

      if pred == actual:
        weight[i][0] = weight[i][0]*beta
        continue
      elif pred > actual:
        false_pos +=1.0/images
      elif pred < actual:
        false_neg +=1.0/images
        
    print "Weights",weight
    print "False Pos = ", false_pos
    print "Strong classifier", strong_classifier
    print "Base =", base
    #normalize your weights
    weight = weight/(np.sum(weight))
    
  # Use the last classifier to create big Theta  
  strong_classifier = reduce_false_pos(strong_classifier,label)
  return strong_classifier, ((1.0/2)*base)



def main(filename = "feat_3.npy"):
  '''Find the best classifier for the entire data set, do this
     -iteratively, re-weighting each step.
     -weight = 1/imgs = 1/4000
     -theta = subtract normal adaboost alpha * ht(x) - theta, sum all
     -beta = error/(1-error)
     - Stop boosting after your false positive rate drops below 30%
     - Set big Theta so no false positives (false when should be true)

  '''
  feature_file = filename
  features = np.load(feature_file)

  images = features.shape[1]
  faces = images/2 # assume #faces and #non_faces are the same

  weight = np.zeros((images,1))
  weight[:,0] = 1.0/images 

  label = np.zeros((images,1))
  label[:faces] = 1.0

  # threshold is 1/2 sum alpha_t
  strong, threshold = get_strong_classifier(faces,features,images,label, weight)
  
  return strong, threshold


if __name__ == "__main__":
    main()