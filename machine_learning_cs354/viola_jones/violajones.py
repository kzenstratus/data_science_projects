import numpy as np 
from numpy import cumsum, matlib
import glob
from scipy import misc
'''
Helper code provided by class - modified by Kevin Zen
'''

'''
Place this script in the folder containing
the faces/ and background/ directories or call 
getHaar with the filepath containing the faces/ and background/
directories
'''

'''
Harr-like feature extraction for one image

Input
filepath: string, the name of the directory containing the faces/ and background/
directories
row, col: ints, dimensions of the training images 
Npos: int, number of face images 
Nneg: int, number of background images

Output
features: ndarray, extracted Haar features
'''

def getHaar (filepath, row, col, Npos, Nneg):
	Nimg = Npos + Nneg
	Nfeatures = 295937 #change this number if you need to use more/less features 
	features = np.zeros((Nfeatures, Nimg))
	
	files = glob.glob(filepath+ "faces/*.jpg")
	for i in xrange (Npos):
		print "\nComputing Haar Face ",i
		imgGray = misc.imread (files[i], flatten=1) #array of floats, gray scale image
		if (i < Npos):
			# convert to integral image
			intImg = np.zeros((row+1,col+1))
			intImg [1:row+1,1:col+1] = np.cumsum(cumsum(imgGray,axis=0),axis=1)	
			# compute features
			features [:,i] = computeFeature(intImg,row,col,Nfeatures) 

			
	files = glob.glob(filepath+ "background/*.jpg")
	for i in xrange (Nneg):
			print "\nComputing Haar Background ",i

			imgGray = misc.imread (files[i], flatten=1) #array of floats, gray scale image
			if (i < Nneg):
				# convert to integral image
				intImg = np.zeros((row+1,col+1))
				intImg [1:row+1,1:col+1] = np.cumsum(cumsum(imgGray,axis=0),axis=1)
				#	print intImg.shape 
				#	import pdb pdb.set_trace()
				# compute features
				features[:,i+Npos] = computeFeature(intImg,row,col,Nfeatures)
			
	# print "feat ", features[1000,:]
	return features
		
		
'''
Given four corner points in the integral image 
calculate the sum of pixels inside the rectangular. 
'''
def sumRect(I, rect_four): 
	
	row_start = rect_four[0]
	col_start = rect_four[1] 
	width = rect_four[2]
	height = rect_four[3] 
	one = I[row_start-1, col_start-1]
	two = I[row_start-1, col_start+width-1]
	three = I[row_start+height-1, col_start-1]
	four = I[row_start+height-1, col_start+width-1]
	rectsum = four + one -(two + three)
	return rectsum 

'''
Computes the features. The cnt variable can be used to count the features. 
If you'd like to have less or more features for debugging purposes, set the 
Nfeatures =cnt in getHaar(). 
'''
def computeFeature(I, row, col, numFeatures): 
	feature = np.zeros(numFeatures)
	
	#extract horizontal features
	cnt = 0 # count the number of features 
	# This function calculates cnt=295937 features.
	
	window_h = 1; window_w=2 #window/feature size 
	for h in xrange(1,row/window_h+1): #extend the size of the rectangular feature
		for w in xrange(1,col/window_w+1):
			for i in xrange (1,row+1-h*window_h+1,4): #stride size=4
				for j in xrange(1,col+1-w*window_w+1,4): 
					rect1=np.array([i,j,w,h]) #4x1
					rect2=np.array([i,j+w,w,h])
					feature [cnt]=sumRect(I, rect2)- sumRect(I, rect1) 
					cnt=cnt+1

	window_h = 2; window_w=1 
	for h in xrange(1,row/window_h+1): 
		for w in xrange(1,col/window_w+1):
			for i in xrange (1,row+1-h*window_h+1,4):
				for j in xrange(1,col+1-w*window_w+1,4):
					rect1=np.array([i,j,w,h])
					rect2=np.array([i+h,j,w,h])
					feature[cnt]=sumRect(I, rect1)- sumRect(I, rect2)
					cnt=cnt+1
	
	return feature 
	
'''
Select best weak classifier for each feature over all images

Input 
features: ndarray, contains the features 
weight: ndarray, vector of weights
label: ndarray, vector of labels
Npos: number of face images 

Output: 
currentMin: min weighted error
theta: threshold
polarity: polarity
featureIdx:  best feature index
bestResult: classification result. Note that this is equivalent 
to h_t(x) from the original Viola-Jones paper and is used to determine the 
strong classifier decision value by cascading several weak classifiers

'''

def getWeakClassifier(features, weight, label, Npos):
	print "Starting Weak Classifier ..."
	Nfeatures, Nimgs = features.shape
	currentMin = np.inf
	tPos = np.matlib.repmat(np.sum(weight[:Npos,0]), Nimgs,1) 
	tNeg = np.matlib.repmat(np.sum(weight[Npos:Nimgs,0]), Nimgs,1)
	
	for i in xrange(Nfeatures):
		#get one feature for all images
		oneFeature = features[i,:]

		# sort feature to thresh for postive and negative
		sortedFeature = np.sort(oneFeature)
		sortedIdx = np.argsort(oneFeature)
	
		# sort weights and labels
		sortedWeight = weight[sortedIdx]
		sortedLabel = label[sortedIdx]
		
		# compute the weighted errors 
		sPos = cumsum(np.multiply(sortedWeight,sortedLabel)) 
		sNeg = cumsum(sortedWeight)- sPos
		
		sPos = sPos.reshape(sPos.shape[0],1)
		sNeg = sNeg.reshape(sNeg.shape[0],1)
		errPos = sPos + (tNeg -sNeg)
		errNeg = sNeg + (tPos -sPos)
	
		# choose the threshold with the smallest error
		allErrMin = np.minimum(errPos, errNeg) # pointwise min
		
		errMin = np.min(allErrMin)
		idxMin = np.argmin(allErrMin)
		
		# classification result under best threshold
		result = np.zeros((Nimgs,1))
		if (errPos [idxMin] <= errNeg[idxMin]):
			p = 1
			end = result.shape[0]
			result[idxMin+1:end] = 1
			result[sortedIdx] = np.copy(result)
			
		else:
			p = -1
			result[:idxMin+1] = 1
			result[sortedIdx] = np.copy(result)
			

		#get the parameters that minimize the classification error
		if (errMin < currentMin):
			currentMin = errMin
			if (idxMin==0):
				theta = sortedFeature[0] - 0.5
			elif (idxMin==Nfeatures-1):
				theta = sortedFeature[Nfeatures-1] + 0.5
			else:
				theta = (sortedFeature[idxMin]+sortedFeature[idxMin - 1])/2
			polarity = p
			featureIdx = i
			bestResult = result
	return currentMin, theta, polarity, featureIdx, bestResult


#Example usage: 

# def main():
#   row=64; col=64 #image size 
#   Npos = 20 #number of face images
#   Nneg = 20 #number of background images

#   features= getHaar("./data/smalldata/", row, col, Npos, Nneg)


#   weight = np.zeros((40,1))
#   weight[:,0] = 0.025
#  # weight [:,0] = [0.1250,0.1250,0.1250,0.1250,0.1667,0.1667,0.1667]
#   label = np.zeros((40,1))
#   label[:20] = 1
#   temp = getWeakClassifier (features, weight, label, Npos)
#   return temp,features



