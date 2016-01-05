import numpy as np
import sklearn
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt


PATH_ROOT = "/Users/kevinzen/repos/data_science_projects"
DATA_PATH = PATH_ROOT + "/data/allstate/"

# use least squared cost function
    # h(t) = t0+t1*x
# Cost function
    # J(t0,t1) = 1/2n SUM( h_t(x) - y)^2
# Min Cost Function: Take Partial w/ both parameters
    # d/d(t0)
    # 1/n SUM (h_t0(x) - y)

    # d/d(t1)
    # 1/n SUM ( h_t1(x) - y)x


def gradient_descent(alpha, x, y, ep = 0.0001, max_iter = 10000):
    
    counter = 0
    converged = False

    while not converged:        
        part0 = 1.0/n * sum([(t0 + t1*x[i] - y[i]) for i range(n)])
        part1 = 1.0/n * sum([(t0 + t1*x[i] - y[i])* x[i] for i range(n)])
    
        # update your theta each iteration
        t0 = t0 - alpha * part0
        t1 = t1 - alpha * part1    
    	

        # mean squared
        # counter += 1
        # if(counter == max_iter):
        # 	converged = True

    return t0,t1



def normalize(x, avg,std):
	return (x-avg)/std


def getAverage(df,col):
	return df[col].mean()

if __name__ == '__main__':

    df_train = pd.read_csv(DATA_PATH+"train.csv")
    df_test = pd.read_csv(DATA_PATH+"test_v2.csv")
     
     
     # print df_train.columns
    df_train = df_train[["shopping_pt","cost"]]
    shoppingAvg = df_train.shopping_pt.mean()
    costAvg = df_train.cost.mean()
    shoppingStd = df_train.shopping_pt.std()
    costStd = df_train.cost.std()

    df_train]"shopping_pt"] = df_train.shopping_pt.apply(func = normalize, args = (shoppingAvg,shoppingStd))
	df_train["shopping_pt"] = df_train.shopping_pt/df_train.shopping_pt.mean()
	# df_train.set_index(column =="day")
	print df_train
	df_train.plot(kind = 'scatter', y = "shopping_pt", x = "cost")
	plt.savefig('test.png')
	alpha = 0.01 # learning rate
	m = len(df.index)
