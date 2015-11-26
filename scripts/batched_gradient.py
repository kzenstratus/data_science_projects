import numpy as np
import sklearn
import pandas as pd
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt


PATH_ROOT = "/home/kzen/repos/data_analysis/kaggle/theano_lasagna"
DATA_PATH = PATH_ROOT + "/data/allstate/"

# def compute_cost(intercept,df,):



if __name__ == '__main__':

	df_train = pd.read_csv(DATA_PATH+"train.csv")
	# print df_train.columns
	df_train = df_train[["shopping_pt","cost"]]
	# df_train.set_index(column =="day")
	print df_train
	df_train.plot(kind = 'scatter', y = "shopping_pt", x = "cost")
	plt.savefig('test.png')
	alpha = 0.01 # learning rate
	m = len(df.index)
