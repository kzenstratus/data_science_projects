import numpy as np
import sklearn
import pandas as pd
from scipy import stats


PATH_ROOT = "/home/kzen/repos/data_analysis/kaggle/theano_lasagna"
DATA_PATH = PATH_ROOT + "/data/allstate/"

if __name__ == '__main__':

	df_train = pd.read_csv(DATA_PATH+"train.csv")
	df_train = df_train[["day","group_size"]]
