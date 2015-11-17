import sqlite3
import datetime
import pandas as pd
import re
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
# matplotlib.style.use('ggplot')


conn = sqlite3.connect('database.sqlite')
c = conn.cursor()

a = c.execute("SELECT ExtractedDateSent From Emails")
l = a.fetchall()


def to_time(date):
	meridian = 0
	try:
		if(str(date['Meridian']) == "PM"):
			meridian = 12

		hour =  100*(float(date['Hour']) + meridian)
		minute = float(date['Minute'])
		# print datetime.datetime(hours= hour, minutes = minute)

		
		# return datetime.time(hour = hour, minute = minute)
		return hour+minute
	except:
		return None


if __name__ == '__main__':
	temp = []
	for x in l:
		temp.append(re.findall(r"[\w']+", x[0]))

	df = pd.DataFrame(temp)
	df = df[[0,1,2,3,4,5,6]]
	df.rename(columns={0:"Day Of Week", 1: "Month", 2:"Day", 3 :"Year", 4:"Hour", 5:"Minute", 6:"Meridian"}, inplace=True)
	df['time'] = df.apply(func = to_time ,axis = 1)
	series = df['time'].dropna()
	# df_final.to_list()
	blah = series.tolist()
	# plot = series.hist(alpha = 0.8)
	plt.hist(blah, alpha = 0.8, bins = [100,200,24000])	
	print series
	# plot.get_figure().savefig('foo.png')
	# df_final.plot(kind = "bar")
	plt.savefig('bar.png')