import os
import sys
import numpy as np
import pandas as pd

def slogit(x,s):
	return np.log((s*x)/(1-x))

np.random.seed(8)

mu, sigma = float(sys.argv[1]), float(sys.argv[2])

os.system("python Main.py round1 "+str(mu)+" "+str(sigma))

for i in xrange(2,11):
	if i == 2:
		df = pd.read_csv('round'+str(i-1)+".csv")
	else:
		df = pd.read_csv('round'+str(i-1)+"b.csv")
	mu, sigma = df.iloc[df.shape[0]-1,4], df.iloc[df.shape[0]-1,5]
	dirc = slogit(np.random.random(),4) > 0
	if dirc:
		mu = mu + 0.002
	else:
		mu = mu - 0.002
	print "Bottom",mu,sigma
	os.system("python Main.py round"+str(i)+"b "+str(mu)+" "+str(sigma))

for i in xrange(2,11):
	if i == 2:
		df = pd.read_csv('round'+str(i-1)+".csv")
	else:
		df = pd.read_csv('round'+str(i-1)+"t.csv")
	mu, sigma = df.iloc[df.shape[0]-1,2], df.iloc[df.shape[0]-1,3]
	dirc = slogit(np.random.random(),0.25) > 0
        if dirc:
                mu = mu + 0.002
        else:
                mu = mu - 0.002
	print "Top",mu,sigma
	os.system("python Main.py round"+str(i)+"t "+str(mu)+" "+str(sigma))

