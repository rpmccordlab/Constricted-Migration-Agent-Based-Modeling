import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
plt.rcParams.update({'font.size': 20})

np.random.seed(8)

for base in ['R2']:
	beff = []
	teff = []

	beff.append([0.005,0.02])
	teff.append([0.005,0.02])

	rounds = [2,3,4,5,6,7,8,9,10]

	for i in xrange(len(rounds)):
		temp = pd.read_csv(base+'/round'+str(rounds[i])+'b.csv').iloc[[-1],[4,5]].values.flatten()
		beff.append(temp)
		temp = pd.read_csv(base+'/round'+str(rounds[i])+'t.csv').iloc[[-1],[2,3]].values.flatten()
		teff.append(temp)

np.savetxt('R2MeanSDB.txt',np.array(beff))
np.savetxt('R2MeanSDT.txt',np.array(teff))
