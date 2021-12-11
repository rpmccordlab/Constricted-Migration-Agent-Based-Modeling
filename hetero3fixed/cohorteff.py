import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
from pylab import rcParams
rcParams['figure.figsize'] = 8, 6
plt.rcParams.update({'font.size': 20})

bdata = []
tdata = []

for base,style in zip(['R1','R2','R3'],['-','--','-.']):
	beff = []
	teff = []

	round1 = pd.read_csv(base+'/round1.csv').iloc[[-1],[0,1]].values.flatten()

	beff.append(float(round1[1])/np.sum(round1))
	teff.append(float(round1[1])/np.sum(round1))

	rounds = [2,3,4,5,6,7,8,9,10]

	for i in xrange(len(rounds)):
		temp = pd.read_csv(base+'/round'+str(rounds[i])+'b.csv').iloc[[-1],[0,1]].values.flatten()
		beff.append(float(temp[1])/np.sum(temp))
		temp = pd.read_csv(base+'/round'+str(rounds[i])+'t.csv').iloc[[-1],[0,1]].values.flatten()
		teff.append(float(temp[1])/np.sum(temp))

	bdata.append(beff)
	tdata.append(teff)

plt.errorbar(xrange(len(beff)),np.mean(bdata,axis=0),yerr=np.std(bdata,axis=0),linewidth=3,marker='o',color='red',label='Bottom')
plt.errorbar(xrange(len(beff)),np.mean(tdata,axis=0),yerr=np.std(tdata,axis=0),linewidth=3,marker='x',color='black',label='Top')
plt.xticks(xrange(len(beff)),[1]+rounds)
plt.xlabel("Round")
plt.ylim([0.0,1.0])
plt.ylabel('Migration Efficiency')
plt.legend()
plt.savefig('ABMCohortEff.eps',bbox_inches='tight', pad_inches=0,dpi=300)


