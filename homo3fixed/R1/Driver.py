import os, sys
import pandas as pd

pp1, pp2 = float(sys.argv[1]), float(sys.argv[2])
v1, v2, v3 = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
v1b = v1

os.system("python Main.py round1 "+str(pp1)+" "+str(pp2)+" "+str(v1)+" "+str(v2)+" "+str(v3))

for i in xrange(2,11):
	if i == 2:
		df = pd.read_csv('round'+str(i-1)+".csv")
	else:
		df = pd.read_csv('round'+str(i-1)+"b.csv")
	p1, p2, p3 = df.iloc[df.shape[0]-1,8], df.iloc[df.shape[0]-1,9], df.iloc[df.shape[0]-1,10]
	pp1, pp2 = p1/float(p1+p2+p3), p2/float(p1+p2+p3)
	v1b += 0.001
	print "Bottom",i,pp1,pp2
	os.system("python Main.py round"+str(i)+"b "+str(pp1)+" "+str(pp2)+" "+str(v1b)+" "+str(v2)+" "+str(v3))

for i in xrange(2,11):
	if i == 2:
		df = pd.read_csv('round'+str(i-1)+".csv")
	else:
		df = pd.read_csv('round'+str(i-1)+"t.csv")
	p1, p2, p3 = df.iloc[df.shape[0]-1,5], df.iloc[df.shape[0]-1,6], df.iloc[df.shape[0]-1,7]
	pp1, pp2 = p1/float(p1+p2+p3), p2/float(p1+p2+p3)
	print "Top",i,pp1,pp2
	os.system("python Main.py round"+str(i)+"t "+str(pp1)+" "+str(pp2)+" "+str(v1)+" "+str(v2)+" "+str(v3))
