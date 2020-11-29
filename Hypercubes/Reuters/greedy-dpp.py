from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
from scipy import linalg
import sys


dim = 300
max_hypercubes = 250
FACTOR = 50

# hypercubes = pd.read_csv("HyperCool.csv", index_col = 0,header=0)
# hypercubes_mean = hypercubes.copy()
# for i in range(len(hypercubes)):
# 	print(i, end='\r')
# 	for j in range(dim):
# 		hypercubes_mean.iloc[i, j] = np.mean(eval(hypercubes.iloc[i, j]))

# np.save("Hypercubes", hypercubes_mean)

hypercubes_mean = np.load("Hypercubes.npy", allow_pickle=True)
# hypercubes_mean = pd.DataFrame(hypercubes_mean)


K =np.array([[0.1]*len(hypercubes_mean) for i in range(len(hypercubes_mean))])

def distance(i,j):
	su = 0
	for ite in range(dim):
		su += (hypercubes_mean[i, ite]- hypercubes_mean[j, ite])**2
	if(int(hypercubes_mean[i,dim])!=int(hypercubes_mean[j,dim])):
		return np.sqrt(su)/FACTOR
	return np.sqrt(su)


for i in range(len(K)):
	print(i, end='\r')
	for j in range(i+1):
		if(i==j):
			K[i,j] = 1
		else:
			temp = 1.0/(np.finfo(np.float32).eps+ (distance(i,j)**2 * hypercubes_mean[i,-1] * hypercubes_mean[j,-1]))
			K[i,j] = temp
			K[j,i] = temp


w,v = np.linalg.eig(K)
print(np.min(w))
exit()
np.save("K.npy", K)
# K = np.load("K.npy")

maxdpplist = []
maxdet = -sys.maxsize
for iter1 in range(1):
	dpplist = [int(np.random.random()*len(K))]
	i = 1
	for iteri in range(1, max_hypercubes):
		# print(iteri, end='\r')
		maxi = -sys.maxsize
		itermax = -1
		for iterj in range(len(K)):
			if(iterj not in dpplist):
				dpplistnew = dpplist + [iterj]
				det = linalg.det([[K[i,j] for i in dpplistnew] for j in dpplistnew])
				if((np.isinf(det))|(np.isnan(det))):
					continue
				if(maxi<det):
					maxi = det
					itermax = iterj
		dpplist.append(itermax)
	deter = linalg.det([[K[i,j] for i in dpplist] for j in dpplist])
	if(maxdet<deter):
		maxdet = deter
		maxdpplist = dpplist

np.save("index_dpp_hypercubes.npy",maxdpplist)
print(maxdpplist)

