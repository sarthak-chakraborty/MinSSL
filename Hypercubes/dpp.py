from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 

dim = 13
max_hypercubes = 3
FACTOR = 50
hypercubes = pd.read_csv("HyperCool.csv", index_col = 0,header=0)

K =np.array([[0.1]*len(hypercubes) for i in range(len(hypercubes))])

def distance(i,j):
	su = 0
	for ite in range(dim):
		ilist = eval(hypercubes.iloc[i, ite])
		jlist = eval(hypercubes.iloc[j, ite])
		su+= (np.mean(ilist)-np.mean(jlist))**2
	if(int(hypercubes.iloc[i,dim])!=int(hypercubes.iloc[j,dim])):
		#print("here")
		su/=(FACTOR*FACTOR)
	return np.sqrt(su)


for i in range(len(K)):
	for j in range(i+1):
		if(i==j):
			K[i,j] = 0
		else:
			K[i,j] = 1.0/(np.finfo(np.float32).eps+ distance(i,j))
			K[j,i] = 1.0/(np.finfo(np.float32).eps+ distance(i,j))


#print(K)
maxdpplist = []
maxdet = -1000000
for iter1 in range(len(K)):
	dpplist = [iter1]
	i = 1
	for iteri in range(1, max_hypercubes):
		maxi = -1000000000
		itermax = -1
		#print(iteri, end = '\r')
		for iterj in range(len(K)):
			#print(iterj)
			if(iterj not in dpplist):
				dpplistnew = dpplist + [iterj]
				if(maxi<np.linalg.det([[K[i,j] for i in dpplistnew] for j in dpplistnew])):
					maxi = np.linalg.det([[K[i,j] for i in dpplistnew] for j in dpplistnew])
					itermax = iterj
		dpplist.append(itermax)
	if(maxdet<np.linalg.det([[K[i,j] for i in dpplist] for j in dpplist])):
		maxdet = np.linalg.det([[K[i,j] for i in dpplist] for j in dpplist])
		maxdpplist = dpplist
print(maxdpplist)
