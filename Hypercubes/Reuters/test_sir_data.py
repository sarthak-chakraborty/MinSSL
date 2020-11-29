import numpy as np
import pandas as pd
import statistics

X = np.loadtxt(open("./test-data/test.txt", "rb"), delimiter=",", skiprows=0)
y = X[..., -1]
X = X[..., :-1]

CLASSES=10

print(len(X))
print(len(X[0]))
print(len(y))

hypercubes = list(pd.read_csv("Hypercubes_test.csv",header=None).as_matrix())
print(hypercubes[0])

correct=0

for i in range(len(X)):
	print(i, end='\r')
	a=hypercubes
	b=[]
	clas = []
	for j in range(len(X[i])):
		a = [a[k] for k in range(len(a)) if((X[i][j] >= a[k][2*j]) and (X[i][j] < a[k][2*j+1]))]

	for k in range(len(a)):
		xx = []
		for c in range(CLASSES):
			xx.append(a[k][-CLASSES-2+c])
		b.append(xx)

	for j in b:
		clas.append(np.argmax(j))

	if(len(clas) > 0):
		final_class = np.max(clas)
	else:
		final_class=statistics.mode(y)

	if(final_class==y[i]):
		correct+=1

print(float(correct)/len(y))