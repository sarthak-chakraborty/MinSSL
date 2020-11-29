import numpy as np 
import pandas as pd 


pointH = np.load("PointsH.npy", allow_pickle=True)
pointV = np.load("PointsV.npy", allow_pickle=True)
pointH_index = np.load("PointsH_index.npy", allow_pickle=True)
pointV_index = np.load("PointsV_index.npy", allow_pickle=True)



# common_point_H = {}
# for i in range(len(pointH_index)):
# 	A = [item for sublist in pointH_index[i] for item in sublist]
# 	A = set(A)
# 	for j in range(i+1, len(pointH_index)):
# 		B = [item for sublist in pointH_index[j] for item in sublist]
# 		B = set(B)
# 		C = A.intersection(B)
# 		if(len(C) > 0):
# 			for k in range(len(C)):
# 				if common_point_H.has_key(C[k]):
# 					common_point_H[C[k]].add(i)
# 					common_point_H[C[k]].add(j)
# 				else:
# 					common_point_H[C[k]]={i,j}


# common_point_V = {}
# for i in range(len(pointV_index)):
# 	A = [item for sublist in pointV_index[i] for item in sublist]
# 	A = set(A)
# 	for j in range(i+1, len(pointV_index)):
# 		B = [item for sublist in pointV_index[i] for item in sublist]
# 		B = set(B)
# 		C = A.intersection(B)
# 		if(len(C) > 0):
# 			for k in range(len(C)):
# 				if common_point_V.has_key(C[k]):
# 					common_point_V[C[k]].add(i)
# 					common_point_V[C[k]].add(j)
# 				else:
# 					common_point_V[C[k]]={i,j}


# print(common_point_H)
# print(common_point_V)


data = []
data_index = []
data_label = []

for k in range(len(pointH)):
	l=[]
	x=[]
	ind=[]
	for i in range(len(pointH[k])):
		for j in range(len(pointH[k][i])):
			l.append(pointH[k][i][j])
			x.append(i)
			ind.append(pointH_index[k][i][j])
	data.append(l)
	data_index.append(ind)
	data_label.append(x)

for k in range(len(pointV)):
	l=[]
	x=[]
	ind=[]
	for i in range(len(pointV[k])):
		for j in range(len(pointV[k][i])):
			l.append(pointV[k][i][j])
			x.append(i)
			ind.append(pointV_index[k][i][j])
	data.append(l)
	data_index.append(ind)
	data_label.append(x)


n_hypercubes = len(data)




def gen_K_matrix(data):
	data = np.array(data)

	for i in range(len(data)):
		data[i] = data[i] / (np.linalg.norm(data[i]))

	K=np.dot(data, data.T)
	# L = np.matmul(K, np.linalg.inv(np.eye(len(data))-K))

	return K


def get_eigen(K):
	return np.linalg.eigh(K)


def normalize_dpp(w,k):
	p = [[np.sum(w[:j+1]**i) for i in range(1,k+1)] for j in range(len(w))]

	e = []
	for i in range(len(w)):
		e_temp=[]
		for j in range(k):
			s=0.0
			if(j==0):
				s=p[i][j]
			else:
				for x in range(j):
					s+=((-1)**x)*e_temp[j-x-1]*p[i][x]
				s+=((-1)**j)*p[i][j]
				s/=(j+1)
			e_temp.append(s)
		e.append(e_temp)

	return e


def orth_subspace(V, index):
	a = [v[index] for v in V]
	alpha = [-float(a[i])/(a[-1]+np.finfo(float).eps) for i in range(len(a)-1)]
	V_orth = [ V[i]+alpha[i]*V[-1] for i in range(len(V)-1)]
	return V_orth



def sample(w, v, k, data, data_index, label):
	e = normalize_dpp(w,k)
	J=[]
	k_act=k
	c=0
	while len(J)<k_act:
		if(c==5):
			break
		for i in range(len(w)):
			u = np.random.uniform(0,1)
			if(u < (w[-i-1]*e[len(w)-i-2][k-2])/e[len(w)-i-1][k-1]):
				J.append(len(w)-i-1)
				k -= 1
				if(k==0):
					break
		c+=1

	V = [v[i] for i in J]
	P = [np.sum([v[i]**2 for v in V])/float(len(V)) for i in range(len(data))]
	index_chosen=[]
	Y=[]
	Y_index=[]
	Y_label=[]

	while(len(V)>0):
		index = np.random.choice(np.arange(len(data)), 1, P)[0]
		if(index in index_chosen):
			continue
		index_chosen.append(index)
		Y.append(data[index])
		Y_index.append(data_index[index])
		Y_label.append(label[index])
		V = orth_subspace(V, index)
		if(len(V)==0):
			break
		V,_ = np.linalg.qr(np.transpose(V))
		V=V.T

	Y_prime = []
	Y_prime_index =[]
	for i in range(len(data)):
		if(i not in index_chosen):
			Y_prime.append(data[i])
			Y_prime_index.append(data_index[i])

	return Y, Y_index, Y_label, Y_prime, Y_prime_index



def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


samples = []
samples_index =[]
samples_label=[]
samples_prime=[]
samples_prime_index = []
for i in range(n_hypercubes):
	K = gen_K_matrix(data[i])
	# print(check_symmetric(L))
	w,v = get_eigen(K)
	k = int(0.2*len(data[i]))
	print(i,k)
	A,B,C,D,E = sample(w,v,k,data[i], data_index[i], data_label[i])
	samples.append(A)
	samples_index.append(B)
	samples_label.append(C)
	samples_prime.append(D)
	samples_prime_index.append(E)



np.save("Rep_Samples.npy", samples)
np.save("Rep_Samples_index.npy", samples_index)
np.save("Rep_Samples_label.npy", samples_label)
np.save("Other_Points.npy",samples_prime)
np.save("Other_Points_index.npy",samples_prime_index)
print([len(samples[i]) for i in range(len(samples))])

print(np.sum([len(samples[i]) for i in range(len(samples))]))
print(np.sum([len(samples_prime[i]) for i in range(len(samples_prime))]))
