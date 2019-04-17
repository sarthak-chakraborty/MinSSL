import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import csv

# a = np.load("index_dpp_hypercubes.npy")

a=pd.read_csv("a.txt",delimiter=",",header=None)
a.iloc[0,0]=int(a.iloc[0,0].replace("[",""))
a.iloc[0,-1]=int(a.iloc[0,-1].replace("]",""))

b=np.array(a)[0][:150]


df_train = pd.read_csv("train.csv", header=None)
df_train_label = pd.read_csv("train_label.csv", header=None)
df_test = pd.read_csv("test.csv", header=None)
df_test_label = pd.read_csv("test_label.csv", header=None)

# print(df_train_label)

X_train = []
for i in range(len(df_train)):
    a = list(df_train.iloc[i])
    X_train.append(a)

y_train = list(df_train_label.iloc[0])

X_test = []
for i in range(len(df_test)):
    a = list(df_test.iloc[i])
    X_test.append(a)

y_test = list(df_test_label.iloc[0])


estimator = tree.DecisionTreeClassifier(random_state=0, max_depth=15)
estimator.fit(X_train, y_train)

predict = estimator.predict(X_train)

c=estimator.apply(X_train)

# def_hcube = []
# for i in range(len(X_train[0])):
#     # print(i)
#     a = np.min(zip(*X_train)[i])
#     b = np.max(zip(*X_train)[i])
#     c = [a,b]
#     def_hcube.append(c)


# def_hcube = np.array([[np.min(zip(*X_train)[i]),np.max(zip(*X_train)[i])] for i in range(len(X_train[0]))])
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
value = estimator.tree_.value

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
# node_cubes = [np.array(def_hcube) for i in range(n_nodes)]
hyper_cubes = []
labels = []
j=0

rep_points=[]
rep_labels=[]

stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    #print(node_cubes[node_id][feature[node_id]])
    node_depth[node_id] = parent_depth + 1
    #node_cubes[children_left[node_id]] = np.array(node_cubes[node_id])
    #node_cubes[children_right[node_id]] = np.array(node_cubes[node_id])
    #print("Feature of node: "+str(feature[node_id]))
    #print("Threshold of node: "+ str(threshold[node_id]))
    if (children_left[node_id] != children_right[node_id]): 
        # if(node_cubes[children_left[node_id]][feature[node_id]][1] > threshold[node_id]):
        #     node_cubes[children_left[node_id]][feature[node_id]][1] = threshold[node_id]
        #     #print("Modifying feature "+str(feature[node_id]))
        #     #print(node_cubes[children_left[node_id]][feature[node_id]])
        # if(node_cubes[children_right[node_id]][feature[node_id]][0] < threshold[node_id]):
        #     node_cubes[children_right[node_id]][feature[node_id]][0] = threshold[node_id]
        #     #print("Modifying feature "+str(feature[node_id]))
        #     #print(node_cubes[children_right[node_id]][feature[node_id]])        
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
    	if j in b:
    		for i in range(len(c)):
    			if c[i]==node_id:
    				rep_points.append(X_train[i])
    				rep_labels.append(y_train[i])
    	print(j,end='\r')
    	j=j+1
        #hyper_cubes.append(1)
        #labels.append(1)
        #print(node_cubes[node_id], np.argmax(value[node_id]), np.sum(value[node_id])
        #is_leaves[node_id] = True

with open("train1.csv","w") as f:
	for i in rep_points:
		writer = csv.writer(f)
		writer.writerow(i)

with open("train1labels.csv","w") as f:
	for i in rep_labels:
		writer = csv.writer(f)
		writer.writerow([i])

otherpoints=[]

for x in X_train:
	if x not in rep_points:
		otherpoints.append(x)

with open("otherpoints.csv","w") as f:
	for i in otherpoints:
		writer = csv.writer(f)
		writer.writerow(i)

print(len(rep_points))
print(len(rep_labels))
print(len(otherpoints))