from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import export_graphviz
import graphviz
import pandas as pd 
dim = 13



df_train = pd.read_csv("train.csv")
df_train_label = pd.read_csv("train_label.csv")
df_test = pd.read_csv("test.csv")
df_test_label = pd.read_csv("test_label.csv")

print(len(list(df_train_label)))
print(df_test)






Data = datasets.load_wine()
X = Data.data
y = Data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

estimator = tree.DecisionTreeClassifier(random_state=0)
estimator.fit(X_train, y_train)
p=export_graphviz(estimator,out_file=None,class_names=[str(x) for x in set(y_train)])
graph=graphviz.Source(p)
graph.render("Entropy");
predict = estimator.predict(X_test)
count = 0
for i in range(len(predict)):
    if(predict[i]==y_test[i]):
        count+=1
print("Test Accuracy: " +str(float(count)/len(predict)))


def_hcube = np.array([[np.min(X_train[:,i]),np.max(X_train[:,i])] for i in range(len(X[0]))])
n_nodes = estimator.tree_.node_count
children_left = estimator.tree_.children_left
children_right = estimator.tree_.children_right
feature = estimator.tree_.feature
threshold = estimator.tree_.threshold
value = estimator.tree_.value

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
node_cubes = [np.array(def_hcube) for i in range(n_nodes)]
hyper_cubes = []
labels = []

stack = [(0, -1)]  # seed is the root node id and its parent depth
while len(stack) > 0:
    node_id, parent_depth = stack.pop()
    #print(node_cubes[node_id][feature[node_id]])
    node_depth[node_id] = parent_depth + 1
    node_cubes[children_left[node_id]] = np.array(node_cubes[node_id])
    node_cubes[children_right[node_id]] = np.array(node_cubes[node_id])
    #print("Feature of node: "+str(feature[node_id]))
    #print("Threshold of node: "+ str(threshold[node_id]))
    if (children_left[node_id] != children_right[node_id]): 
        if(node_cubes[children_left[node_id]][feature[node_id]][1] > threshold[node_id]):
            node_cubes[children_left[node_id]][feature[node_id]][1] = threshold[node_id]
            #print("Modifying feature "+str(feature[node_id]))
            #print(node_cubes[children_left[node_id]][feature[node_id]])
        if(node_cubes[children_right[node_id]][feature[node_id]][0] < threshold[node_id]):
            node_cubes[children_right[node_id]][feature[node_id]][0] = threshold[node_id]
            #print("Modifying feature "+str(feature[node_id]))
            #print(node_cubes[children_right[node_id]][feature[node_id]])        
        stack.append((children_left[node_id], parent_depth + 1))
        stack.append((children_right[node_id], parent_depth + 1))
    else:
        hyper_cubes.append(node_cubes[node_id])
        labels.append(np.argmax(value[node_id]))
        print(node_cubes[node_id], np.argmax(value[node_id]), np.sum(value[node_id]))
        is_leaves[node_id] = True




