from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import export_graphviz
#import graphviz
import pandas as pd 

df_train = pd.read_csv("train1.csv", header=None)
df_train_label = pd.read_csv("train1labels.csv", header=None)
df_unknown_train=pd.read_csv("otherpoints.csv",header=None)
df_test = pd.read_csv("test.csv", header=None)
df_test_label = pd.read_csv("test_label.csv", header=None)


X_train=[]
for i in range(len(df_train)):
    a = list(df_train.iloc[i])
    X_train.append(a)
X_train2 = list(X_train)
X_train_unknown=[]
for i in range(len(df_unknown_train)):
    a = list(df_unknown_train.iloc[i])
    X_train_unknown.append(a)

y_train=[]
for i in range(len(df_train_label)):
	y_train.append(df_train_label.iloc[i,0])
print(y_train)

X_test = []
for i in range(len(df_test)):
    a = list(df_test.iloc[i])
    X_test.append(a)

y_test = list(df_test_label.iloc[0])
# print(y_test)


estimator = tree.DecisionTreeClassifier(random_state=0, max_depth=15)
estimator.fit(X_train, y_train)
# p=export_graphviz(estimator,out_file=None,class_names=[str(x) for x in set(y_train)])
# graph=graphviz.Source(p)
# graph.render("Train1")

# predict = estimator.predict(X_train_unknown)
predict = estimator.predict(X_train_unknown)
for i in range(len(X_train_unknown)):
	print(i, end='\r')
	X_train.append(X_train_unknown[i])
	y_train.append(predict[i])
	if(i%100==0):
		estimator.fit(X_train, y_train)
		predict = estimator.predict(X_train_unknown)
	

estimator = tree.DecisionTreeClassifier(random_state=0, max_depth=15)
estimator.fit(X_train, y_train)
# p=export_graphviz(estimator,out_file=None,class_names=[str(x) for x in set(y_train)])
# graph=graphviz.Source(p)
# graph.render("Train final")

count = 0
predict = estimator.predict(X_train)
for i in range(len(predict)):
    if(predict[i]==y_train[i]):
        count+=1
print("Train Accuracy: " +str(float(count)/len(predict)))

predict = estimator.predict(X_test)
count = 0
for i in range(len(predict)):
    if(predict[i]==y_test[i]):
        count+=1
print("Test Accuracy: " +str(float(count)/len(predict)))

