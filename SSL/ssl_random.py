from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import export_graphviz
# from sklearn.model_selec import train_test_split
#import graphviz
import pandas as pd 

df_train = pd.read_csv("train.csv", header=None)
df_train_label = pd.read_csv("train_label.csv", header=None).T

df_test = pd.read_csv("test.csv", header=None)
df_test_label = pd.read_csv("test_label.csv", header=None)


train_unknown, train, label_unknown, label = train_test_split(df_train, df_train_label)

X_train=[]
for i in range(len(train)):
    a = list(train.iloc[i])
    X_train.append(a)

X_train2 = list(X_train)
X_train_unknown=[]
for i in range(len(train_unknown)):
    a = list(train_unknown.iloc[i])
    X_train_unknown.append(a)

y_train=[]
for i in range(len(label)):
	y_train.append(label.iloc[i,0])
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

