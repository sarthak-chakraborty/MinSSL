from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import export_graphviz
import graphviz
import pandas as pd 
import xgboost as xgb
import matplotlib.pyplot as plt


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


data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)

model = xgb.XGBClassifier(max_depth=50, objective='reg:logistic')
model.fit(np.array(X_train), np.array(y_train))
predict = model.predict(np.array(X_train))

count = 0
for i in range(len(predict)):
    if(predict[i]==y_train[i]):
        count+=1
print("Train Accuracy: " +str(float(count)/len(predict)))

predict = model.predict(np.array(X_test))
count = 0
for i in range(len(predict)):
    if(predict[i]==y_test[i]):
        count+=1
print("Test Accuracy: " +str(float(count)/len(predict)))

xgb.plot_tree(model,num_trees=0)
plt.figure()
plt.savefig("Tree_xg.png")

# xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)