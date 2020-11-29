import numpy as np 
import pandas as pd 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences 
import matplotlib.cm as cm 

################## KERAS DATA ###################################
# np_load_old = np.load
# np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k) 
# train_set,test_set = reuters.load_data(path="reuters.npz",num_words=20000,skip_top=0,index_from=3)
# np.load = np_load_old
# X_train,y_train = train_set[0],train_set[1]
# X_test,y_test = test_set[0],test_set[1]
# X_train_pad = pad_sequences(X_train,maxlen=300)
# X_test_pad = pad_sequences(X_test,maxlen=300)



################### ATTENTION WEIGHTS ############################
df_train = pd.read_csv("train.csv",header=None)
df_train_label = pd.read_csv("train_label.csv", header=None)
X_train=[]
for i in range(len(df_train)):
    a = df_train.iloc[i]
    X_train.append(a)
y_train = df_train_label.iloc[0]



################## SAMPLED POINTS #################################
# X = np.load("Rep_Samples.npy",allow_pickle=True)
# y = np.load("Rep_Samples_label.npy",allow_pickle=True)

# X_train,y_train=[],[]
# for i in X:
# 	for j in i:
# 		X_train.append(j)
# for i in y:
# 	for j in i:
# 		y_train.append(j)



tsne = TSNE(n_components=2, perplexity=40)
result = tsne.fit_transform(X_train, y_train)

np.save("./tSNE/TSNE_Result_attention.npy", result)
result = np.load("./tSNE/TSNE_Result_attention.npy")

color=cm.rainbow(np.linspace(0,1,len(y_train)))
plt.figure()
plt.scatter(result[:,0], result[:,1], s=2, c=np.array(color))
plt.savefig("./tSNE/tSNE_attention.png")
