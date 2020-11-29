
# coding: utf-8

# In[60]:


from sklearn import datasets
import numpy as np
import pandas as pd


# In[61]:

# X, y = datasets.load_iris(return_X_y = True)

X = np.loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=0)
y = np.loadtxt(open("train_label.csv", "rb"), delimiter=",", skiprows=0)
X_index = [i for i in range(len(X))]

# X = np.loadtxt(open("./test-data/train.txt", "rb"), delimiter=",", skiprows=0)
# y = X[..., -1]
# X = X[..., :-1]
# print(X)
# print(y)
print(len(X))
print(len(X[0]))
print(len(y))
print(max(y))
# exit()
FEATURES = len(X[0])
CLASSES = int(max(y))+1
CUTOFF = 50



def computeMinDistance(X, y):
    minima = 99999
    for i in range(len(X)):
        print(i, end="\r")
        for j in range(i):
            if(y[i] != y[j]):
                if(np.linalg.norm(X[i]-X[j]) < minima):
                    minima = np.linalg.norm(X[i] - X[j])
    return minima
    


def computeH(Z):
    minima = 999999
    for c in range(CLASSES):
        for cdash in range(CLASSES):
            if(c != cdash):
                fval = 0
                if((len(Z[c]) == 0) & (len(Z[cdash]) == 0)):
                    fval = 0
                else:
                    fval = float(np.abs(len(Z[c]) - len(Z[cdash])))/(len(Z[c]) + len(Z[cdash]))
                if(minima > fval):
                    minima = fval
    return minima
                
                
        


# l_star = computeMinDistance(X,y)
l_star=0.013
print(l_star)
print("L computation done")
l = l_star/1.5
# l=0.0001
initcube = [[np.min(X[:,i]),np.max(X[:,i])] for i in range(FEATURES)]
initZ = [[X[i] for i in range(len(X)) if y[i] == c ] for c in range(CLASSES)]
initZ_index = [[X_index[i] for i in range(len(X_index)) if y[i] == c ] for c in range(CLASSES)]

HI = []
ZI = []
HV = []
ZV = []
HH = []
ZH = []
HE = []
pointH, pointV = [],[]
pointH_index, pointV_index = [],[]
Zlist = []
Zlist_index =[]
HI.append(initcube)
Zlist.append(initZ)
Zlist_index.append(initZ_index)
iii=0
while(len(HI) > 0):
    print(iii, end='\r')
    iii+=1
    cube = HI.pop()
    Z = Zlist.pop()
    Z_index = Zlist_index.pop()
    calH = []
    for i in range(FEATURES):
        if(cube[i][1] - cube[i][0] < 2*l):
            calH.append(-1000000)
            continue
        cubea = [list(cube[n]) for n in range(FEATURES)]
        cubea[i][1] = (cubea[i][1] - cubea[i][0])/2 + cubea[i][0]
        Za = []
        for c in range(CLASSES):
            Ztemp = []
            for j in range(len(Z[c])):
                if(Z[c][j][i] <= cubea[i][1]):
                    Ztemp.append(Z[c][j])
            Za.append(Ztemp)
        cubeb = [list(cube[n]) for n in range(FEATURES)]
        cubeb[i][0] = cubeb[i][1] - (cubeb[i][1] - cubeb[i][0])/2
        Zb = []
        for c in range(CLASSES):
            Ztemp = []
            for j in range(len(Z[c])):
                if(Z[c][j][i] > cubeb[i][0]):
                    Ztemp.append(Z[c][j])
            Zb.append(Ztemp)
        if(np.sum([len(Za[i]) for i in range(CLASSES)]) < CUTOFF or np.sum([len(Zb[i]) for i in range(CLASSES)]) < CUTOFF):
            calH.append(-1000000)
            continue
        ha = computeH(Za)
        hb = computeH(Zb)
        
        calH.append(max(ha, hb))

    # print(calH)
    alpha = np.argmax(calH)
    # print(alpha)
    if(calH[alpha] == -1000000):
        HV.append(cube)
        ZV.append([len(Z[i]) for i in range(CLASSES)])
        pointV.append(Z)
        pointV_index.append(Z_index)
        continue

    cubea = [list(cube[n]) for n in range(FEATURES)]
    cubea[alpha][1] = (cubea[alpha][1] - cubea[alpha][0])/2 + cubea[alpha][0]
    Za = []
    Za_index = []
    for c in range(CLASSES):
        Ztemp = []
        Ztemp_index=[]
        for j in range(len(Z[c])):
            if(Z[c][j][alpha] <= cubea[alpha][1]):
                Ztemp.append(Z[c][j])
                Ztemp_index.append(Z_index[c][j])
        Za.append(Ztemp)
        Za_index.append(Ztemp_index)

    flag = 0
    for c in range(CLASSES):
        if(len(Za[c]) > 0):
            flag += 1
    if(flag == 0):
        HE.append(cubea)
    elif(flag == 1):
        HH.append(cubea)
        ZH.append([len(Za[i]) for i in range(CLASSES)])
        pointH.append(Za)
        pointH_index.append(Za_index)
    elif((cubea[alpha][1] - cubea[alpha][0] < 2*l) and (np.sum([len(Za[i]) for i in range(CLASSES)]) < CUTOFF)):
        HV.append(cubea)
        ZV.append([len(Za[i]) for i in range(CLASSES)])
        pointV.append(Za)
        pointV_index.append(Za_index)
    else:
        HI.append(cubea)
        Zlist.append(Za)
        Zlist_index.append(Za_index)

    cubeb = [list(cube[n]) for n in range(FEATURES)]
    cubeb[alpha][0] = cubeb[alpha][1] - (cubeb[alpha][1] - cubeb[alpha][0])/2
    Zb = []
    Zb_index=[]
    for c in range(CLASSES):
        Ztemp = []
        Ztemp_index=[]
        for j in range(len(Z[c])):
            if(Z[c][j][alpha] > cubeb[alpha][0]):
                Ztemp.append(Z[c][j])
                Ztemp_index.append(Z_index[c][j])
        Zb.append(Ztemp)
        Zb_index.append(Ztemp_index)
    flag = 0
    for c in range(CLASSES):
        if(len(Zb[c]) > 0):
            flag += 1
    if(flag == 0):
        HE.append(cubeb)
    elif(flag == 1):
        HH.append(cubeb)
        ZH.append([len(Zb[i]) for i in range(CLASSES)])
        pointH.append(Zb)
        pointH_index.append(Zb_index)
    elif((cubeb[alpha][1] - cubeb[alpha][0] < 2*l) and (np.sum([len(Zb[i]) for i in range(CLASSES)]) < CUTOFF)):
        HV.append(cubeb)
        ZV.append([len(Zb[i]) for i in range(CLASSES)])
        pointV.append(Zb)
        pointV_index.append(Zb_index)
    else:
        HI.append(cubeb)
        Zlist.append(Zb)
        Zlist_index.append(Zb_index)
    


print(len(pointH))
print(len(HH))
print(len(pointV))
print(len(HV))
np.save("PointsH.npy",pointH)
np.save("PointsV.npy",pointV)
np.save("PointsH_index.npy",pointH_index)
np.save("PointsV_index.npy",pointV_index)



with open('Hypercubes.csv','w') as f:
    for i in range(len(HH)):
        for dim in HH[i]:
            f.write(str(dim[0]) + "," + str(dim[1]) + ",")
        for c in range(len(ZH[i])):
            f.write(str(ZH[i][c])+",")
        f.write(str(np.sum(ZH[i])) + ",\n")

    for i in range(len(HV)):
        for dim in HV[i]:
            f.write(str(dim[0]) + "," + str(dim[1]) + ",")
        for c in range(len(ZV[i])):
            f.write(str(ZV[i][c])+",")
        f.write(str(np.sum(ZV[i])) + ",\n")


with open("Hypercubes_list.csv", "w") as f:
    for i in range(len(ZV)):
        f.write(str(i)+",")
        for c in range(len(ZV[i])):
            f.write(str(ZV[i][c])+",")
        f.write(",\n")


    


