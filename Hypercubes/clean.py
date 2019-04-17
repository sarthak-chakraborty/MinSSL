import pandas as pd 
dim = 300

a = pd.read_csv("out.txt", delimiter='\s{2, }', header=None)
a = a.drop(0)
a = a.drop(1)
a = a.reset_index(drop=True)

b = pd.DataFrame(data=[[0.0]*(dim+2) for i in range(int(len(a)/dim))])

for i in range(len(a)):
	print(str(i) , end='\r')
	row = int(i/dim);
	column = int(i%dim)
	if(column is 0):
		x = a.iloc[i,0].replace('(array([', '')
		x = x.strip(',')
	elif(column == dim-1):
		sep = ']), '
		x = a.iloc[i,0].split(sep,1)[0]
		b.iloc[row, dim] = int(a.iloc[i,0].split(sep,1)[1].split(',')[0])
		b.iloc[row, dim+1] = float(a.iloc[i,0].split(sep,1)[1].split(',')[1].strip(')'))
	else:
		x = a.iloc[i,0].strip(',')
	b.iloc[row,column] = x

b.to_csv("HyperCool.csv")