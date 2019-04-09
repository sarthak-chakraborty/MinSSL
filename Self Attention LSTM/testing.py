import torch
import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import torch.utils.data as data_utils
from keras.datasets import reuters
import csv

max_len = 300

train_set,test_set = reuters.load_data(path="reuters.npz",num_words=20000,skip_top=0,index_from=3)

x_train,y_train = train_set[0],train_set[1]
print(len(y_train))

# with open("train_label.csv", "w") as f:
#    	wr = csv.writer(f, dialect='excel')
#    	wr.writerow(y_train)

x_test,y_test = test_set[0],test_set[1]
print(len(y_test))

# with open("test_label.csv", "w") as f:
#     wr = csv.writer(f, dialect='excel')
#     wr.writerow(y_test)

word_to_id = reuters.get_word_index(path="reuters_word_index.json")
word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2
word_to_id['<EOS>'] = 3
id_to_word = {value:key for key,value in word_to_id.items()}

x_train_pad = pad_sequences(x_train,maxlen=max_len)
x_test_pad = pad_sequences(x_test,maxlen=max_len)

train_data = data_utils.TensorDataset(torch.from_numpy(x_train_pad).type(torch.LongTensor),torch.from_numpy(y_train).type(torch.LongTensor))
print(len(train_data))
train_loader = data_utils.DataLoader(train_data,batch_size=512,drop_last=True)

for batch_idx,train in enumerate(train_loader):
	# print(batch_idx)
	print(train[0])
	print("")

# print(id_to_word)