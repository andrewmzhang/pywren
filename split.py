import numpy as np
from sklearn import preprocessing

import math
import boto3
import pickle

# Split the large file into bits
batch_size = 20      # size of minibatch


header_name = "sample"
suffix_name =".txt"

input_file = "train.txt"
max_size = 1 #500

with open(input_file, "rb") as f:
    cnt = 0
    file_idx = 0;
    out = open(header_name + str(file_idx) + suffix_name, "wb")
    for line in f:
        idx = int(cnt / batch_size)
        out.write(line)
        cnt += 1
        if (cnt % batch_size == 0 and cnt > 0):
            out.close()
            file_idx += 1
            out = open(header_name + str(file_idx) + suffix_name, "wb")
        if (idx > max_size):
            break;
    out.close()



HASH = 1000000
batches = file_idx

print("this many batches", batches)

converter = {}
for i in range(14):
    converter[i] =  lambda s: float(s.strip() or 0)
for i in range(14, 40):
    converter[i] = lambda s: hash(s) % HASH
s3 = boto3.resource('s3')

scaler = preprocessing.MinMaxScaler()

fit_mat = [
[0,-3, 0, 0, 0, 0,0,0, 0,0,0, 0,0], 
[5775, 257675, -3, 65535, 969, 23159500, 431037, 56311, 6047, 29019, 11, 231, 4008, 7393] ]

scaler.fit(fit_mat)
for b in range(min(batches, max_size)):
    
    data = np.loadtxt("sample"+str(b)+'.txt', converters=converter, delimiter="\t")
    key = '1k-' + str(b)
    
    print("Setting up", key)
    assert data.shape[0] == batch_size
    ys = data[:, 0]
    xs_dense = data[:, 1:14]
    xs_sparse = data[:, 14:]

    xs_dense = scaler.transform(xs_dense)
    xs_dense = np.column_stack([np.ones((xs_dense.shape[0])), xs_dense]) # N by (D+1)
    exit()

    batch = (xs_dense, xs_sparse, ys)
    datastr = pickle.dumps(batch)
#    s3.Bucket('camus-pywren-489').put_object(Key=key, Body=datastr)

