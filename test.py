import numpy as np
from sklearn import preprocessing

import math
import boto3
import pickle

HASH = 1000000

converter = {}
for i in range(14):
    converter[i] =  lambda s: float(s.strip() or 0)
for i in range(14, 40):
    converter[i] = lambda s: hash(s) % HASH
    
data = np.loadtxt('testset.txt', converters=converter, delimiter="\t")

print("Setting up", key)
assert data.shape[0] == batch_size
ys = data[:, 0]
xs_dense = data[:, 1:14]
xs_sparse = data[:, 14:]

min_max_scaler = preprocessing.MinMaxScaler()
xs_dense = min_max_scaler.fit_transform(xs_dense)
xs_dense = np.column_stack([np.ones((xs_dense.shape[0])), xs_dense]) # N by (D+1)

batch = (xs_dense, xs_sparse, ys)
datastr = pickle.dumps(batch)


out = open("testset.data", "w")

out.write(datastr)

out.close()
