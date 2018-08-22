import numpy as np
from sklearn import preprocessing

import math
import boto3
import pickle

HASH = 524288

converter = {}
for i in range(14):
    converter[i] =  lambda s: float(s.strip() or 0)
for i in range(14, 40):
    converter[i] = lambda s: hash(s) % HASH

data = np.loadtxt('testset.txt', converters=converter, delimiter="\t")


ys = data[:, 0]
xs_dense = data[:, 1:14]
xs_sparse = data[:, 14:]

scaler = preprocessing.MinMaxScaler()
fit_mat = [
        [0,-3, 0, 0, 0, 0,0,0, 0,0,0, 0,0], 
        [5775, 257675, 65535, 969, 23159500, 431037, 56311, 6047, 29019, 11, 231, 4008, 7393] ]
fit_mat = np.matrix(fit_mat)
scaler.fit(fit_mat)


xs_dense = scaler.transform(xs_dense)

xs_dense = np.column_stack([np.ones((xs_dense.shape[0])), xs_dense]) # N by (D+1)

batch = (xs_dense, xs_sparse, ys)
out = open("testset.data", "wb")
pickle.dump(batch, out)
out.close()

