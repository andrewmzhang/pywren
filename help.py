import numpy as np
from sklearn import preprocessing

import math
import boto3
import pickle


def get_test():
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

    xs_dense = np.column_stack([xs_dense]) # N by (D+1)

    training_data = []

    for i in range(xs_dense.shape[0]):
        label = ys[i]

        cnt = 0
        row = []
        for x in xs_dense[i]:
            row.append((cnt, x))
            cnt += 1
        for idx in xs_sparse[i]:
            row.append((int(idx+14),1))
        training_data.append([label, row])
        if i % 10000 == 0:
            print(i, data.shape[0])
    return training_data

