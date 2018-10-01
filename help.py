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

    data = np.loadtxt('randtest1.txt', converters=converter, delimiter="\t")


    ys = data[:, 0]
    xs_dense = data[:, 1:14]
    xs_sparse = data[:, 14:]


    scaler = preprocessing.MinMaxScaler()
    fit_mat = [
        [0,      -2,     0,   0,        0,      0,    0,    0,     0,    0,   0,    0,    0], 
        [936, 19999, 65535, 390,  2502894,  27526, 6428, 2466,  9858,    7, 144,  451, 2278] ]

    training_data = []

    for i in range(xs_dense.shape[0]):
        label = ys[i]

        cnt = 0
        row = []
        for x in xs_dense[i]:
            new_value = (x - fit_mat[0][cnt]) / (fit_mat[1][cnt] - fit_mat[0][cnt])
            row.append((cnt, new_value))
            cnt += 1
        for idx in xs_sparse[i]:
            row.append((int(idx+14),1))
        training_data.append([label, row])
        if i % 10000 == 0:
            print(i, data.shape[0])
    return training_data

