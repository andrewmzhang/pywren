from scipy import sparse

import sys
import pywren
import numpy as np
import math
import boto3
import pickle
import time
import random



# Prediction function
def prediction(param_dense, param_sparse, x_dense, x_sparse):
    val = -x_dense.dot(param_dense)
    val2 = -x_sparse.dot(param_sparse)
    val = val + val2
    out =  1 / (1 + np.exp(val))
    return out


# Log loglikelihood func
def loglikelihood(test_data, model):
    xs_dense, xs_sparse, ys = test_data
    param_dense, param_sparse = model
    preds = prediction(param_dense, param_sparse, xs_dense, xs_sparse)
    ys_temp = ys.reshape((-1, 1))
    tot = np.multiply(ys_temp, np.log(preds)) + np.multiply((1 - ys_temp), np.log(1 - preds))
    return np.mean(tot)


def get_local_test_data():
    f = open("testset.data", "rb")
    x_dense_test, x_idx_test, y_test = pickle.load(f)
    f.close()
    x_sparse_test = sparse.lil_matrix((x_dense_test.shape[0], 1000000))
    for i in range(x_dense_test.shape[0]):
        x_sparse_test[i, x_idx_test[i]] = np.ones(len(x_idx_test[i]))
    return (x_dense_test, x_sparse_test, y_test)


large_test = get_local_test_data()


outf = open("tmp-0.000500-loss.csv", "w")
with open("tmp-0.000500-loss.pkl", 'rb') as f:
    for i in range(300):
        t, model = pickle.load(f)
        error = loglikelihood(large_test, model)
        print("wrote: %f %f" % (t, error))
        outf.write("%f, %f\n" % (t, error))
outf.close()

