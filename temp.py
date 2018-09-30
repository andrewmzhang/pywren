
HASH = 524288

import redis
import sys
import pywren
import numpy as np
import math
import boto3
import pickle
import time
import random
import json
from threading import Event
from threading import Thread
from sklearn import preprocessing
from scipy import sparse
from functools import reduce
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import OneHotEncoder

import scipy

def get_local_test_data():
    f = open("testset.data", "rb")
    x_dense_test, x_idx_test, y_test = pickle.load(f)
    x_sparse_test = sparse.lil_matrix((x_dense_test.shape[0], HASH))
    for i in range(x_dense_test.shape[0]):
        try:
            x_sparse_test[i, x_idx_test[i]] = np.ones(len(x_idx_test[i]))
        except:
            print(x_sparse_test.shape, i, x_idx_test[i])
            return
    f.close()
    return (x_dense_test, x_sparse_test, y_test)


get_local_test_data()
