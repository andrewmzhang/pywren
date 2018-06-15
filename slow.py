
import pywren
import numpy as np
import math
import boto3
import pickle
import time

from sklearn import preprocessing
from scipy import sparse
from functools import reduce

HASH = 1000000
lr = .000002
batch_size = 1000
total_batches = 900
batch_file_size = 5

from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import OneHotEncoder


# Prediction function
def prediction(param_dense, param_sparse, x_dense, x_sparse):
    val = -x_dense.dot(param_dense)
    val2 = -x_sparse.dot(param_sparse)
    val = val + val2
    out =  1 / (1 + np.exp(val))
    return out

# Map function
def gradient_batch(xpy):
    t = time.time()
    left, right, det = gradient(xpy)
    out = (np.sum(left, axis=0), np.sum(right, axis=0))
    #return out[0], out[1]
    duration = time.time() - t
    det['dur'] = duration
    return t, time.time(), time.time() - t, det, out[0], out[1]

# convert matrices to dense and sparse halves
def convert(x_idx, shape):
    x_sparse = sparse.lil_matrix((shape, HASH))
    for i in range(shape):
        x_sparse[i, x_idx[i]] = np.ones(len(x_idx[i]))    
    return x_sparse

def gradient(xpy):
    det = {}

    t = time.time()
    param_dense, param_sparse = get_data('model')
    det['fetch_model'] = time.time() - t
    t = time.time()
    x_dense, x_idx, y = get_data(xpy)    
    det['fetch_data'] = time.time() - t
    t = time.time()
    x_sparse = convert(x_idx, x_dense.shape[0])
    det['convert'] = time.time() - t 

    t = time.time()
    y = np.reshape(y, (-1, 1))
    error = y - prediction(param_dense, param_sparse, x_dense, x_sparse)
    error = np.reshape(error, (-1,))
    det['pred'] = time.time() - t
    t = time.time()
    left_grad = np.multiply(x_dense, error.reshape(-1, 1))
    det['lmult'] = time.time() - t
   
    t = time.time()
    temp = sparse.lil_matrix((x_dense.shape[0], x_dense.shape[0]))
    det['rmult0'] = time.time() - t
    t = time.time()
    temp.setdiag(error.A1)
    det['rmult1'] = time.time() - t
    t = time.time()
    right = temp.T * x_sparse
    det['rmult2'] = time.time() - t
    t = time.time()
    right_grad = right
    det['rmult3'] = time.time() - t
    return left_grad, right_grad, det

# Reduce function
def reduce_sum(lst):
    start_time = time.time()
    l = [l[4] for l in lst]
    r = [l[5] for l in lst]
    left_grad, right_grad = np.sum(np.vstack(l), axis=0), np.sum(np.vstack(r), axis=0)
    return left_grad, right_grad

# Log loglikelihood func 
def loglikelihood(test_data, model):
    xs_dense, xs_sparse, ys = test_data
    param_dense, param_sparse = model
    preds = prediction(param_dense, param_sparse, xs_dense, xs_sparse)
    ys_temp = ys.reshape((-1, 1))
    tot = np.multiply(ys_temp, np.log(preds)) + np.multiply((1 - ys_temp), np.log(1 - preds))
    return np.mean(tot)

# AWS helper function
def get_data(key):
    s3 = boto3.resource('s3')
    obj = s3.Object('camus-pywren-991', key)
    body = obj.get()['Body'].read()
    data = pickle.loads(body)
    return data

def store_model(model):
    param_dense, param_sparse = model
    s3 = boto3.resource('s3')
    key = 'model'
    model = (param_dense, param_sparse)
    datastr = pickle.dumps(model)
    s3.Bucket('camus-pywren-991').put_object(Key=key, Body=datastr)

def get_minibatches(index, num):
    if index + batch_file_size > total_batches:
        index = 1
    begin, end = index, index + num
    minis = []
    for b in range(begin, end):
        key = 'small' + str(b)
        minis.append(key)
    return minis

def update_model(model, gradient):
    left, right = gradient
    left = np.reshape(left, (14, 1))
    right = np.reshape(right, (HASH, 1))
    param_dense, param_sparse = model
    param_dense = np.add(param_dense, np.multiply(lr, left))
    param_sparse = sparse.lil_matrix(np.add(param_sparse.todense(), np.multiply(lr, right)))
    return (param_dense, param_sparse)

def init_model():
    param_dense = np.zeros((14, 1))
    param_sparse = sparse.lil_matrix((HASH, 1))
    model = (param_dense, param_sparse)
    return model

def get_test_data():
    test_key = "small0"
    x_dense_test, x_idx_test, y_test = get_data(test_key)
    x_sparse_test = sparse.lil_matrix((x_dense_test.shape[0], HASH))
    for i in range(x_dense_test.shape[0]):
        x_sparse_test[i, x_idx_test[i]] = np.ones(len(x_idx_test[i]))
    return (x_dense_test, x_sparse_test, y_test)


def start_batch(minibatches):
    wrenexec = pywren.default_executor()
    futures = wrenexec.map(gradient_batch, minibatches)  # Map future
    return futures

def m(f):
    if f.done():
        return f.result(), f

if __name__ == "__main__":

    # Get Test data
    test_data = get_test_data()

    # Initialize model
    model = init_model()

    print("Starting Training" + '-' * 30)
    start_time = time.time()
    index = 1
    fs = []

    fin = batch_file_size
    store_model(model)
    
    # start jobs
    minibatches = get_minibatches(index, fin)
    index += fin
    fs.extend(start_batch(minibatches))
    fin = 0

    iter = 0
    while iter < 100:
        # Store model
        
        
        fin = 0
        res = []
        ded = []

        
        t = time.time()
        print(pywren.get_all_results(fs))
        print(time.time() - t)
        exit()



        exit()
        print("Start pool")
        t = time.time()
        pool = ThreadPool(6)
        resa = []
        resa = pool.map(m, fs)
        print("End pool: %f" % (time.time() - t))
        
        res = []
        for a in resa:
            if a != None:
                fs.remove(a[1]) 
                res.append(a[0])
                print(a[0][3])
        fin = len(res)
        iter += fin
        print("Processed: %d" % fin)
        if fin > 0:
            gradients = reduce_sum(res)
            model = update_model(model, gradients)
            store_model(model)
            print(time.time() - start_time, loglikelihood(test_data, model))

            minibatches = get_minibatches(index, fin)
            index += fin
        
            # Run Map Reduce with Pywren
            fs.extend(start_batch(minibatches))
            print("Iteration: %d, finished: %d" % (iter, fin))
