
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
batch_size = 50000
total_batches = 900
batch_file_size = 5

from sklearn.preprocessing import OneHotEncoder


def prediction(param_dense, param_sparse, x_dense, x_sparse):
    boom = time.time()
    val = -x_dense.dot(param_dense)
    print("dense part: ", time.time() - boom)
    val2 = -x_sparse.dot(param_sparse)
    print("sparse part: ", time.time() - boom)
    #print(x_sparse.shape, param_sparse.shape)
    val = val + val2
    out =  1 / (1 + np.exp(val))
    return out

def gradient(xpy):
    # key = xpy
    param_dense, param_sparse = get_data('model')
    x_dense, x_idx, y = get_data(xpy)
    x_sparse = sparse.lil_matrix((x_dense.shape[0], HASH))

    start_time = time.time()
    print("Starting conversion")

    #for i in range(x_dense.shape[0]):
    #    for idx in x_idx[i]:
    #        x_sparse[i, idx] = 1

    for i in range(x_dense.shape[0]):
        x_sparse[i, x_idx[i]] = np.ones(len(x_idx[i]))
    print("end conversion", time.time() - start_time)

    y = np.reshape(y, (-1, 1))
    ttt = time.time()
    print("Prediction starts")
    error = y - prediction(param_dense, param_sparse, x_dense, x_sparse)
    print("Prediction stops", time.time() - ttt)
    error = np.reshape(error, (-1,))

    # left = np.multiply(logits, x_dense)
    left = np.multiply(x_dense, error.reshape(-1, 1))

    # logits = np.tile(error, (x_sparse.shape[1], 1)).T
    # right = np.multiply(logits, x_sparse)
    # right = np.multiply(x_sparse, error.T)
    temp = sparse.lil_matrix((x_dense.shape[0], x_dense.shape[0]))
    temp.setdiag(error.A1)
    right = x_sparse.T * temp

    # right = np.multiply(x_sparse, error.T)
    right = right.T
    print("gb", right.shape)

    # logits = np.tile(error, (x_dense.shape[1], 1)).T
    #
    # #print("end error", x_sparse.shape)
    #
    # left = np.multiply(logits, x_dense)
    #
    # logits = np.tile(error, (x_sparse.shape[1], 1)).T
    # right = np.multiply(logits, x_sparse)
    print("end multiply", right.shape)
    return left, right

def gradient_batch(xpy):
    s = time.time()
    left, right = gradient(xpy)
    #print("rhight", right.shape)
    out = (np.sum(left, axis=0), np.sum(right, axis=0))
    print("tt", time.time() - s)
    return (out[0], out[1], time.time() - s)
def loglikelihood(xs_dense, xs_sparse, ys, param_dense, param_sparse):

    preds = prediction(param_dense, param_sparse, xs_dense, xs_sparse)
    ys_temp = ys.reshape((-1, 1))
    tot = np.multiply(ys_temp, np.log(preds)) + np.multiply((1 - ys_temp), np.log(1 - preds))
    ##print("tot", tot.shape)
    return np.mean(tot)

def loglikelihood_each(kpp):

    # key, param_dense, param_sparse = kpp
    param_dense, param_sparse = get_data('model')
    x_dense, x_idx, y = get_data(kpp)
    x_sparse = sparse.lil_matrix((x_dense.shape[0], HASH))
    print("Starting conversion")

    for i in range(x_dense.shape[0]):
        x_sparse[i, x_idx[i]] = np.ones(len(x_idx[i]))
    print("end conversion")

    preds = prediction(param_dense, param_sparse, x_dense, x_sparse)
    ys_temp = y.reshape((-1, 1))
    tot = np.multiply(ys_temp, np.log(preds)) + np.multiply((1 - ys_temp), np.log(1 - preds))
    ##print("tot", tot.shape)
    return np.mean(tot)

def reduce_sum(lst):
    #l1, r1 = lst[0]
    #l2, r2 = lst[1]
    print("start lc")


    print("end lc")
    ##print("r1shapes", r1.shape, r2.shape)
    # return np.sum(np.vstack(ls), axis=0), np.sum(np.vstack(rs), axis=0)
    
    s = [l[2] for l in lst]
    s = sum(s) / len(lst)

    start_time = time.time()

    l = [l[0] for l in lst]
    r = [l[1] for l in lst]
    a = np.sum(np.vstack(l), axis=0), np.sum(np.vstack(r), axis=0)
    return a, b, s + (time.time() - start_time)

def reduce_mean(lst):
    return np.mean(lst)

upload = True

dict = {}
def get_data(key):
    s3 = boto3.resource('s3')
    obj = s3.Object('camus-pywren-991', key)
    body = obj.get()['Body'].read()
    data = pickle.loads(body)
    return data
    #return dict[key]


if __name__ == "__main__":


    # converter = {}
    # for i in range(14):
    #     converter[i] =  lambda s: float(s.strip() or 0)
    # for i in range(14, 40):
    #     converter[i] = lambda s: hash(s) % HASH

    # data = np.loadtxt("sample.txt", converters=converter, delimiter="\t")

    # ys = data[:, 0]
    # xs_dense = data[:, 1:14]
    # xs_sparse = data[:, 14:]

    test_key = 'test0'
    print("Start downloading")
    x_dense_test, x_idx_test, y_test = get_data(test_key)
    print("End downloading")
    x_sparse_test = sparse.lil_matrix((x_dense_test.shape[0], HASH))
    #print("Starting conversion")

    #for i in range(x_dense_test.shape[0]):
    #     for idx in x_idx_test[i]:
    #         x_sparse_test[i, idx] = 1

    for i in range(x_dense_test.shape[0]):
        x_sparse_test[i, x_idx_test[i]] = np.ones(len(x_idx_test[i]))


    param_dense = np.zeros((x_dense_test.shape[1], 1))
    param_sparse = sparse.lil_matrix((HASH, 1))

    # batches = int(math.ceil(xs_dense.shape[0] / batch_size))
    s3 = boto3.resource('s3')

    #all_files = []
    #for b in range(1, total_batches):
    #  key = 'small' + str(b)
    #  all_files.append((key, param_dense, param_sparse))

    ##print("initial: ", loglikelihood(xs_dense, ys, param))

    key = 'model'
    model = (param_dense, param_sparse)
    datastr = pickle.dumps(model)
    s3.Bucket('camus-pywren-991').put_object(Key=key, Body=datastr)

    start_time = time.time()

    for iter in range(100):
        index = 1
        print("starting while")
        while index + batch_file_size <= total_batches:
            begin, end = index, index + batch_file_size
            index += batch_file_size
        # begin, end = 1, total_batches
            temp = []
            for b in range(begin, end):
                key = 'test' + str(b)
                temp.append(key)
        # print("done upload")
        wrenexec = pywren.default_executor()

        futures = wrenexec.map(gradient_batch, temp)
        reduce_future = wrenexec.reduce(reduce_sum, futures)
        start_time = time.time()
        left, right, tt = reduce_future.result()
        print("Persp", time.time() - start_time, tt)
        #a = [gradient_batch(t) for t in temp]
        #left, right = reduce_sum(a)
        print(left)
        print(right)

        print("starting map", time.time() - start_time)
        print("end construction")

        left = np.reshape(left, (14, 1))
        right = np.reshape(right, (HASH, 1))
        ##print(param_sparse.shape, np.multiply(lr, right).shape)
        ##print(param_sparse)
        ##print("2", np.multiply(lr, right))

        param_dense = np.add(param_dense, np.multiply(lr, left))
        # param_sparse = np.add(param_sparse, np.multiply(lr, right))
        param_sparse = sparse.lil_matrix(np.add(param_sparse.todense(), np.multiply(lr, right)))


        model = (param_dense, param_sparse)
        datastr = pickle.dumps(model)
        s3.Bucket('camus-pywren-991').put_object(Key='model', Body=datastr)

        #print(loglikelihood(xs_dense, xst_sparse, ys, param_dense, param_sparse))
        print(time.time() - start_time, loglikelihood(x_dense_test, x_sparse_test, y_test, param_dense, param_sparse))
        #print(time.time() - start_time, "Finished updating")
        #print("computing training loss")

        #futures = wrenexec.map(loglikelihood_each, temp)
        #reduce_future = wrenexec.reduce(reduce_mean, futures)
        #log = [utils.loglikelihood_each(t) for t in temp]
        #log = utils.reduce_mean(log)

        #print(time.time() - start_time, reduce_future.result())z
