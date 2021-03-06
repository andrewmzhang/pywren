import queue
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

HASH = 524288

lr = 0.0001            # Learning rate
minibatch_size = 20    # Size of minibatch
batch_size = 3000      # Size of whole batch
total_batches = 500   # Entire number of batches in dataset
batch_file_size = 10    # Number of lambda 
num_lambdas = 10
total_time = 60
log = False
fname = "def.txt"
outf = None


from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import OneHotEncoder

fin = []

# Prediction function
def prediction(param_dense, param_sparse, x_dense, x_sparse):
    val = -x_dense.dot(param_dense)
    val2 = -x_sparse.dot(param_sparse)
    val = val + val2
    out =  1 / (1 + np.exp(val))
    return out

def store_update(update):
    t0 = time.time()
    s3 = boto3.resource('s3')
    key = 'gradient_%d/g_%d' % (np.random.randint(1, 9), random.randint(1, 10))
    datastr = pickle.dumps(update)
    t1 = time.time()
    def up_func(s3, key, datastr):
        s3.Bucket('camus-pywren-489').put_object(Key=key, Body=datastr)
    thread = Thread(target = up_func, args = (s3, key, datastr ))
    thread.start()
    # Return reserialzie, upload
    return t1 - t0, time.time() - t1


# Map function for pywren main
def gradient_batch(xpys):
    start = time.time()
    model = get_data('model')
    fetch_model_time = time.time() - start
    xpys = xpys.split(" ")
    reser, upload, model_fetch, model_deser = 0, 0, 0, 0

    iterno = 0
    for xpy in xpys:
        det = {}
        for mb in get_minis(xpy, det):
            if iterno > 0:
                det['loop_time'] = time.time() - prev_start_time
            prev_start_time = time.time()

            left, right, det = gradient(mb, model, det)
            out = (np.sum(left, axis=0), np.sum(right, axis=0))

            if iterno == 0:
                det['init_model_fetch'] = fetch_model_time

            if iterno > 0:
                det['reserialize_time'] = reser
                det['upload_time'] = upload
                det['model_deser_time'] = model_deser
                det['model_fetch_time'] = model_fetch

            det['subtime'] = time.time()
            to_store = [det, out[0], out[1]]
            reser, upload = store_update(to_store)
            model, model_deser, model_fetch = get_data('model', True)
            iterno += 1
            break;
        break;
    
    to_store[0]['lambda_time_alive'] = time.time() - start
    to_store[0]['total_iters'] = iterno
    return to_store

# convert matrices to dense and sparse halves
def convert(x_idx, shape):
    x_sparse = sparse.lil_matrix((shape, HASH))
    for i in range(shape):
        x_sparse[i, x_idx[i]] = np.ones(len(x_idx[i]))
    return x_sparse

def gradient(xpy, model, det):
    
    start = time.time()
    # Fetch model
    param_dense, param_sparse = model
    
    # Fetch data and deser
    (x_dense, x_idx, y) = xpy

    # Convert data to sparse
    t = time.time()
    x_sparse = convert(x_idx, x_dense.shape[0])
    det['convert_to_sparse'] = time.time() - t


    
    # Predictions
    t = time.time()
    y = np.reshape(y, (-1, 1))
    error = y - prediction(param_dense, param_sparse, x_dense, x_sparse)
    error = np.reshape(error, (-1,))
    det['prediction_time'] = time.time() - t


    # Calculation left_gradient
    t = time.time()
    left_grad = np.multiply(x_dense, error.reshape(-1, 1))
    det['lgradient_calc'] = time.time() - t

    # Calculate right gradient (sparse part)
    t = time.time()
    temp = sparse.lil_matrix((x_dense.shape[0], x_dense.shape[0]))
   
    temp.setdiag(error.A1)
    right = temp.T * x_sparse
    right_grad = right
    det['rgradient_calc'] = time.time() - t
    
    det['total_gradient_loop' ] = time.time() - start
    return left_grad, right_grad, det

# Reduce function
def reduce_sum(lst):
    start_time = time.time()
    l = [l[0] for l in lst]
    r = [l[1] for l in lst]
    left_grad, right_grad = np.sum(np.vstack(l), axis=0), np.sum(np.vstack(r), axis=0)
    return left_grad, right_grad

# Log loglikelihood func
def loglikelihood(test_data, model):
    xs_dense, xs_sparse, ys = test_data
    param_dense, param_sparse = model
    preds = prediction(param_dense, param_sparse, xs_dense, xs_sparse)
    ys_temp = ys.reshape((-1, 1))
    cors = np.around(preds)
    cors -= ys_temp
    cors = np.abs(cors)
    print("Accu", np.mean(cors))
    tot = np.multiply(ys_temp, np.log(preds)) + np.multiply((1 - ys_temp), np.log(1 - preds))
    return np.mean(tot)

def get_minis(key, det):
    data, deser, fetch = get_data(key, True)
    det['batch_deserialized_time'] = deser
    det['batch_fetch_time'] = fetch
    a, b, c = data
    idx = 0
    while idx + minibatch_size <= batch_size:
        yield a[idx: idx +minibatch_size], b[idx: idx +minibatch_size], c[idx: idx +minibatch_size], 
        idx += minibatch_size


# AWS helper function
def get_data(key, a = False):
    s3 = boto3.resource('s3')
    t0 = time.time()
    obj = s3.Object('camus-pywren-489', key)
    t1 = time.time()
    body = obj.get()['Body'].read()
    data = pickle.loads(body)
    # Return data, time to deseralize, time to fetch
    if a:
        return data, time.time() - t1, t1 - t0
    return data

def store_model(model):
    param_dense, param_sparse = model
    s3 = boto3.resource('s3')
    key = 'model'
    model = (param_dense, param_sparse)
    datastr = pickle.dumps(model)
    s3.Bucket('camus-pywren-489').put_object(Key=key, Body=datastr)

index = 1
def get_minibatches(num, over=2):
    global index
    group = []
    for i in range(num):
        if index + batch_file_size > total_batches:
            index = 1
        begin, end = index, index + over
        minis = []
        for b in range(begin, end):
            key = '3k-' + str(b)
            minis.append(key)
        index = index + over
        group.append(' '.join(mini for mini in minis))
    print(group)
    return group

def update_model(model, gradient):
    global lr
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
    test_key = "3k-0"
    x_dense_test, x_idx_test, y_test = get_data(test_key)
    x_sparse_test = sparse.lil_matrix((x_dense_test.shape[0], HASH))
    for i in range(x_dense_test.shape[0]):
        x_sparse_test[i, x_idx_test[i]] = np.ones(len(x_idx_test[i]))
    return (x_dense_test, x_sparse_test, y_test)

def get_local_test_data():
    f = open("testset.data", "rb")
    f.seek(0)
    x_dense_test, x_idx_test, y_test = pickle.load(f)
    x_sparse_test = sparse.lil_matrix((x_dense_test.shape[0], HASH))
    for i in range(x_dense_test.shape[0]):
        x_sparse_test[i, x_idx_test[i]] = np.ones(len(x_idx_test[i]))
    f.close()
    return (x_dense_test, x_sparse_test, y_test)

def start_batch(minibatches):
    wrenexec = pywren.default_executor()
    futures = wrenexec.map(gradient_batch, minibatches)  # Map future
    return futures

def m(f):
    try:
        if f.done():
            return f.result(), f
    except:
        return [], f


def error_thread(model):
    global grad_q
    global log
    global fname
    global index
    
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('camus-pywren-489')
    num = 0
    print("Starting error thread")
    start_time = time.time()
    # Clear existing gradients

    test_data = get_test_data()

    saves = 0

    if True:
        print(fname[:-4] + ".pkl")
        f = open(fname[:-4] + ".pkl", 'wb')
    time_model_lst = []
    while time.time() - start_time < total_time:
        
        if not grad_q.empty():
            sz = grad_q.qsize()
            grads = []
            for _ in range(sz):
                grad = grad_q.get()
                grads.append(grad)
                grad_q.task_done()
            bg = reduce_sum(grads)
            model = update_model(model, bg)
            store_model(model)
            num += len(grads)
            #error = loglikelihood(test_data, model)
            curr_time = time.time() - start_time
            print("[ERROR_TASK]", curr_time, 0, "this many grads:", num, "Sec / Grad:", (time.time() - start_time)/ num)
            if True:
                print("dumping")
                pickle.dump((curr_time, model), f)
                print("dump done")
                saves += 1

    print("Saves: ", saves, "Index:", index)
    if True:
        large_test = get_local_test_data()
        f.close()
        outf = open(fname[:-4] + ".csv", "w")
        with open(fname[:-4] + ".pkl", 'rb') as f:
            for i in range(saves):
                t, model = pickle.load(f)
                error = loglikelihood(large_test, model)
                print("wrote: %f %f" % (t, error))
                outf.write("%f, %f\n" % (t, error))
        outf.close()
         




def main(thread, log=False):

    global outf
    global total_time

    # Initialize model

    print("Starting Training" + '-' * 30)
    start_time = time.time()
    fs = []

    fin = batch_file_size

    # start jobs
    minibatches = get_minibatches(fin)
    fs.extend(start_batch(minibatches))
    fin = 0
    iter = 0

    thread.start()
    print("Main thread start")
    while time.time() - start_time < total_time:
        print("hit", time.time() - start_time)
        # Store model
        fin = 0
        res = []
        ded = []
        t = time.time()



        fin = len(res)
        iter += fin
        print("Processed: %d" % fin)
        minibatches = get_minibatches(fin)
        # Run Map Reduce with Pywren
    print("Main thread has stopped")


if __name__ == "__main__":
    print(len(sys.argv)) 
    global outf
    global lr
    global total_time
    global log
    global fname
    log = False
    if len(sys.argv) >= 2:
        data = json.loads(sys.argv[1])
        total_time = float(data['total_time'])
        log = True
        fname = data['fname']
        lr = float(data['lr'])
        outf = open(fname, "w")
        
        outf.write("lr: %f\n" % lr)
        outf.write("minibatch_size: %f\n" % minibatch_size)
        outf.write("batch_file_size: %d\n" % batch_file_size)
        outf.write("num_lambdas: %d\n" % num_lambdas)
        outf.write("fname: %s\n" % fname)
        outf.write("total_time: %d\n" % total_time)
        
        print("Logging was requested with output file: %s and rate: %f" % (fname, lr))

    s3 = boto3.resource('s3')

    my_bucket = s3.Bucket('camus-pywren-489')

    for i in range(1, 9):
        string = "gradient_%d/" % i
        for object in my_bucket.objects.filter(Prefix=string).all():
            object.delete()

    time.sleep(5)
    model = init_model()
    store_model(model)

    thread = Thread(target=error_thread, args=(model, ))

    main(thread)



    if log:
        print("issued log halt")
        thread.join()
        time.sleep(10)
        outf.close()
        print("outf closed by END")
