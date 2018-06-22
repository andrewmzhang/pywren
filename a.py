import sys
import pywren
import numpy as np
import math
import boto3
import pickle
import time
import random

from threading import Event
from threading import Thread
from sklearn import preprocessing
from scipy import sparse
from functools import reduce

HASH = 1000000
lr = 0.0001
batch_size = 1000
total_batches = 1000
batch_file_size = 3

from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import OneHotEncoder


kill_signal = Event()

# Prediction function
def prediction(param_dense, param_sparse, x_dense, x_sparse):
    val = -x_dense.dot(param_dense)
    val2 = -x_sparse.dot(param_sparse)
    val = val + val2
    out =  1 / (1 + np.exp(val))
    return out

def store_update(update, lid, gid):
    s3 = boto3.resource('s3')
    key = 'gradient_%d/g_%d' % (np.random.randint(1, 5), random.randint(1, 1000))
    datastr = pickle.dumps(update)
    s3.Bucket('camus-pywren-489').put_object(Key=key, Body=datastr)

# Map function for pywren main
def gradient_batch(xpys):
    t = time.time()
    model = get_data('model')
    xpys = xpys.split(" ")
    for xpy in xpys:
        left, right, det = gradient(xpy, model)
        out = (np.sum(left, axis=0), np.sum(right, axis=0))
        #return out[0], out[1]
        duration = time.time() - t
        det['dur'] = duration
        to_store = [t, time.time(), time.time() - t, det, out[0], out[1]]
        store_update(to_store, 1, 1)
        model = update_model(model, to_store[-2:])
    return to_store

# convert matrices to dense and sparse halves
def convert(x_idx, shape):
    x_sparse = sparse.lil_matrix((shape, HASH))
    for i in range(shape):
        x_sparse[i, x_idx[i]] = np.ones(len(x_idx[i]))
    return x_sparse

def gradient(xpy, model):
    det = {}

    t = time.time()
    param_dense, param_sparse = model

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
    tot = np.multiply(ys_temp, np.log(preds)) + np.multiply((1 - ys_temp), np.log(1 - preds))
    return np.mean(tot)

# AWS helper function
def get_data(key):
    s3 = boto3.resource('s3')
    obj = s3.Object('camus-pywren-489', key)
    body = obj.get()['Body'].read()
    data = pickle.loads(body)
    return data

def store_model(model):
    param_dense, param_sparse = model
    s3 = boto3.resource('s3')
    key = 'model'
    model = (param_dense, param_sparse)
    datastr = pickle.dumps(model)
    s3.Bucket('camus-pywren-489').put_object(Key=key, Body=datastr)

index = 1
def get_minibatches(num, over=10):
    global index
    group = []
    for i in range(num):
        if index + batch_file_size > total_batches:
            index = 1
        begin, end = index, index + over
        minis = []
        for b in range(begin, end):
            key = '1k-' + str(b)
            minis.append(key)
        index = index + over
        group.append(' '.join(mini for mini in minis))
    print(group)
    return group

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
    test_key = "1k-0"
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


grad_q = []

def fetch_thread(i):
    global grad_q
    global kill_signal
    s3 = boto3.resource('s3')

    my_bucket = s3.Bucket('camus-pywren-489')

    num = 0
    while not kill_signal.is_set():
        for object in my_bucket.objects.filter(Prefix='gradient_%d/' % i).all():
            s = time.time()
            obj = object.get()
            grad = pickle.loads(obj['Body'].read())
            grad_q.append(grad[-2:])
            object.delete()
            num += 1
            print("Fetched: %d, took: %f, thread: %d" % (num, time.time() - s, i))
            if kill_signal.is_set():
                return;


def error_thread(model):
    global grad_q
    global log
    global fname
    global kill_signal
    global outf 
    
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket('camus-pywren-489')
    num = 0
    print("Starting error thread")
    start_time = time.time()
    # Clear existing gradients

    test_data = get_test_data()

    saves = 0

    if log:
        f = open(fname.split(".")[0] + ".pkl", 'ab')
    while not kill_signal.is_set():
        grads = grad_q[:]
        grad_q = []
        if len(grads) > 0:
            bg = reduce_sum(grads)
            model = update_model(model, bg)
            store_model(model)
            num += len(grads)
            error = loglikelihood(test_data, model)
            curr_time = time.time() - start_time
            print("[ERROR_TASK]", curr_time, error, "this many grads:", num, "Sec / Grad:", (time.time() - start_time)/ num)
            if log:
                print("dumping")
                pickle.dump((curr_time, model), f)
                saves += 1
    if log:
        large_test = get_test_data()
        f.close()

        with open(fname.split(".")[0] + ".pkl", 'rb') as f:
            for i in range(saves):
                t, model = pickle.load(f)
                error = loglikelihood(test_data, model)
                print("wrote: %f %f" % (t, error))
                outf.write("%f, %f\n" % (t, error))

            




def main(thread, log=False):

    global kill_signal

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
    while time.time() - start_time < 30:
        print("hit")
        # Store model
        fin = 0
        res = []
        ded = []

        #print("Start pool")
        t = time.time()
        pool = ThreadPool(6)
        resa = []
        resa = pool.map(m, fs)
        #print("End pool: %f" % (time.time() - t))

        res = []
        for a in resa:
            if a != None:
                fs.remove(a[1])
                res.append(a[0])
                print(a[0][3])


        fin = len(res)
        iter += fin
        if fin > 0:
            print("Processed: %d" % fin)
            minibatches = get_minibatches(fin)
            # Run Map Reduce with Pywren
            fs.extend(start_batch(minibatches))
    kill_signal.set()
    print("Main thread has stopped")
log = False
fname = ""
outf = None
if __name__ == "__main__":


    log = False
    if len(sys.argv) > 2:
        log = True
        fname = sys.argv[1]
        lr = float(sys.argv[2])
        outf = open(fname, "w")
        print("Logging was requested with output file: %s and rate: %f" % (fname, lr))

    s3 = boto3.resource('s3')

    my_bucket = s3.Bucket('camus-pywren-489')
    for object in my_bucket.objects.filter(Prefix='gradient_1/').all():
        object.delete()
    for object in my_bucket.objects.filter(Prefix='gradient_2/').all():
        object.delete()
    for object in my_bucket.objects.filter(Prefix='gradient_3/').all():
        object.delete()
    for object in my_bucket.objects.filter(Prefix='gradient_4/').all():
        object.delete()
    time.sleep(5)
    model = init_model()
    store_model(model)

    thread = Thread(target=error_thread, args=(model, ))
    ft = Thread(target=fetch_thread, args=(1, ))
    ft2 = Thread(target=fetch_thread, args=(2, ))
    ft3 = Thread(target=fetch_thread, args=(3, ))
    ft4 = Thread(target=fetch_thread, args=(4, ))

    ft.start()
    ft2.start()
    ft3.start()
    ft4.start()

    try:
        main(thread, log)
    except KeyboardInterrupt:
        kill_signal.set()
        ft.join()
        ft2.join()
        ft3.join()
        ft4.join()
        thread.join()
        if log:
            outf.close()
            print("outf closed by interrupt")
        exit()


    if log:
        kill_signal.set()
        print("issued log halt")
        ft.join()
        ft2.join()
        ft3.join()
        ft4.join()
        thread.join()
        time.sleep(10)
        outf.close()
        print("outf closed by END")
