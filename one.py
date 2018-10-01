from random import seed
from random import randrange
from csv import reader
from math import exp
import time
import mmh3
import sys
import numpy as np

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

from help import get_test

HASH = 524288
HASH_SIZE = HASH
lr = 0.0001            # Learning rate
minibatch_size = 20    # Size of minibatch
MB_SIZE = minibatch_size
batch_size = 20      # Size of whole batch
total_batches = 100   # Entire number of batches in dataset
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
def predict(row_values, coefficients):
    yhat = coefficients[0]
    for i in range(len(row_values)):
        index = row_values[i][0]
        value = row_values[i][1]
        yhat += coefficients[index + 1] * value
    try:
        res = 1.0 / (1.0 + exp(-yhat))
    except:
        print("yhat: ", yhat)
        sys.exit(-1)
    return res

def store_update(update):
    s3 = boto3.resource('s3')
    key = 'gradient_%d/g_%d' % (np.random.randint(1, 9), random.randint(1, 10))
    datastr = pickle.dumps(update)
    s3.Bucket('camus-pywren-489').put_object(Key=key, Body=datastr)

def get_model():
    s3 = boto3.resource('s3')
    key = 'model'
    obj = s3.Object('camus-pywren-489', key)
    body = obj.get()['Body'].read()
    return pickle.loads(body)

# Map function for pywren main
def gradient_batch(xpys):
    start = time.time()
    model = get_model()
    fetch_model_time = time.time() - start
    xpys = xpys.split(" ")
    reser, upload, model_fetch, model_deser = 0, 0, 0, 0

    for xpy in xpys:
        det = {}
        for mb in get_minis(xpy):
            # Calculate gradient from minibatch
            grad = gradient(mb, model)
            store_update(grad)
            model = get_model()
            #model = update_model(model, grad)
            #store_model(model)
    return "Success!!"

def gradient(train_mb, model):
    global lr
    global MB_SIZE
    cnt = 0
    coef_g = {0: 0}
    gradient = []
    for row in train_mb:
        cnt += 1
        label = row[0]
        values = row[1]

        yhat = predict(values, model)
        error = label - yhat
        coef_g[0] += error * 1.0
        for i in range(len(values)):
            index = values[i][0]
            value = values[i][1]
            coef_g[index + 1] = error * value + coef_g.get(index+1, 0)
        if cnt % MB_SIZE == 0:
            for k in coef_g.keys():
                gradient.append((k, lr * coef_g[k] / float(MB_SIZE)))
            coef_g = {0: 0}

    return gradient

# Calculate accuracy percentage
def logloss_metric(actual, probs):
    total = 0
    for i in range(len(actual)):
        total += (actual[i] * np.log(probs[i])) + ((1 - actual[i]) * np.log(1 - probs[i]))
    return total / len(actual)


# Log loglikelihood func
def loglikelihood(test_data, model):
    actuals = []
    logits = []
    for row in test_data:
        y = row[0]
        yhat = predict(row[1], model)
        actuals.append(y)
        logits.append(yhat)

    return logloss_metric(actuals, logits)

def get_minis(key):
    data = get_data(key)
    idx = 0
    while idx + minibatch_size <= batch_size:
        yield data[idx: idx +minibatch_size]
        idx += minibatch_size


# AWS helper function
def get_data(key, a = False):
    s3 = boto3.resource('s3')
    t0 = time.time()
    obj = s3.Object('camus-pywren-489', key)
    t1 = time.time()
    body = obj.get()['Body'].read()
    data = pickle.loads(body)
    return data

def store_model(model):
    s3 = boto3.resource('s3')
    key = 'model'
    def lamb():
        datastr = pickle.dumps(model)
        s3.Bucket('camus-pywren-489').put_object(Key=key, Body=datastr)
    thread = Thread(target=lamb,)
    thread.start()

index = 1
def get_minibatches(num, over=1):
    global index
    group = []
    for i in range(num):
        if index + batch_file_size > total_batches:
            index = 1
        begin, end = index, index + over
        minis = []
        for b in range(begin, end):
            key = 'mini20:lst-' + str(b)
            minis.append(key)
        index = index + over
        group.append(' '.join(mini for mini in minis))
    print(group)
    return group

def update_model(model, gradient):
    for (k, v) in gradient:
        model[k] += v
    return model


def init_model():
    model = [0.0 for i in range(HASH_SIZE+14 + 1)]
    return model

def get_test_data():
    test_key = "mini20:lst-0"
    test = get_data(test_key)
    return test

def get_local_test_data():
    return get_test()

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


grad_q = queue.Queue()

def fetch_thread(i):
    global outf
    global grad_q
    s3 = boto3.resource('s3')

    my_bucket = s3.Bucket('camus-pywren-489')

    num = 0
    start_time = time.time()
    while time.time() - start_time < total_time:
        lst = my_bucket.objects.filter(Prefix='gradient_%d/' % i).all()
        for object in lst:
            s = time.time()
            obj = object.get()
            grad = pickle.loads(obj['Body'].read())

            grad_q.put(grad)
            object.delete()
            num += 1
            #print("Fetched: %d, took: %f, thread: %d. Sit time: %f" % (num, time.time() - s, i, time.time() - grad[0]['subtime']))
            if time.time() - start_time > total_time:
                return;


def error_thread(model, outf):
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
    last_dump = -1000
    while time.time() - start_time < total_time:

        if not grad_q.empty():
            sz = grad_q.qsize()
            grads = []
            for _ in range(sz):
                grad = grad_q.get()
                model = update_model(model, grad)
                store_model(model)
                grad_q.task_done()
                num += 1
            #model = get_model()
            #error = loglikelihood(test_data, model)
            curr_time = time.time() - start_time
            print("[ERROR_TASK]", curr_time, loglikelihood(test_data, model), "this many grads:", num, "Sec / Grad:", (time.time() - start_time)/ num)
            outf.write("[ERROR_TASK] " +str(curr_time)+ " this many grads: " + str(num) + " Sec / Grad: " + str( (time.time() - start_time)/ num) )
            if True and curr_time - last_dump > 1:
                print("dumping")
                pickle.dump((curr_time, model), f)
                print("dump done")
                last_dump = curr_time
                saves += 1

    print("Saves: ", saves, "Index:", index)
    if True:
        large_test = get_test()
        f.close()
        outf = open(fname[:-4] + ".csv", "w")
        with open(fname[:-4] + ".pkl", 'rb') as f:
            last = -1000
            for i in range(saves):
                t, model = pickle.load(f)
                if t - last < 5:
                    continue
                last = t
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
    futures = start_batch(minibatches)
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

        try:
            pywren.get_all_results(futures)
        except:
            continue

        fin = len(futures)
        iter += fin
        if fin > 0:
            print("Processed: %d" % fin)
            minibatches = get_minibatches(fin)
            futures = start_batch(minibatches)
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

    time.sleep(1)
    model = init_model()
    store_model(model)

    thread = Thread(target=error_thread, args=(model,outf, ))
    fetchers = []

    for i in range(1, 9):
        ft = Thread(target=fetch_thread, args = (i, ))
        ft.start()
        fetchers.append(ft)
    try:
        main(thread, log)
        print(fin)
    except KeyboardInterrupt:
        for ft in fetchers:
            ft.join()

        thread.join()
        if log:
            outf.close()
            print("outf closed by interrupt")
        exit()


    if log:
        print("issued log halt")
        for ft in fetchers:
            ft.join()
        thread.join()
        time.sleep(10)
        outf.close()
        print("outf closed by END")
