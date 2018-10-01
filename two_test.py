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
from threading import Barrier
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
total_batches = 2500   # Entire number of batches in dataset
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
        #print(index)
        yhat += coefficients[index + 1] * value
    try:
        res = 1.0 / (1.0 + exp(-yhat))
    except:
        print("yhat: ", yhat)
        sys.exit(-1)
    return res

def store_update(update, number):
    s3 = boto3.resource('s3')
    key = 'gradient_indiv_%d' % (number)
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
    xpys, number = xpys
    start = time.time()
    fetch_model_time = time.time() - start
    xpys = xpys.split(" ")

    for xpy in xpys:
        det = {}
        for mb in get_minis(xpy):
            # Calculate gradient from minibatch
            #print(mb)
            model = get_model()
            grad = gradient(mb, model)
            store_update(grad, number)
            #model = update_model(model, grad)
            #store_model(model)
            begin = time.time()
            while check_key('gradient_indiv_%d' % (number)) and time.time() - begin < 10:
                pass
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

# Reduce function
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
def get_minibatches(num, over=25):
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
    f = open("testset.data", "rb")
    f.seek(0)
    x_dense_test, x_idx_test, y_test = pickle.load(f)
    x_sparse_test = sparse.lil_matrix((x_dense_test.shape[0], HASH))
    for i in range(x_dense_test.shape[0]):
        x_sparse_test[i, x_idx_test[i]] = np.ones(len(x_idx_test[i]))
    f.close()
    return (x_dense_test, x_sparse_test, y_test)

def start_batch(minibatches):
    print(minibatches)
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

def check_key(key):
    s3 = boto3.resource('s3')
    try:
        s3.Object('camus-pywren-489', key).load()
    except:
        return False
    return True

b = Barrier(num_lambdas, timeout=300)
def fetch_thread(i):
    global outf
    global grad_q
    global b
    s3 = boto3.resource('s3')

    my_bucket = s3.Bucket('camus-pywren-489')
    test = get_test()
    num = 0
    start_time = time.time()
    while time.time() - start_time < total_time:
        key = 'gradient_indiv_%d' % i
        begin = time.time()
        while time.time() - start_time < total_time and (not check_key(key)):
            if time.time() - begin > 10:
                print("Thread %d took too long" % i)
                b.wait()
                break
            pass
        obj = my_bucket.Object('gradient_indiv_%d' % i)
        try:
            grad = pickle.loads(obj.get()['Body'].read())
        except:
            continue

        grad_q.put(grad)
        #model = get_model()
        #model = update_model(model, grad)
        #store_model(model)
        print("Thread %d waiting..." % i)
        b.wait()
        if num % 10 == 0:
            print('[ERROR]', num, time.time() - start_time, loglikelihood(test, model))
        print("Thread %d moving..." % i)
        obj.delete()
        #if i == 0:
        #    b.reset()

        num += 1
        if time.time() - start_time > total_time:
            return;


def dump_thread(q, f):
    global total_time
    start = time.time()
    print("DUMP THREAD STARTED")
    outf = open(fname[:-4] + ".csv2", "w")
    testdata = get_test()
    while time.time() - start < total_time or not q.empty():
        if time.time() - start > total_time and q.empty():
            break;
        if not q.empty():
            t, model = q.get()
            print("dumping")
            s = time.time()
            #pickle.dump(time_model, f)
            loss = loglikelihood(testdata, model)
            print("\033[92m wrote: %f %f \033[0m" % (t, loss))
            outf.write("%f, %f\n" % (t, loss))
            print("dump done took", time.time() - s)
            q.task_done()
    outf.close()
    print("DUMP THREAD STOPPED")






def error_thread(model, outf):
    global grad_q
    global log
    global fname
    global index



    num = 0
    print("Starting error thread")
    start_time = time.time()
    # Clear existing gradients

    #test_data = get_test_data()

    saves = 0

    print(fname[:-4] + ".pkl")
    f = open(fname[:-4] + ".pkl", 'wb')
    time_model_q = queue.Queue()
    dump_t = Thread(target=dump_thread, args=(time_model_q, f, ))
    dump_t.start()

    last_dump = -1000
    print("[ERROR TASK STARTING]")
    while time.time() - start_time < total_time:
        if not grad_q.empty():
            grad = grad_q.get()
            s = time.time()
            model = update_model(model, grad)
            print("Updating took", time.time() - s)
            s = time.time()
            def up():
                store_model(model)
            up_thread = Thread(target=up)
            up_thread.start()
            print("Store took", time.time() - s)
            grad_q.task_done()
            num += 1

            #model = get_model()
            #error = loglikelihood(test_data, model)
            curr_time = time.time() - start_time
            print("[ERROR_TASK]", curr_time, 0, "this many grads:", num, "Sec / Grad:", (time.time() - start_time)/ num)
            #outf.write("[ERROR_TASK] " +str(curr_time)+ " this many grads: " + str(num) + " Sec / Grad: " + str( (time.time() - start_time)/ num) )

            if True and curr_time - last_dump > 1:
                s = time.time()
                #print("dumping into thread")
                time_model_q.put((curr_time, model[:]))
                #print("dump into thread", time.time() - s)
                last_dump = curr_time
                saves += 1

    print("Saves: ", saves, "Index:", index)
    dump_t.join()
    print("Dumpt is good")
    f.close()
    if False:
        large_test = get_test_data()
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
    minibatches = list(zip(minibatches, range(len(minibatches))))
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

        pywren.get_all_results(futures)

        fin = len(futures)
        iter += fin
        if fin > 0:
            print("Processed: %d" % fin)
            minibatches = get_minibatches(fin)
            minibatches = list(zip(minibatches, range(len(minibatches))))
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

    for i in range(0, 10):
        string = "gradient_invid_%d/" % i
        for object in my_bucket.objects.filter(Prefix=string).all():
            object.delete()

    time.sleep(1)
    model = init_model()
    store_model(model)

    thread = Thread(target=error_thread, args=(model,outf, ))
    fetchers = []

    for i in range(0, num_lambdas):
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
