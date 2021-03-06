# Logistic Regression on Diabetes Dataset
from random import seed
from random import randrange
from csv import reader
from math import exp
import time
import mmh3
import sys
import numpy as np
HASH_SIZE = 100000
MB_SIZE = 20
# result is a list of tuples (label, list of tuples (index, value))
# Load a CSV file
def load_csv_sparse_tab(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file, dialect="excel-tab")
        for row in csv_reader:
            tuples_list = list()

            # add numerical values to list
            for i in range(1, 14):
                if row[i] == '':
                    row[i] = "0"
                tuples_list.append((i-1, float(row[i])))

            global HASH_SIZE
            dic = dict()
            for i in range(14, len(row)):
                if row[i] == '':
                    continue
                hashed = mmh3.hash(row[i], 42)
                hashed = hashed % HASH_SIZE + 14

                if hashed in dic:
                    dic[hashed] += 1
                else:
                    dic[hashed] = 1

            for i in dic.keys():
                tuples_list.append((i, float(dic[i])))

            if len(tuples_list) == 0:
                print("ERROR")

            row_entry = (float(row[0]), tuples_list)
            dataset.append(row_entry)
    #print(dataset)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        if row[column] == '':
            row[column] = "0"
        row[column] = float(row[column].strip())

# Find the min and max values for each column
def dataset_minmax(dataset):
    #minmax = list()
    col_values = dict()
    col_counts = dict()
    col_min_max = dict()

    for row in dataset:
        row_values = row[1]
        for i in range(len(row_values)):
            index = row_values[i][0]
            value = row_values[i][1]

            if index in col_values:
                col_values[index] += value
                col_counts[index] += 1
                col_min_max[index] = ( \
                        min(col_min_max[index][0], value),\
                        max(col_min_max[index][1], value))
            else:
                col_values[index] = value
                col_counts[index] = 1
                #print("Inserting min max on index ", index, " value: ", value)
                col_min_max[index] = (value, value)

    #print("col_min_max: ", col_min_max)
    return col_min_max

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        row_values = row[1]
        for i in range(len(row_values)):
            index = row_values[i][0]
            value = row_values[i][1]
            if index > 14:
                continue

            if minmax[index][0] == minmax[index][1]:
                continue
            new_value = (value - minmax[index][0]) / (minmax[index][1] - minmax[index][0])

            row_values[i] = (row_values[i][0], new_value)

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Calculate accuracy percentage
def logloss_metric(actual, probs):
    total = 0
    for i in range(len(actual)):
        total += (actual[i] * np.log(probs[i])) + ((1 - actual[i]) * np.log(1 - probs[i]))
    return total / len(actual)

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    print("evaluate_algorithm")
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        start_time = time.time()
        print "Evaluating new fold (out of %d)" % len(folds)
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            #row_copy[-1] = None
        predicted = algorithm(train_set, fold, toround=False, *args)
        actual = [row[0] for row in fold]
        loss = logloss_metric(actual, predicted)
        print("Loss:", loss)
        scores.append(loss)
        print("Elapsed time fold: ", time.time() - start_time)
    return scores

# Make a prediction with coefficients
def predict(row_values, coefficients):
    #print("row: ", row)
    #print("coefficients: ", coefficients)
    yhat = coefficients[0]
    for i in range(len(row_values)):

        #print("row_values:", row_values)
        #try:
        index = row_values[i][0]
        value = row_values[i][1]
        #except:
            #print("row_values:", row_values)
            #sys.exit(-1)

        yhat += coefficients[index + 1] * value
    try:
        res = 1.0 / (1.0 + exp(-yhat))
    except:
        print("yhat: ", yhat)
        sys.exit(-1)
    return res

# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, test, l_rate, n_epoch):
    global HASH_SIZE
    global MB_SIZE
    coef = [0.0 for i in range(HASH_SIZE+14 + 1)] # +1 for the biasa
    errors = []
    for epoch in range(n_epoch):
        print("epoch: ", epoch)
        cnt = 0
        coef_g = {0: 0} # +1 for the bias
        for row in train:
            cnt += 1
            start = time.time()
            label = row[0]
            values = row[1]

            yhat = predict(values, coef)
            error = label - yhat
            #errors.append(error)
            coef_g[0] += error * 1.0
            for i in range(len(values)):
                index = values[i][0]
                value = values[i][1]
                coef_g[index + 1] = error * value + coef_g.get(index+1, 0)


            if cnt % MB_SIZE == 0:
                for k in coef_g.keys():
                    coef[k] += l_rate * coef_g[k] / float(MB_SIZE)
                errors = []
                coef_g = {0: 0}
            if cnt % MB_SIZE == 0:
                print("Iterno", cnt / MB_SIZE)
                logits = []
                for row in test:
                    label = row[0]
                    values = row[1]
                    yhat = predict(values, coef)
                    logits.append(yhat)
                actual = [row[0] for row in test]
                print("logloss", logloss_metric(actual, logits)) 
    return coef

# Linear Regression Algorithm With Stochastic Gradient Descent
def logistic_regression(train, test, l_rate, n_epoch, toround=True):
    predictions = list()
    coef = coefficients_sgd(train, test, l_rate, n_epoch)
    for row in test:
        #print("test row: ", row)
        label = row[0]
        values = row[1]

        yhat = predict(values, coef)
        if toround:
            yhat = round(yhat)
        predictions.append(yhat)
    return(predictions)


# Test the logistic regression algorithm on the diabetes dataset
seed(1)
# load and prepare data
#filename = 'day_1_1K.csv'
#filename = 'filtered_data_shuffled.csv'
filename = 'train_sample.txt'

print("Loading dataset")
dataset = load_csv_sparse_tab(filename)

#print("Processing dataset")
#for i in range(len(dataset[0])):
#    str_column_to_float(dataset, i)

# it's already normalized
# normalize
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
print(dataset[0:10])


# evaluate algorithm
n_folds = 10
l_rate = 0.01
n_epoch = 10

print("Evaluating algorithm")
scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
