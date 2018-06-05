import pywren
import numpy as np
import math
import boto3
import pickle



def my_function(x):
    return x + 7

wrenexec = pywren.default_executor()
futures = wrenexec.map(my_function, range(10))
pywren.get_all_results(futures)
