import pywren
import time


def my_function(x):
    return time.time()


a = time.time()
wrenexec = pywren.default_executor()
futures = wrenexec.map(my_function, range(10))

[print(t - a) for t in pywren.get_all_results(futures)]
