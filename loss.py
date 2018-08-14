import os


for lr in [ 0.00001, 0.000025, 0.00005, 0.000075, 0.0001]:
    fname = "tmp-%f-loss.txt" % lr
    os.system("python3 a.py %s %f" % (fname, lr))
