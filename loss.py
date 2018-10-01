import time
import os
import json
config = {
    "lr": 0.0001,
    "total_time": 500,
}



programs = ['', 'one_test.py', 'two_test.py', 'three_test.py', 'four.py', 'five.py']
#  (0.0005,1), (0.0005,2), (0.0005,3),
for a in [(0.001,2)]:
    lr, it = a
    fname = "tmp-%f-loss-%d.txt" % (lr, it)
    config['fname'] = fname
    config['lr'] = lr
    config['total_time'] = 1200
    cmd = "python3 %s '%s' " % (programs[it], json.dumps(config))
    print(cmd)
    os.system(cmd)
