import time
import os
import json
config = {
    "lr": 0.0001,
    "total_time": 600,
}


for lr in [ 0.0005, 0.00001, 0.00005]:
    fname = "tmp-%f-loss.txt" % lr
    config['fname'] = fname
    config['lr'] = lr
    cmd = "python3 a.py '%s'" % json.dumps(config)
    print(cmd)
    os.system(cmd)
