import time
import os
import json
config = {
    "lr": 0.0001,
    "total_time": 600,
}
# 0.00009 0.00004 0.00003

for lr in [ 0.00008]:
    fname = "tmp-%f-loss.txt" % lr
    config['fname'] = fname
    config['lr'] = lr
    cmd = "python3 a.py '%s'" % json.dumps(config)
    print(cmd)
    os.system(cmd)
    os.system("rm *.pkl")
