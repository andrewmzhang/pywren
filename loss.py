import time
import os
import json
config = {
    "lr": 0.0001,
    "total_time": 500,
}

for lr in [ 0.0005 ]:
    fname = "tmp-%f-loss.txt" % lr
    config['fname'] = fname
    config['lr'] = lr
#    if lr == 0.0005:
#        config['total_time'] = 1200
#    else:
#        config['total_time'] = 600
    cmd = "python3 a.py '%s'" % json.dumps(config)
    print(cmd)
    os.system(cmd)
    os.system("rm *.pkl")
