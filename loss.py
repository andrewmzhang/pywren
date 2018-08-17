import os
import json
config = {
    "lr": 0.0001,
    "total_time": 30,
}


for lr in [ 0.00001, 0.000025, 0.00005, 0.000075, 0.0001]:
    fname = "tmp-%f-loss.txt" % lr
    config['fname'] = fname
    cmd = "python3 a.py '%s'" % json.dumps(config)
    print(cmd)
    os.system(cmd)
