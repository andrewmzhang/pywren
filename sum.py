import sys

datas = {}

fname = sys.argv[1]
print("opening", fname)
with open(fname, 'r') as f:

    for line in f:
        label, value = (line.rstrip()).split(" ")
        try:
            value = float(value)
        except:
            continue;
        if label not in datas.keys():
            datas[label] = [value]
        else:
            datas[label].append(value)


for key in datas.keys():
    print key
    data = datas[key]
    print sum(data) / len(data)
