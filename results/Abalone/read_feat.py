import sys

import matplotlib.pyplot as plt
import numpy as np

res = []

flist = open(f'feat3AIC_test.txt', 'r')
done = True
feats = {}
words = []
for i, line in enumerate(flist):
    #print("NEW LINE")
    line = line.translate({ord(i): None for i in  "\n"})
    stripped = line.translate({ord(i): None for i in  "[]"})
    if line.startswith('['):
        done = False
    if not done:
        words = words + stripped.split(' ')
        if line.endswith(']'):
            done = True
            for e in words:
                e = e.translate({ord(i): None for i in  "''"})
                if e in feats:
                    feats[e] += 1
                else: 
                    feats[e] = 1
            #print("PARSING DONE")
            words = []
        else:
            #print("PARSING NOT DONE!")
            done = False
    


count = 0
for key, val in feats.items():
    if 'Male' in key and 'Female' in key and 'ShuckedWeight' in key and not '^' in key and not 'Diameter' in key:
        count += val
        feats[key] = 0
    if val > 100:
        feats[key] = 0
feats = {x:y for x,y in feats.items() if y != 0}
feats['a + b*Male + c*Female + d*ShuckedWeight'] = count

import operator
feats = dict(sorted(feats.items(), key=operator.itemgetter(1),reverse=True))

count = 0
for key, val in feats.items():
    if val > 10:
        print(key, ":",val)
        count+=1

print(count)
