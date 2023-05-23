import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

true_hard = ['x1', 'x8', 'x10', 'x4a', 'log(abs(x6))' '(x5)^-1' '(abs(x1))^1/2', 'log(abs(x3))(x3)^-1/2', '(abs(x3))^-1/2']
true_soft = ['x1', 'x3', 'x5', 'x6', 'x8', 'x10', 'x4a']

res_hard, res_soft, res_fphard, res_fpsoft = [], [], [], []
for idx, alpha in enumerate(['0.00000000001', '0.0000000001', '0.000000001', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001', '0.01', '0.1', '1', '5', '10', '25', '50', '100']):
    flist = open(f'feat_art_{alpha}.txt', 'r')
    done = True
    feats = []
    words = []
    for i, line in enumerate(flist):
        #print("NEW LINE")
        line = line.translate({ord(i): None for i in  "\n"})
        stripped = line.translate({ord(i): None for i in  "[]''"})
        if line.startswith('['):
            done = False
        if not done:
            words = words + stripped.split(' ')
            if line.endswith(']'):
                done = True
                feats.append(words)
                #print("PARSING DONE")
                words = []
            else:
                #print("PARSING NOT DONE!")
                done = False


    tp_hard = np.zeros(100)
    tp_soft = np.zeros(100)
    fp_hard = np.zeros(100)
    fp_soft = np.zeros(100)
    for idx, lst in enumerate(feats):
        positives = len(lst)
        for e in lst:
            if e in true_hard: 
                tp_hard[idx] += 1
            else:
                for t in true_soft:
                    if t in e:
                        tp_soft[idx] += 1
            fp_soft[idx] = positives - tp_hard[idx] - tp_soft[idx]
            fp_hard[idx] = positives - tp_hard[idx]
    res_hard.append(tp_hard)
    res_soft.append(tp_soft)
    res_fpsoft.append(fp_soft)
    res_fphard.append(fp_hard)

pow_hard = np.zeros(16)
pow_soft = np.zeros(16)
fp_soft = np.zeros(16)
fp_hard = np.zeros(16)

for i, h, s, fs, fh in enumerate(zip(res_hard, res_soft, res_fpsoft, res_fphard)):
    pow_hard[i] = np.sum(h)/100
    pow_soft[i] = np.sum(s)/100
    fp_soft[i] = np.sum(fs)/100
    fp_hard[i] = np.sum(fh)/100


print(pow_hard, pow_soft, fp_soft)

"""
x = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
xi = list(range(len(x)))

sns.set()
fig, ax = plt.subplots()
#plt.plot(xi, fp10, label = 'FP, N_IAFLAYERS = 10', linestyle = "dashed", marker = 'o')
ax.plot(xi, fp5, label = 'Flow', linestyle = "dashed", marker = 'o')
ax.plot(xi, fp2, label = 'Mean-field', linestyle = "dashed", marker = 'o')
ax.legend()
plt.xticks(xi,x)
ax.set_xlabel(r'$\alpha$', fontsize = 14)
ax.set_ylabel('False Positive', fontsize = 14)
fig.savefig('plot_sim1_corr_mf.png')
plt.close()

"""