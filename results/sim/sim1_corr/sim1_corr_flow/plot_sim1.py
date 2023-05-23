import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

true_hard = ['x5', 'x6', 'x7', 'x8', 'x9', 'x15', 'x16', 'x17', 'x18','x19']
false = ['x0','x1','x2','x3','x4','x10', 'x11','x12', 'x13', 'x14']

res_hard_f, res_soft_f, res_fpr_f, res_fp_f = [], [], [], []
res_hard_g, res_soft_g, res_fpr_g, res_fp_g = [], [], [], []
positives_f = []
positives_g = []
for idx, alpha in enumerate(['0.9','0.8', '0.7', '0.6', '0.5','0.4', '0.3', '0.2', '0.1']):
    flist = open(f'feat_sim1_corr_{alpha}_MF.txt', 'r')
    glist = open(f'feat_sim1_corr_{alpha}_2.txt', 'r')
    done = True
    feats_f = []
    feats_g = []
    words = []
    for i, line in enumerate(flist):
        #print("NEW LINE")
        line = line.translate({ord(e): None for e in  "\n"})
        stripped = line.translate({ord(e): None for e in  "[]''"})
        if line.startswith('['):
            done = False
        if not done:
            words = words + stripped.split(' ')
            if line.endswith(']'):
                done = True
                feats_f.append(words)
                #print("PARSING DONE")
                words = []
            else:
                #print("PARSING NOT DONE!")
                done = False
    for i, line in enumerate(glist):
        #print("NEW LINE")
        line = line.translate({ord(j): None for j in  "\n"})
        stripped = line.translate({ord(j): None for j in  "[]''"})
        if line.startswith('['):
            done = False
        if not done:
            words = words + stripped.split(' ')
            if line.endswith(']'):
                done = True
                feats_g.append(words)
                #print("PARSING DONE")
                words = []
            else:
                #print("PARSING NOT DONE!")
                done = False

    tp_hard_f = np.zeros(100)
    tp_soft_f = np.zeros(100)
    fpr_f = np.zeros(100)
    fp_f = np.zeros(100)
    pos_f = np.zeros(100)
    for i, lst in enumerate(feats_f):
        while("" in lst):
            lst.remove("")
        pos_f[i] = len(lst)
        for e in lst:
            if e in true_hard:
                tp_hard_f[i] += 1/10
            else: 
                fpr_f[i] += 1
            for f in false:
                if f == e:
                    fp_f[i] += 1/10
        
        fp_f[i] = fp_f[i]
        fpr_f[i] = fpr_f[i]/pos_f[i]
    #print(np.sum(tp_hard_f))
    res_hard_f.append(tp_hard_f)
    res_soft_f.append(tp_soft_f)
    res_fp_f.append(fp_f)
    res_fpr_f.append(fpr_f)


    tp_hard_g = np.zeros(100)
    tp_soft_g = np.zeros(100)
    fpr_g = np.zeros(100)
    fp_g = np.zeros(100)
    pos_g = np.zeros(100)
    for i, lst in enumerate(feats_g):
        while("" in lst):
            lst.remove("")
        pos_g[i] = len(lst)
        for e in lst:
            if e in true_hard:
                tp_hard_g[i] += 1/10
            else: 
                fpr_g[i] += 1
            for f in false:
                if f == e:
                    fp_g[i] += 1/10
        
        fp_g[i] = fp_g[i]
        fpr_g[i] = fpr_g[i]/pos_g[i]
    #print(np.sum(tp_hard_f))
    res_hard_g.append(tp_hard_g)
    res_soft_g.append(tp_soft_g)
    res_fp_g.append(fp_g)
    res_fpr_g.append(fpr_g)


pow_hard, pow_hard_lower, pow_hard_upper = np.zeros(9), np.zeros(9), np.zeros(9)
pow_soft, pow_soft_lower, pow_soft_upper = np.zeros(9), np.zeros(9), np.zeros(9)
fp, fp_lower, fp_upper = np.zeros(9), np.zeros(9), np.zeros(9)
fpr, fpr_lower, fpr_upper = np.zeros(9), np.zeros(9), np.zeros(9)

pow_hard_g, pow_hard_lower_g, pow_hard_upper_g = np.zeros(9), np.zeros(9), np.zeros(9)
pow_soft_g, pow_soft_lower_g, pow_soft_upper_g = np.zeros(9), np.zeros(9), np.zeros(9)
fp_g, fp_lower_g, fp_upper_g = np.zeros(9), np.zeros(9), np.zeros(9)
fpr_g, fpr_lower_g, fpr_upper_g = np.zeros(9), np.zeros(9), np.zeros(9)

res_hard_f = np.array(res_hard_f)
res_soft_f = np.array(res_soft_f)
res_fp_f = np.array(res_fp_f)
res_fpr_f = np.array(res_fpr_f)

res_hard_g = np.array(res_hard_g)
res_soft_g = np.array(res_soft_g)
res_fp_g = np.array(res_fp_g)
res_fpr_g = np.array(res_fpr_g)

for i in range(9):
    pow_hard[i] = np.mean(res_hard_f[i])
    pow_soft[i] = np.mean(res_soft_f[i])
    fp[i] = np.mean(res_fp_f[i])
    fpr[i] = np.mean(res_fpr_f[i])

    pow_hard_upper[i] = np.quantile(res_hard_f[i], 0.95)
    pow_soft_upper[i] = np.quantile(res_soft_f[i], 0.95)
    fp_upper[i] = np.quantile(res_fp_f[i], 0.95)
    fpr_upper[i] = np.quantile(res_fpr_f[i], 0.95)

    pow_hard_lower[i] = np.quantile(res_hard_f[i], 0.05)
    pow_soft_lower[i] = np.quantile(res_soft_f[i], 0.05)
    fp_lower[i] = np.quantile(res_fp_f[i], 0.05)
    fpr_lower[i] = np.quantile(res_fpr_f[i], 0.05)

for i in range(9):
    pow_hard_g[i] = np.mean(res_hard_g[i])
    pow_soft_g[i] = np.mean(res_soft_g[i])
    fp_g[i] = np.mean(res_fp_g[i])
    fpr_g[i] = np.mean(res_fpr_g[i])

    pow_hard_upper_g[i] = np.quantile(res_hard_g[i], 0.95)
    pow_soft_upper_g[i] = np.quantile(res_soft_g[i], 0.95)
    fp_upper_g[i] = np.quantile(res_fp_g[i], 0.95)
    fpr_upper_g[i] = np.quantile(res_fpr_g[i], 0.95)

    pow_hard_lower_g[i] = np.quantile(res_hard_g[i], 0.05)
    pow_soft_lower_g[i] = np.quantile(res_soft_g[i], 0.05)
    fp_lower_g[i] = np.quantile(res_fp_g[i], 0.05)
    fpr_lower_g[i] = np.quantile(res_fpr_g[i], 0.05)

x = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
xi = list(range(len(x)))

sns.set()
# plt.plot(xi, pow_hard, label = 'Power', color = 'green')
# plt.plot(xi, pow_hard_lower, linestyle = 'dashed', color = 'green', alpha=0.5)
# plt.plot(xi, pow_hard_upper, linestyle = 'dashed', color = 'green', alpha = 0.5)
# plt.plot(xi, fpr, label = 'Mean-field', color = 'blue')
# plt.plot(xi, fpr_lower, linestyle = 'dashed', color = 'blue', alpha=0.5)
# plt.plot(xi, fpr_upper, linestyle = 'dashed', color = 'blue', alpha = 0.5)
plt.plot(xi, fp, label = 'Mean-field', color = 'blue')
plt.plot(xi, fp_lower, linestyle = 'dashed', color = 'blue', alpha=0.5)
plt.plot(xi, fp_upper, linestyle = 'dashed', color = 'blue', alpha = 0.5)

# plt.plot(xi, pow_hard_g, label = 'Power', color = 'green')
# plt.plot(xi, pow_hard_lower_g, linestyle = 'dashed', color = 'green', alpha=0.5)
# plt.plot(xi, pow_hard_upper_g, linestyle = 'dashed', color = 'green', alpha = 0.5)
# plt.plot(xi, fpr_g, label = 'Flow', color = 'orange')
# plt.plot(xi, fpr_lower_g, linestyle = 'dashed', color = 'orange', alpha=0.5)
# plt.plot(xi, fpr_upper_g, linestyle = 'dashed', color = 'orange', alpha = 0.5)
plt.plot(xi, fp_g, label = 'Flow', color = 'red')
plt.plot(xi, fp_lower_g, linestyle = 'dashed', color = 'red', alpha=0.5)
plt.plot(xi, fp_upper_g, linestyle = 'dashed', color = 'red', alpha = 0.5)
plt.legend()
ax = plt.gca()
ax2 = ax.twiny()
ax.set_xticks(xi)
ax2.set_xticks(xi)
ax.set_xticklabels(x)
ax2.set_xticklabels([0.9940,0.9706,0.9202,0.8334,0.7081,0.5543,0.3914,0.2378,0.1039],rotation =40)
ax2.set_xlim([-0.4, 8.4])
plt.tight_layout()
plt.ylim([-0.01,0.11])
plt.savefig('plot_sim1_corr.png')

x = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
xi = list(range(len(x)))

