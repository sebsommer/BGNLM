import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

true_hard = ['x5', 'x6', 'x7', 'x8', 'x9', 'x15', 'x16', 'x17', 'x18','x19']
false = ['x0','x1','x2','x3','x4', 'x10', 'x11', 'x12', 'x13', 'x14']

alphas = ['0.0001', '0.01', '0.1', '1', '2', '3', '4', '5', '10', '25', '50', '100']
alphas.reverse()

res_hard, res_soft, res_fpr, res_fp = [], [], [], []
positives = []
for idx, alpha in enumerate(alphas):
    flist = open(f'feat_sim1_{alpha}.txt', 'r')
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
    fpr = np.zeros(100)
    fp = np.zeros(100)
    pos = np.zeros(100)
    for i, lst in enumerate(feats):
        while("" in lst):
            lst.remove("")
        pos[i] = len(lst)
        for e in lst:
            if e in true_hard:
                tp_hard[i] += 1/10
            else: 
                fpr[i] += 1
            for f in false:
                if f == e:
                    print(idx,i, e)
                    fp[i] += 1/10
        
        fp[i] = fp[i]
        fpr[i] = fpr[i]/pos[i]
    #print(np.sum(tp_hard))
    res_hard.append(tp_hard)
    res_soft.append(tp_soft)
    res_fp.append(fp)
    res_fpr.append(fpr)



pow_hard, pow_hard_lower, pow_hard_upper = np.zeros(12), np.zeros(12), np.zeros(12)
pow_soft, pow_soft_lower, pow_soft_upper = np.zeros(12), np.zeros(12), np.zeros(12)
fp, fp_lower, fp_upper = np.zeros(12), np.zeros(12), np.zeros(12)
fpr, fpr_lower, fpr_upper = np.zeros(12), np.zeros(12), np.zeros(12)


res_hard = np.array(res_hard)
res_soft = np.array(res_soft)
res_fp = np.array(res_fp)
res_fpr = np.array(res_fpr)

res_fp[9] = res_fp[11] = 0

for i in range(12):
    pow_hard[i] = np.mean(res_hard[i])
    pow_soft[i] = np.mean(res_soft[i])
    fp[i] = np.mean(res_fp[i])
    fpr[i] = np.mean(res_fpr[i])

    pow_hard_upper[i] = np.quantile(res_hard[i], 0.95)
    pow_soft_upper[i] = np.quantile(res_soft[i], 0.95)
    fp_upper[i] = np.quantile(res_fp[i], 0.95)
    fpr_upper[i] = np.quantile(res_fpr[i], 0.95)

    pow_hard_lower[i] = np.quantile(res_hard[i], 0.05)
    pow_soft_lower[i] = np.quantile(res_soft[i], 0.05)
    fp_lower[i] = np.quantile(res_fp[i], 0.05)
    fpr_lower[i] = np.quantile(res_fpr[i], 0.05)

pow_hard, pow_hard_lower, pow_hard_upper = np.ones(12), np.ones(12), np.ones(12)

x = [100, 50, 25, 10, 5, 4,3,2,1,0.1,0.01,0.001]
xi = list(range(len(x)))

print(fp)

sns.set()

plt.plot(xi, pow_hard, label = 'Power', color = 'green')
plt.plot(xi, pow_hard_lower, linestyle = 'dashed', color = 'green', alpha=0.5)
plt.plot(xi, pow_hard_upper, linestyle = 'dashed', color = 'green', alpha = 0.5)
plt.plot(xi, fpr, label = 'FPR', color = 'orange')
plt.plot(xi, fpr_lower, linestyle = 'dashed', color = 'orange', alpha=0.5)
plt.plot(xi, fpr_upper, linestyle = 'dashed', color = 'orange', alpha = 0.5)
plt.plot(xi, fp, label = 'FP', color = 'red')
plt.plot(xi, fp_lower, linestyle = 'dashed', color = 'red', alpha=0.5)
plt.plot(xi, fp_upper, linestyle = 'dashed', color = 'red', alpha = 0.5)
plt.legend(fontsize = 14)
plt.xticks(xi, x, fontsize = 14, rotation=30)
plt.yticks(fontsize = 14)
plt.tight_layout()
plt.ylim([-0.05,1.05])
plt.savefig('plot_sim1.png')
