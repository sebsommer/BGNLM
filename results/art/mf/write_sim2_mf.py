import sys

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

true_hard = ['x1', 'x8', 'x10', 'x4a', 'log(abs(x6))', '(x5)^-1', '(abs(x1))^1/2', 'log(abs(x3))(x3)^-1/2', '(abs(x3))^-1/2']
true_soft = ['x1', 'x3', 'x5', 'x6', 'x8', 'x10', 'x4a']
false = ['x2', 'x7', 'x4b', 'x9a', 'x9b']

alphas = ['0.0000000001', '0.000000001', '0.00000001', '0.0000001', '0.000001', '0.00001', '0.0001', '0.001', '0.01', '0.1', '1', '5', '10', '25', '50', '100']
alphas.reverse()

res_hard, res_soft, res_fphard, res_fpsoft = [], [], [], []
positives = []
for idx, alpha in enumerate(alphas):
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
    pos = np.zeros(100)
    for i, lst in enumerate(feats):
        while("" in lst):
            lst.remove("")
        pos[i] = len(lst)
        checked = []
        for e in lst:
            if e in true_hard:
                tp_hard[i] += 1/9
            else: 
                fp_hard[i] += 1
            for ts in true_soft:
                if ts in e and ts not in checked:
                    tp_soft[i] += 1/7
                    checked.append(ts)
            for f in false:
                if f in e:
                    fp_soft[i] +=1
        
        fp_soft[i] = fp_soft[i]/pos[i]
        fp_hard[i] = fp_hard[i]/pos[i]
    #print(np.sum(tp_hard))
    res_hard.append(tp_hard)
    res_soft.append(tp_soft)
    res_fpsoft.append(fp_soft)
    res_fphard.append(fp_hard)



pow_hard, pow_hard_lower, pow_hard_upper = np.zeros(16), np.zeros(16), np.zeros(16)
pow_soft, pow_soft_lower, pow_soft_upper = np.zeros(16), np.zeros(16), np.zeros(16)
fp_soft, fp_soft_lower, fp_soft_upper = np.zeros(16), np.zeros(16), np.zeros(16)
fp_hard, fp_hard_lower, fp_hard_upper = np.zeros(16), np.zeros(16), np.zeros(16)



res_hard = np.array(res_hard)
res_soft = np.array(res_soft)
res_fpsoft = np.array(res_fpsoft)
res_fphard = np.array(res_fphard)



for i in range(16):
    pow_hard[i] = np.mean(res_hard[i])
    pow_soft[i] = np.mean(res_soft[i])
    fp_soft[i] = np.mean(res_fpsoft[i])
    fp_hard[i] = np.mean(res_fphard[i])

    pow_hard_upper[i] = np.quantile(res_hard[i], 0.95)
    pow_soft_upper[i] = np.quantile(res_soft[i], 0.95)
    fp_soft_upper[i] = np.quantile(res_fpsoft[i], 0.95)
    fp_hard_upper[i] = np.quantile(res_fphard[i], 0.95)

    pow_hard_lower[i] = np.quantile(res_hard[i], 0.05)
    pow_soft_lower[i] = np.quantile(res_soft[i], 0.05)
    fp_soft_lower[i] = np.quantile(res_fpsoft[i], 0.05)
    fp_hard_lower[i] = np.quantile(res_fphard[i], 0.05)

x = [100, 50, 25, 10, 5, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]
xi = list(range(len(x)))

print(list(pow_hard), "\n", list(pow_hard_lower),"\n", list(pow_hard_upper), "\n")
print(list(pow_soft), "\n",list(pow_soft_lower), "\n", list(pow_soft_upper), "\n")
print(list(fp_soft), "\n",list(fp_soft_lower), "\n",list(fp_soft_upper),"\n")
print(list(fp_hard), "\n",list(fp_hard_lower), "\n",list(fp_hard_upper),"\n")
"""sns.set()
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
axes = axes.flatten()
#plt.plot(xi, fp10, label = 'FP, N_IAFLAYERS = 10', linestyle = "dashed", marker = 'o')

for i in range(4):
    plt.sca(axes[i])
    plt.xticks(xi,x, rotation = 70, fontsize = 9)
    plt.ylim([-0.05,1.05])


    

axes[0].set_ylabel('Power (strict)', fontsize = 12)
axes[0].plot(xi, pow_hard, label = 'Flow', color = 'blue')
axes[0].plot(xi, pow_hard_lower, linestyle = 'dashed', color = 'blue', alpha=0.5)
axes[0].plot(xi, pow_hard_upper, linestyle = 'dashed', color = 'blue', alpha = 0.5)
axes[0].plot(xi, [0.0214,0.0957,0.1457,0.15,0.1571,0.5071,0.6128,0.6828,0.7142,0.7142,0.7142,0.7142,0.7142,0.7142,0.7142,0.7142], linestyle = 'dashed', color = 'orange', alpha = 0.5)
axes[0].plot(xi, [0.0071,0.07,0.1357,0.1357,0.1428,0.4657,0.59,0.6571,0.7142,0.7142,0.7142,0.7142,0.7142,0.7142,0.7142,0.7142], linestyle = 'dashed', color = 'orange', alpha = 0.5)
axes[0].plot(xi, [0.0143,0.0842,0.1414,0.1428,0.15,0.4871,0.6014,0.6714,0.7142,0.7142,0.7142,0.7142,0.7142,0.7142,0.7142,0.7142], label = 'MFP', color = 'orange')
axes[1].set_ylabel('Power (soft)', fontsize = 12)
axes[1].plot(xi, pow_soft, label = 'Flow', color = 'blue')
axes[1].plot(xi, pow_soft_lower, linestyle = 'dashed', color = 'blue', alpha=0.5)
axes[1].plot(xi, pow_soft_upper, linestyle = 'dashed', color = 'blue', alpha = 0.5)
axes[1].plot(xi, [0.0742,0.2185,0.3528,0.4442,0.47,0.8585,1,1,1,1,1,1,1,1,1,1], linestyle = 'dashed', color = 'orange', alpha = 0.5)
axes[1].plot(xi, [0.04,0.182857142857143,0.324285714285714,0.43,0.442857142857143,0.825714285714286,1,1,1,1,1,1,1,1,1,1], linestyle = 'dashed', color = 'orange', alpha = 0.5)
axes[1].plot(xi, [0.0585714285714286,0.201428571428571,0.338571428571429,0.437142857142857,0.455714285714286,0.841428571428571,1,1,1,1,1,1,1,1,1,1], label = 'MFP', color = 'orange')
axes[2].set_ylabel('FDR (hard)', fontsize = 12)
axes[2].plot(xi, fp_hard, label = 'Flow', color = 'blue')
axes[2].plot(xi, fp_hard_lower, linestyle = 'dashed', color = 'blue', alpha=0.5)
axes[2].plot(xi, fp_hard_upper, linestyle = 'dashed', color = 'blue', alpha = 0.5)
axes[2].plot(xi, [0.8889,0.6728,0.6552,0.7635,0.7607,0.5343,0.48375,0.425,0.375,0.375,0.375,0.375,0.375,0.375,0.375,0.375], linestyle = 'dashed', color = 'orange', alpha = 0.5)
axes[2].plot(xi, [0.6842,0.5620,0.6031,0.7419,0.7387,0.4957,0.46125,0.4025,0.375,0.375,0.375,0.375,0.375,0.375,0.375,0.375], linestyle = 'dashed', color = 'orange', alpha = 0.5)
axes[2].plot(xi, [0.7872,0.6169,0.6333,0.7542,0.75,0.5142,0.4737,0.4125,0.375,0.375,0.375,0.375,0.375,0.375,0.375,0.375], label = 'MFP', color = 'orange')
axes[3].set_ylabel('FDR (soft)', fontsize = 12)
axes[3].plot(xi,fp_soft, label = 'Flow', color = 'blue')
axes[3].plot(xi, fp_soft_lower, linestyle = 'dashed', color = 'blue', alpha=0.5)
axes[3].plot(xi, fp_soft_upper, linestyle = 'dashed', color = 'blue', alpha = 0.5)
axes[3].plot(xi, [0.1632,0.0662,0.0246,0.0223,0.0244,0.0422,0,0,0,0,0,0,0,0,0,0], linestyle = 'dashed', color = 'orange', alpha=0.5)
axes[3].plot(xi, [0.0652,0.0270,0.0083,0.0065,0.0062,0.0213,0,0,0,0,0,0,0,0,0,0], linestyle = 'dashed', color = 'orange', alpha = 0.5)
axes[3].plot(xi,[0.1632,0.0662,0.0246,0.0223,0.0244,0.0422,0,0,0,0,0,0,0,0,0,0], label = 'MFP', color = 'orange')

axes[0].legend()
plt.tight_layout()
fig.savefig('plot_sim2.png')
plt.close()"""
