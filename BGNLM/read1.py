import numpy as np

A, B, C = [], [], []

with open('results3BIC_test.txt', 'r') as f:
    f.readline()
    f.readline()
    f.readline()
    Lines = f.readlines()
    for line in Lines:
        line = line.split()
        acc = float(line[0])
        fnr = float(line[1])
        fpr = float(line[2])
        A.append(acc)
        B.append(fnr)
        C.append(fpr)
    
    f.close()

with open('results_final.txt', 'a') as f:
    f.write('Results for 3 (BIC):\n-------------------------------------------------------------------------------------------------\n')
    f.write(f'RMSE: {np.round(np.median(A), 5)}, ({np.round(np.min(A), 5)}, {np.round(np.max(A), 5)}), MAE: {np.round(np.median(B), 5)}, ({np.round(np.min(B), 5)}, {np.round(np.max(B), 5)}), Corr: {np.round(np.median(C), 5)}, ({np.round(np.min(C), 5)}, {np.round(np.max(C), 5)})\n-------------------------------------------------------------------------------------------------\n\n')

    f.close()