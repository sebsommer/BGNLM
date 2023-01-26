N_GENERATIONS = 20
EXTRA_FEATURES = 10
POPULATION_SIZE = x_train.shape[1] + EXTRA_FEATURES
COMPLEXITY_MAX = 10

# strategy for optimizing projections:
opt_strategy = None

# initializing first generation:
if POPULATION_SIZE > x_test.shape[1]:
    test_values = np.zeros((x_test.shape[0], POPULATION_SIZE))
    for i in range(x_test.shape[1]):
        test_values[:,i] = x_test[:,i]  
else:    
    test_values = x_test

F0 = {}
for j in range(x_train.shape[1]):
    F0[j] = {}
    F0[j]['feature'] = f'x{j}'
    F0[j]['mprob'] = 0
    F0[j]['complexity'] = 0
    F0[j]['values'] = x_train[:, j]
    F0[j]['type'] = Linear(f'x{j}')

for j in range(x_train.shape[1], POPULATION_SIZE):
    F0[j] = {}
    mprob = 1/x_train.shape[1]
    population = np.array([F0[k]['feature'] for k in range(x_train.shape[1])])
    train_values = np.array([F0[k]['values'] for k in range(x_train.shape[1])])
    complexities = np.array([F0[k]['complexity'] for k in range(x_train.shape[1])])
    while(True):
        feat, train_vals, test_vals, obj = generate_feature(population, train_values.T, fprobs, [mprob for _ in range(x_train.shape[1])], tprobs, x_test)
        comp = get_complexity(feat)
        collinear = check_collinearity(train_vals, train_values)
        if comp < COMPLEXITY_MAX and np.all(np.isfinite(train_vals)) and not collinear:
            F0[j]['mprob'] = mprob
            F0[j]['complexity'] = comp
            F0[j]['feature'] = feat
            F0[j]['values'] = train_vals
            F0[j]['type'] = obj
            test_values[:,j] = test_vals
            break
        continue
