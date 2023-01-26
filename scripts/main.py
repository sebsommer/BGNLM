# TRAINING:
test_acc = []
for i in range(N_GENERATIONS):
    # VI step:
    train_values = np.array([F0[k]['values'] for k in range(POPULATION_SIZE)])
    complexities = np.array([F0[k]['complexity'] for k in range(POPULATION_SIZE)])
    mprobs, net = var_inference(train_values, complexities, i, verbose=False)
    population = np.array([F0[k]['feature'] for k in range(POPULATION_SIZE)])

    for id, _ in F0.items():
        F0[id]['mprob'] = mprobs[id]
        if F0[id]['mprob'] < 0.5:
            while (True):
                feat, train_vals, test_vals, obj = generate_feature(population, train_values.T, fprobs, mprobs / np.sum(mprobs), tprobs, test_values)
                comp = get_complexity(feat)
                collinear = check_collinearity(train_vals, train_values)

                if comp < COMPLEXITY_MAX and np.all(np.isfinite(train_vals)) and not collinear:
                    F0[id]['complexity'] = comp
                    F0[id]['feature'] = feat
                    F0[id]['values'] = train_vals
                    F0[id]['type'] = obj
                    #if opt_strategy == 3 and isinstance(obj, Projection):
                    #    obj.optimize(feat, train_vals, strategy=3)
                    test_values[:, id] = test_vals
                    break
                continue
    #TESTING FOR CURRENT POPULATION:
    test_mean = test_values.mean(axis=0)
    test_std = test_values.std(axis=0)
    test_values = np.divide((test_values - test_mean), test_std)

    test_data = torch.tensor(np.column_stack((test_values, y_test)), dtype=torch.float32).to(DEVICE)
    accuracy = test(net, test_data, p = POPULATION_SIZE+1)
    test_acc.append(accuracy)

# printing final population and test results:
for key, v in F0.items():
    f, m, c = v['feature'], v['mprob'], v['complexity']
    m = np.round(float(m),3)
    t = type(v['type']).__name__
    print("ID: {:<3}, Feature: {:<60}, mprob:{:<15}, complexity: {:<5}, type: {:<10}".format(key,f,m,c,t))

print('\n\nTEST RESULTS:\n----------------------')
print('FINAL GENERATION ACCURACY:', accuracy)
print('MEAN TEST ACCURACY: ', np.mean(test_acc))
print('BEST ACCURACY:', np.max(test_acc), f'(generation {test_acc.index(max(test_acc))})')