def var_inference(values, comps, iteration, verbose = False):
    tmpx = torch.tensor(np.column_stack((values.T, y_train)), dtype=torch.float32).to(DEVICE)

    data_mean = tmpx.mean(axis=0)[0:-1]
    data_std = tmpx.std(axis=0)[0:-1]
    tmpx[:,0:-1] = (tmpx[:,0:-1] - data_mean) / data_std

    prior_a = torch.tensor(np.exp(-comps - 0.5), dtype=torch.float32)
    p = tmpx.shape[1]
    net = BayesianNetwork(p, prior_a).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(epochs):
        nll, loss = train(net, optimizer, tmpx, p, verbose=verbose)
        print('epoch =', epoch)
        print(f'generation: {iteration}')

    a = net.l1.alpha_q.data.detach().cpu().numpy().squeeze()
    return a, net