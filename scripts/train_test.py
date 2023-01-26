def train(net, optimizer, dtrain, p, batch_size=BATCH_SIZE,verbose=False):
    net.train()
    old_batch = 0
    accs = []
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = dtrain[old_batch: batch_size * batch, 0:p - 1]
        _y = dtrain[old_batch: batch_size * batch, -1]
        old_batch = batch_size * batch
        target = Variable(_y).to(DEVICE)
        data = Variable(_x).to(DEVICE)
        net.zero_grad()
        outputs = net(data, sample=True)
        target = target.unsqueeze(1).float()
        negative_log_likelihood = net.loss(outputs, target)
        loss = negative_log_likelihood + net.kl() / NUM_BATCHES
        loss.backward()
        optimizer.step()
        pred = outputs.squeeze().detach().cpu().numpy()
        pred = np.round(pred, 0)
        acc = np.mean(pred == _y.detach().cpu().numpy())
        accs.append(acc)
    if verbose:
        print('loss', loss.item())
        print('nll', negative_log_likelihood.item())
        print('accuracy =', np.mean(accs))
    return negative_log_likelihood.item(), loss.item()

def test(net, dtest, p, verbose=False):
    data = dtest[:, 0:p - 1]
    target = dtest[:, -1].unsqueeze(1).float()
    outputs = net(data)
    negative_log_likelihood = net.loss(outputs, target)
    loss = negative_log_likelihood + net.kl() / NUM_BATCHES
    pred = outputs.squeeze().cpu().detach().numpy()
    pred = np.round(pred, 0)

    acc = np.mean(pred == target.detach().cpu().numpy().ravel())
    
    if verbose:
        print('\nTEST STATS: \n---------------------')
        print('loss', loss.item())
        print('nll', negative_log_likelihood.item())
        print('accuracy =', acc)
        
    return acc