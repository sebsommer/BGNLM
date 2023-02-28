import math
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
# from tensorboardx import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd

from nonlinear import *
from features import *

np.random.seed(1)

# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

# setting batch size and epochs
BATCH_SIZE = 2000
epochs = 200

USE_FLOWS = True #If false, mean-field aproximations are used

if USE_FLOWS:
    from flow_models import *
else:
    from mf_models import * 

N_GENERATIONS = 10
EXTRA_FEATURES = 10
COMPLEXITY_MAX = 10
dataset_nr = 5

#binomial
if dataset_nr == 1:
    x_df = pd.read_csv(
        'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/test.csv').iloc[
           :, 1:-1]

    y_df = pd.read_csv(
        'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/test.csv').iloc[
           :, -1]

    x_test = np.array(x_df)
    y_test = np.array(y_df)

    x_train = pd.read_csv(
        'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/train.csv').iloc[
              :, 1:-1].to_numpy()
    y_train = pd.read_csv(
        'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/train.csv').iloc[
              :, -1].to_numpy()

    family = 'binomial'

#binomial
elif dataset_nr == 2:
    x_df = pd.read_csv(
        'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-X.txt',
        header=None)

    y_df = pd.read_csv(
        'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-Y.txt',
        header=None)
    x = np.array(x_df)
    y = np.array(y_df)

    # Split:
    x_train = x_df.sample(frac=0.8, random_state=100)
    y_train = y_df.sample(frac=0.8, random_state=100)

    x_test = x_df[~x_df.index.isin(x_train.index)]
    y_test = y_df[~y_df.index.isin(y_train.index)]

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    family = 'binomial'

#gaussian
elif dataset_nr == 3:
    te_ids = pd.read_csv("https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/abalone%20age/teid.csv", header = 1, sep = ";").iloc[:,-1] -1
    df = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/abalone%20age/abalone.data', header=None)
    x_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]

    dummies = pd.get_dummies(x_df.iloc[:,0])
    res = pd.concat([x_df, dummies], axis = 1)
    x_df = res.drop([0], axis = 1)

    x_test = x_df.iloc[te_ids,:]
    y_test = y_df.iloc[te_ids]

    x_train = x_df[~x_df.index.isin(te_ids)]
    y_train = y_df[~y_df.index.isin(te_ids)]

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    family = 'gaussian'

# gaussian
elif dataset_nr == 4:

    df = pd.read_csv(
        'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/kepler%20and%20mass/exa1.csv')#[:,1:10]
    x_df = df.iloc[:, [1,2,3,5,6,7,8,9,10]]
    y_df = df.iloc[:, 4]

    x_test = x_df
    y_test =  y_df

    x_train = x_df
    y_train = y_df

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    family = 'gaussian'

#gaussian
elif dataset_nr == 5:
    x_gen = np.random.normal(0,1, (10000, 10)) + 5*np.ones((10000,10))
    x_gen[:,0] = x_gen[:,0]*np.sqrt(2)
    x_gen[:,1] = 1 / (np.exp(-x_gen[:,1]) + 1)
    y_gen = np.power(x_gen[:,0]*x_gen[:,0]*x_gen[:,1], 1/3)

    x_test = x_gen
    x_train = x_gen

    y_test = y_gen
    y_train = y_gen

    family = 'gaussian'

def normalize(data: np.array):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return np.divide((data - data_mean), data_std)

# normailizing data:
x_train = normalize(x_train)
x_test = normalize(x_test)

dtrain = torch.from_numpy(np.column_stack((x_train, y_train)))
dtest = torch.from_numpy(np.column_stack((x_test, y_test)))

TRAIN_SIZE = len(dtrain)
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE

class BayesianNetwork(nn.Module):
    def __init__(self, p, alpha_prior, family='binomial', useflows = USE_FLOWS):
        super().__init__()
        # set the architecture
        if not useflows:
            self.l1 = BayesianLinear(p - 1, 1, alpha_prior)
        else:
            self.l1 = BayesianLinearFlow(p - 1, 1, 3, alpha_prior)
        
        self.family = family
        if self.family == 'binomial':
            self.loss = nn.BCELoss(reduction='sum')
        elif self.family == 'gaussian':
            self.loss = nn.MSELoss(reduction='sum')
        elif self.family == 'poisson':
            self.loss = self.poissonLoss

    def poissonLoss(self, xbeta, y):
        """Custom loss function for Poisson model."""
        loss = torch.mean(torch.exp(xbeta) - y * xbeta)
        return loss

    def forward(self, x, sample=False):
        x = self.l1(x, sample)
        if self.family == 'binomial':
            x = torch.sigmoid(x)
        return x

    def kl(self):
        return self.l1.kl

def train(net, optimizer, dtrain, p, batch_size=BATCH_SIZE, verbose=False):
    family = net.family
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
        if family == 'binomial':
            pred = np.round(pred, 0)
            acc = np.mean(pred == _y.detach().cpu().numpy())
            accs.append(acc)
    if verbose:
        print('loss', loss.item())
        print('nll', negative_log_likelihood.item())
        if family == 'binomial':
            print('accuracy =', np.mean(accs))
    return loss.item(), negative_log_likelihood.item(),

def test(net, dtest, p, samples = 20, lower_q=0.05, upper_q=0.95, verbose=False):
    family = net.family

    data = dtest[:, 0:p - 1].to(DEVICE)
    target = dtest[:, -1].squeeze().float().to(DEVICE)
    outputs = net(data)
    for _ in range(samples - 1):
        outputs = torch.column_stack((outputs, net(data)))

    outputs_mean = outputs.mean(dim=1)
    outputs_upper = outputs.quantile(upper_q, interpolation='higher').detach().cpu().numpy()
    outputs_lower = outputs.quantile(lower_q, interpolation='lower').detach().cpu().numpy()

    pred = outputs_mean.cpu().detach().numpy()

    negative_log_likelihood = net.loss(outputs_mean, target)
    loss = negative_log_likelihood + net.kl() / NUM_BATCHES

    if family == 'binomial':
        pred = np.round(pred, 0)
        acc = np.mean(pred == target.detach().cpu().numpy().ravel())

    if verbose:
        print('\nTEST STATS: \n---------------------')
        print('loss', loss.item())
        print('nll', negative_log_likelihood.item())
        if family == 'binomial':
            print('accuracy =', acc)
    if family == 'binomial':
        return loss.item(), negative_log_likelihood.item(), acc, pred, outputs_upper, outputs_lower
    else:
        return loss.item(), negative_log_likelihood.item(), pred, outputs_upper, outputs_lower

def gamma_prior(comps):
    return torch.from_numpy(np.exp(-comps-2))

def var_inference(values, comps, iteration, family, prior_a = gamma_prior,verbose=False):
    tmpx = torch.tensor(np.column_stack((values.T, y_train)), dtype=torch.float32).to(DEVICE)

    data_mean = tmpx.mean(axis=0)[0:-1]
    data_std = tmpx.std(axis=0)[0:-1]
    tmpx[:, 0:-1] = (tmpx[:, 0:-1] - data_mean) / data_std
    
    n, p = tmpx.shape
    prior_a = gamma_prior(comps)

    net = BayesianNetwork(p, prior_a, family=family).to(DEVICE)
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(epochs):
        nll, loss = train(net, optimizer, tmpx, p, verbose=verbose)
        print('epoch =', epoch)
        print(f'generation: {iteration}')
    
    a = net.l1.alpha_q.data.detach().cpu().numpy().squeeze()

    return a, net

# SEB CODE:

def generate_feature(pop, val, target, comps, fprobs, mprobs, tprobs, family, test_x):
    rand = np.random.choice(4, size=1, p=fprobs)
    values = val.copy()
    # print("!:", values.shape)

    if rand == 0:
        idx = np.random.choice(pop.shape[0], size=2, p=mprobs, replace=True)
        mult = Multiplication(pop[idx], values[:, idx], np.sum(comps[idx]))
        feat, val = mult.feature, mult.values
        if test_x is not None:
            test_val = mult.evaluate(test_x[:, idx])
        else:
            test_val = None
        obj = mult
        # print("0:",val.shape)

    elif rand == 1:
        idx = np.random.choice(pop.shape[0], size=1, p=mprobs)
        g = np.random.choice(transformations, p=tprobs)
        mod = Modification(pop[idx], values[:, idx], g, comps[idx])
        feat, val = mod.feature, mod.values.ravel()
        if test_x is not None:
            test_val = mod.evaluate(test_x[:, idx]).ravel()
        else:
            test_val = None
        obj = mod
        # print("1:",val.shape)

    elif rand == 2:
        g = np.random.choice(transformations, p=tprobs)
        s = np.random.choice([2, 3, 4], size=1)
        idx = np.random.choice(pop.shape[0], size=s, p=mprobs, replace=False)
        proj = Projection(pop[idx], values[:, idx], target, g, family, 2*s + np.sum(comps[idx]))
        feat, val = proj.feature, proj.values
        if test_x is not None:
            test_val = proj.evaluate(test_x[:, idx])
        else:
            test_val = None
        obj = proj
        # print("2:",val.shape)

    elif rand == 3:
        idx = np.random.choice(x_train.shape[1])
        new = Linear(f'x{idx}')
        feat, val = str(new), new.evaluate(x_train[:, idx])
        if test_x is not None:
            test_val = test_x[:, idx]
        else:
            test_val = None
        obj = new
        # print("3:",val.shape)

    comp = float(obj.complexity)
    return feat, val, test_val, obj, comp

def check_collinearity(val, values):
    for i in range(values.shape[0]):
        if np.all(val == values[i,:], axis = 0):
            return True
    return False

# nonlinear features. Should be decided by user.
transformations = [Sigmoid, Sine, Cosine, Ln, Exp, x72, x52, x13]
tprobs = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])

# feature types
# mult, mod, proj, new:
fprobs = np.array([1/4, 1/4, 1/4, 1/4])

if dataset_nr == 4:
    transformations = [x13]
    tprobs = np.array([1])
    fprobs = np.array([1/3,1/3,0,1/3])

POPULATION_SIZE = x_train.shape[1] + EXTRA_FEATURES

# initializing first generation:
if POPULATION_SIZE > x_test.shape[1]:
    test_values = np.zeros((x_test.shape[0], POPULATION_SIZE))
    for i in range(x_test.shape[1]):
        test_values[:, i] = x_test[:, i]
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
    mprob = 1 / x_train.shape[1]
    population = np.array([F0[k]['feature'] for k in range(x_train.shape[1])])
    train_values = np.array([F0[k]['values'] for k in range(x_train.shape[1])])
    complexities = np.array([F0[k]['complexity'] for k in range(x_train.shape[1])])
    while (True):
        feat, train_vals, test_vals, obj, comp = generate_feature(population, train_values.T, y_train, complexities, fprobs,
                                                            [mprob for _ in range(x_train.shape[1])], tprobs, family, x_test)
        
        collinear = check_collinearity(train_vals, train_values)
        if comp < COMPLEXITY_MAX and np.all(np.isfinite(train_vals)) and not collinear:
            F0[j]['mprob'] = mprob
            F0[j]['complexity'] = comp
            F0[j]['feature'] = feat
            F0[j]['values'] = train_vals
            F0[j]['type'] = obj
            test_values[:, j] = test_vals
            break
        continue

# TRAINING:
if family == 'binomial':
    test_acc = []
test_loss = []


#Generation 0:
train_values = normalize(np.array([F0[k]['values'] for k in range(POPULATION_SIZE)]))
complexities = np.array([F0[k]['complexity'] for k in range(POPULATION_SIZE)])
mprobs, net = var_inference(train_values, complexities, 0, family, verbose=False)
for id, _ in F0.items():
        F0[id]['mprob'] = mprobs[id]

objects_list = [[F0[k]['type'] for k in range(POPULATION_SIZE)]]

predicted_vals, lower_vals, upper_vals, nets = [], [], [], [net]

test_values = normalize(test_values)
test_data = torch.tensor(np.column_stack((test_values, y_test)), dtype=torch.float32).to(DEVICE)

if family == 'binomial':
    loss, nll, accuracy, means, uppers, lowers = test(net, test_data, p=POPULATION_SIZE + 1)
    test_acc.append(accuracy)
    test_loss.append(loss)
else:
    loss, nll, means, uppers, lowers = test(net, test_data, p=POPULATION_SIZE + 1)
    test_loss.append(loss)

predicted_vals.append(means)
lower_vals.append(lowers)
upper_vals.append(uppers)

#Generation 1, .... , N_GENERATIONS:
for i in range(1, N_GENERATIONS + 1):
    # VI step:
    train_values = normalize(np.array([F0[k]['values'] for k in range(POPULATION_SIZE)]))
    complexities = np.array([F0[k]['complexity'] for k in range(POPULATION_SIZE)])
    mprobs, net = var_inference(train_values, complexities, i, family, verbose=False)
    population = np.array([F0[k]['feature'] for k in range(POPULATION_SIZE)])
    
    for id, _ in F0.items():
        F0[id]['mprob'] = mprobs[id]
        if F0[id]['mprob'] < 0.3:
            while (True):
                feat, train_vals, test_vals, obj, comp = generate_feature(population, train_values.T, y_train, complexities, fprobs,
                                                                    mprobs / np.sum(mprobs), tprobs, family, test_values)
                
                collinear = check_collinearity(train_vals, train_values)

                if comp < COMPLEXITY_MAX and np.all(np.isfinite(train_vals)) and not collinear:
                    F0[id]['complexity'] = comp
                    F0[id]['feature'] = feat
                    F0[id]['values'] = train_vals
                    F0[id]['type'] = obj
                    test_values[:, id] = test_vals
                    break
                continue
    
    objects_list.append([F0[k]['type'] for k in range(POPULATION_SIZE)])
    nets.append(net)

    # TESTING FOR CURRENT POPULATION:
    test_values = normalize(test_values)
    test_data = torch.tensor(np.column_stack((test_values, y_test)), dtype=torch.float32).to(DEVICE)

    if family == 'binomial':
        loss, nll, accuracy, means, uppers, lowers = test(net, test_data, p=POPULATION_SIZE + 1)
        test_acc.append(accuracy)
        test_loss.append(loss)
    else:
        loss, nll, means, uppers, lowers = test(net, test_data, p=POPULATION_SIZE + 1)
        test_loss.append(loss)

    predicted_vals.append(means)
    lower_vals.append(lowers)
    upper_vals.append(uppers)

# printing final population and test results:
for key, v in F0.items():
    f, m, c = v['feature'], v['mprob'], v['complexity']
    m = np.round(float(m), 3)
    t = type(v['type']).__name__
    print("ID: {:<3}, Feature: {:<60}, mprob:{:<15}, complexity: {:<5}, type: {:<10}".format(key, f, m, c, t))
if family == 'binomial':
    print('\n\nTEST RESULTS:\n----------------------')
    print('FINAL GENERATION ACCURACY:', accuracy)
    print('MEAN TEST ACCURACY: ', np.mean(test_acc))
    print('BEST ACCURACY:', np.max(test_acc), f'(generation {test_acc.index(max(test_acc))})')
elif family == 'gaussian':
    print('\n\nTEST RESULTS:\n----------------------')
    print('FINAL GENERATION LOSS:', np.sqrt(loss/x_test.shape[0]))
    print('MEAN TEST LOSS: ', np.sqrt(np.mean(test_loss)/x_test.shape[0]))
    print('BEST LOSS:', np.sqrt(np.min(test_loss)/x_test.shape[0]), f'(generation {test_loss.index(min(test_loss))})')

    #Best features
    for o in objects_list[test_loss.index(min(test_loss)) + 1]:
        print(o.feature)

sns.set()
fig1, ax1 = plt.subplots()
ax1.plot(predicted_vals[test_loss.index(min(test_loss))], color = "red", label = 'predicted', linestyle = " ", marker = '.')
ax1.axhline(upper_vals[test_loss.index(min(test_loss))], color = "blue", label = '0.95 quantile' ,linestyle='dashed')
ax1.axhline(lower_vals[test_loss.index(min(test_loss))], color= "blue", label = '0.05 quantile' ,linestyle='dashed')
ax1.plot(y_test, label = 'target', color = "black", linestyle = " ", marker = '.')
ax1.legend()
fig1.savefig('test_fig1.png')
plt.close(fig1)

if USE_FLOWS:
    fig2, ax2 = plt.subplots()
    sns.kdeplot(ax = ax2, x = predicted_vals[test_loss.index(min(test_loss))], label = 'predicted')
    #sns.kdeplot(up[test_loss.index(min(test_loss))], label = '0.95 quantile' ,linestyle='dashed', alpha =0.5)
    #sns.kdeplot(low[test_loss.index(min(test_loss))], label = '0.05 quantile' ,linestyle='dashed', alpha =0.5)
    sns.kdeplot(ax = ax2, x = y_test, label = 'target')
    ax2.legend()
    fig2.savefig('test_fig2.png')
    plt.close(fig2)
else:
    fig3, ax3 = plt.subplots()
    sns.kdeplot(ax = ax3, x = predicted_vals[test_loss.index(min(test_loss))], label = 'predicted', marker = 'o')
    #sns.kdeplot(up[test_loss.index(min(test_loss))], label = '0.95 quantile' ,linestyle='dashed', alpha =0.5)
    #sns.kdeplot(low[test_loss.index(min(test_loss))], label = '0.05 quantile' ,linestyle='dashed', alpha =0.5)
    sns.kdeplot(ax = ax3, x = y_test, label = 'target', marker = 'o')
    ax3.legend()
    fig3.savefig('test_fig3.png')
    plt.close(fig3)
