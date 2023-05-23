import time
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from torch.autograd import Variable
import pandas as pd

from nonlinear import *
from features import *
from dataloader import *

np.random.seed(100)

st = time.time()


# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


# setting batch size and epochs
BATCH_SIZE = 500
epochs = 600
custom_loss = None
LEARNIG_RATE = 0.01

USE_FLOWS = True

if USE_FLOWS:
    from flow_models import *
else:
    from mf_models import *


dataset_nr = sys.argv[1]

N_IAFLAYERS = 2
N_GENERATIONS = int(sys.argv[2])
EXTRA_FEATURES = int(sys.argv[3])
VERBOSE = True
COMPLEXITY_MAX = 10
norm = True
normalize = standardize

if dataset_nr=='3':
    BATCH_SIZE = 500
    epochs = 300
    LEARNIG_RATE = 0.01
    
    N_IAFLAYERS = 2

    N_GENERATIONS = int(sys.argv[2])
    EXTRA_FEATURES = int(sys.argv[3])
    COMPLEXITY_MAX = 10
    norm = True

crit = sys.argv[4]

if crit == 'AIC':
    def gamma_prior(comps,n,p):
        return torch.from_numpy(np.exp(-2*np.log1p(comps)-0.0001))
elif crit == 'BIC':
    def gamma_prior(comps,n,p):
        return torch.from_numpy(np.exp(-np.log(n)*np.log1p(comps)-0.0001))
elif crit == 'SIM':
    def gamma_prior(comps,n,p):
        return torch.from_numpy(np.exp(-np.log(n)*np.log1p(comps+1)-0.0001))
elif crit == 'SIM2':
    def gamma_prior(comps, n, p):
        return torch.from_numpy(np.exp(-(1 + np.log(2**(comps)))*np.log(n)))
else:
    print('No gamma-prior privided')
    sys.exit(2)


x_train, x_test, y_train, y_test, family, x_cols = dataloader(dataset_nr)

# normailizing data:
if norm:
    x_train = normalize(x_train)
    x_test = normalize(x_test)


TRAIN_SIZE = x_train.shape[0]
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE

class BayesianNetwork(nn.Module):
    def __init__(self, p, alpha_prior, family='binomial', useflows = USE_FLOWS, num_transforms = N_IAFLAYERS, special_loss = custom_loss):
        super().__init__()
        # set the architecture
        if not useflows:
            self.l1 = BayesianLinear(p - 1, 1, alpha_prior)
        else:
            self.l1 = BayesianLinearFlow(p - 1, 1, num_transforms, alpha_prior)
        
        self.family = family

        if special_loss is not None:
            self.loss = special_loss
        elif self.family == 'binomial':
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

def train(net, optimizer, dtrain, p, batch_size=BATCH_SIZE, verbose=VERBOSE):
    family = net.family
    net.train()

    old_batch = 0
    accs = []
    for batch in range(int(np.ceil(dtrain.shape[0] / batch_size))):
        batch = (batch + 1)
        _x = dtrain[old_batch: batch_size * batch, 0:p - 1]
        _y = dtrain[old_batch: batch_size * batch, -1]
        old_batch = batch_size * batch
        target = _y.to(DEVICE)
        data = _x.to(DEVICE)
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
    if not np.isfinite(loss.item()):
        print("RuntimeWarning: NaN discovered in loss!")
    if family == 'binomial':
        return loss.item(), negative_log_likelihood.item(), np.mean(accs)
    else:
        return loss.item(), negative_log_likelihood.item()

def test(net, dtest, samples = 30, lower_q=0.05, upper_q=0.95, verbose=VERBOSE):
    family = net.family

    net.eval()
    data = dtest[:,:-1]
    target = dtest[:,-1].squeeze().float()
    sdy = dtest[:, -1].squeeze().float().detach().cpu().numpy().std()
    outputs = net(data).detach().cpu().numpy()
    for _ in range(samples - 1):
        outputs = np.column_stack((outputs, net(data).detach().cpu().numpy()))
    outputs_prd = np.random.normal(outputs,np.zeros_like(outputs)+sdy) 

    outputs_mean = torch.tensor(outputs.mean(axis=1), device=DEVICE)
    outputs_upper = np.quantile(outputs_prd,upper_q, axis=1)
    outputs_lower = np.quantile(outputs_prd,lower_q, axis=1)

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
    else:
        if family == 'binomial':
            print(f'Validation Accuracy {acc}\n')
        else:
            print(f'Validation loss (RMSE): {np.sqrt(loss.item()/outputs.shape[0])}\n')
    if family == 'binomial':
        return loss.item(), negative_log_likelihood.item(), acc, pred, outputs_upper, outputs_lower
    else:
        return loss.item(), negative_log_likelihood.item(), pred, outputs_upper, outputs_lower

def var_inference(values, comps, iteration, family, net = None, prior_func_a = gamma_prior, verbose=VERBOSE, learnig_rate = LEARNIG_RATE):
    tmpx = torch.tensor(np.column_stack((values, y_train)), dtype=torch.float32).to(DEVICE)
    
    n, p = tmpx.shape
    prior_a = prior_func_a(comps, n, p - 1)

    #print("Prior vals", prior_a)

    if net is not None:
        bayes_net = net
    else:    
        bayes_net = BayesianNetwork(p, prior_a, family=family).to(DEVICE)
    optimizer = optim.Adam(bayes_net.parameters(), lr=learnig_rate)

    for epoch in range(epochs):
        if family == 'binomial':
            nll, loss, acc = train(bayes_net, optimizer, tmpx, p, verbose=verbose)
        else:
            nll, loss = train(bayes_net, optimizer, tmpx, p, verbose=verbose)
        if verbose:
            print('epoch =', epoch)
            print(f'generation: {iteration}')
    
    a = bayes_net.l1.alpha_q.data.detach().cpu().numpy().squeeze()

    return a, bayes_net

# SEB CODE:

def generate_feature(pop, val, target, comps, fprobs, mprobs, tprobs, family, train_x, test_x, x_cols):
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
        idx = np.random.choice(train_x.shape[1])
        new = Linear(x_cols[idx], train_x[:,idx])
        feat, val = str(new), new.values
        if test_x is not None:
            test_val = test_x[:, idx]
        else:
            test_val = None
        obj = new
        # print("3:",val.shape)

    comp = float(obj.complexity)
    return feat, val, test_val, obj, comp

def check_collinearity(val, values):
    v = values.copy()
    v_test = val.copy()
    idx = np.random.choice(v.shape[0], size = v.shape[0]//3, replace = False)
    for i in range(v.shape[1]):
        corr = np.corrcoef(v[idx,i], v_test[idx])[0][1]
        #corr = np.corrcoef(v[i,:], v_test[:])[0][1]
        if np.abs(corr) > 0.9:
            #print("NOPE", corr)
            return True
    #print("YES")
    return False

# nonlinear features. Should be decided by user.
transformations = [Gauss, Sigmoid, Sine, Cosine, Tanh, Atan, Ln, Exp, x72, x52, x13]
tprobs = np.array([1/len(transformations) for _ in transformations])

# feature types
# mult, mod, proj, new:
fprobs = np.array([1/4, 1/4, 1/4, 1/4])
    
POPULATION_SIZE = x_train.shape[1] + EXTRA_FEATURES

F0 = {}
for j in range(x_train.shape[1]):
    F0[j] = {}
    F0[j]['feature'] = x_cols[j]
    F0[j]['mprob'] = 0
    F0[j]['complexity'] = 0
    F0[j]['values'] = x_train[:, j]
    F0[j]['type'] = Linear(x_cols[j], x_train[:,j])

# TRAINING:
if family == 'binomial':
    test_acc = []
test_loss = []

#Generation 0:
print("\nTraining/testing generation 0:")
train_values = x_train
complexities = np.array([F0[k]['complexity'] for k in range(x_train.shape[1])])
mprobs, net = var_inference(train_values, complexities, 0, family, None, gamma_prior)
for id, _ in F0.items():
        F0[id]['mprob'] = mprobs[id]

objects_list = [[F0[k]['type'] for k in range(x_train.shape[1])]]
mprobs_list = [mprobs]

predicted_vals, lower_vals, upper_vals, nets = [], [], [], [net]

test_values = x_test
test_data = torch.tensor(np.column_stack((test_values, y_test)), dtype=torch.float32).to(DEVICE)

if family == 'binomial':
    loss, nll, accuracy, means, uppers, lowers = test(net, test_data)
    test_acc.append(accuracy)
    test_loss.append(loss)
else:
    loss, nll, means, uppers, lowers = test(net, test_data)
    test_loss.append(loss)

predicted_vals.append(means)
lower_vals.append(lowers)
upper_vals.append(uppers)

if POPULATION_SIZE > x_test.shape[1]:
    test_values = np.zeros((x_test.shape[0], POPULATION_SIZE))
    for i in range(x_test.shape[1]):
        test_values[:, i] = x_test[:, i]

if N_GENERATIONS > 0:
    col = train_values
    count=1
    for j in range(x_train.shape[1], POPULATION_SIZE):
        F0[j] = {}
        mprobs = np.array([F0[k]['mprob'] for k in range(x_train.shape[1])])
        population = np.array([F0[k]['feature'] for k in range(x_train.shape[1])])
        train_values = np.array([F0[k]['values'] for k in range(x_train.shape[1])]).T
        complexities = np.array([F0[k]['complexity'] for k in range(x_train.shape[1])])
        while (True):
            feat, train_vals, test_vals, obj, comp = generate_feature(population, train_values, y_train, complexities, fprobs,
                                                                mprobs / np.sum(mprobs), tprobs, family, x_train, x_test, x_cols)
            
            collinear = check_collinearity(train_vals, col)
            if comp <= COMPLEXITY_MAX and np.all(np.isfinite(train_vals)) and not collinear:
                F0[j]['mprob'] = None
                F0[j]['complexity'] = comp
                F0[j]['feature'] = feat
                F0[j]['values'] = train_vals
                F0[j]['type'] = obj
                col = np.column_stack((col, train_vals))
                test_values[:, j] = test_vals
                count += 1
                break
            continue

#Generation 1, .... , N_GENERATIONS:
print("\nStarting training for", N_GENERATIONS, "generations:\n" )
for i in range(1, N_GENERATIONS + 1):
    # VI step:
    print(f"Training/testing for generation {i}:")
    train_values = np.array([F0[k]['values'] for k in range(POPULATION_SIZE)]).T
    if norm:
        train_values = normalize(train_values)
    population = np.array([F0[k]['feature'] for k in range(POPULATION_SIZE)])
    complexities = np.array([F0[k]['complexity'] for k in range(POPULATION_SIZE)])
    
    #Training current population:
    mprobs, net = var_inference(train_values, complexities, i, family, None, gamma_prior)
    mprobs_list.append(mprobs)
    
    #Storing mprobs:
    for id, _ in F0.items():
        F0[id]['mprob'] = mprobs[id]
    if i < N_GENERATIONS:
        #Replacing all low probability things for next generation: 
        for id, _ in F0.items():
            #print("ID:", id, "mprob",F0[id]['mprob'])
            if F0[id]['mprob'] < 0.3:
                #print("Inside", id)
                if F0[id]['mprob'] > np.random.uniform(): #Keep bad features with some probability
                    stop = True
                else:
                    stop = False
                    #print("Not stopping")
                while not stop:
                    #print(id, "mprob", F0[id]['mprob'], "OLD: ", F0[id]['feature'])
                    
                    #Finding new feature:
                    feat, train_vals, test_vals, obj, comp = generate_feature(population, train_values, y_train, complexities, fprobs,
                                                                        mprobs / np.sum(mprobs), tprobs, family, x_train, test_values, x_cols)
                    
                    collinear = check_collinearity(train_vals, train_values)
                    #print(id, "mprob", F0[id]['mprob'], "NEW: ", feat, "\n----------\n")

                    #Replace if it is not already in population:
                    if comp <= COMPLEXITY_MAX and not collinear and np.all(np.isfinite(train_vals)):
                        F0[id]['mprob'] = None
                        F0[id]['complexity'] = comp
                        F0[id]['feature'] = feat
                        F0[id]['values'] = train_vals
                        F0[id]['type'] = obj
                        train_values[:, id] = train_vals
                        test_values[:, id] = test_vals
                        stop = True
                        #print(id, "NEW: ", feat, "\n----------\n")
                    

    objects_list.append([F0[k]['type'] for k in range(POPULATION_SIZE)])
    nets.append(net)

    # TESTING FOR CURRENT POPULATION:
    
    if norm:
        test_values = normalize(test_values)
    test_data = torch.tensor(np.column_stack((test_values, y_test)), dtype=torch.float32).to(DEVICE)

    if family == 'binomial':
        loss, nll, accuracy, means, uppers, lowers = test(net, test_data)
        test_acc.append(accuracy)
        test_loss.append(loss)
    else:
        loss, nll, means, uppers, lowers = test(net, test_data)
        test_loss.append(loss)

    predicted_vals.append(means)
    lower_vals.append(lowers)
    upper_vals.append(uppers)

# printing final population and test results:

def test_generation(neural_net, test_vals, test_history, pred_history = None, lower_history = None, upper_history = None):
    """
    
    """
    if neural_net.family == 'binomial':
        loss, nll, accuracy, means, uppers, lowers = test(neural_net, test_vals)
        loss_list = test_history + [accuracy]
    else:
        loss, nll, means, uppers, lowers = test(neural_net, test_data)
        loss_list = test_history + [loss]

    retlist = []
    if pred_history is not None:
        pred = pred_history + [means]
        retlist.append(pred)
    if lower_history is not None:
        lower = lower_history + [lowers]
        retlist.append(lower)
    if upper_history is not None:
        upper = upper_history + [uppers]
        retlist.append(upper)

    return loss_list, retlist

if VERBOSE:
    print("Loss evolution:", test_loss)

def print_best(family, test_history, object_history, mprob_history, print_best = True, mprob_treshold=0.3):
    #assert family == 'binomial' or family == 'gaussian', "Family must be 'binomial' or 'gaussian"

    if family == 'binomial':
        best = test_history.index(max(test_history))
        print('\n\nTEST RESULTS:\n----------------------')
        print('FINAL GENERATION ACCURACY:', test_history[-1])
        print('MEAN TEST ACCURACY: ', np.mean(test_history))
        print('BEST ACCURACY:', np.max(test_history), f'(generation {best})')
    elif family == 'gaussian':
        best = test_history.index(min(test_history))
        print('\n\nTEST RESULTS:\n----------------------')
        print('FINAL GENERATION LOSS:', test_history[-1])
        print('MEAN TEST LOSS: ', np.mean(test_history))
        print('BEST LOSS (RMSE):', np.min(test_history), f'(generation {best})')

    #Best features
    if print_best:
        print(f'Best generation ({best}):')
        for i, o in enumerate(object_history[best]):
                m = mprob_history[best][i]
                if m > mprob_treshold:
                    print("Feature: {:<60}, mprob:{:<15}, complexity: {:<5}, type: {:<10}".format(o.feature, m, o.complexity, o.__name__))

printout= True
if printout:
    print_best(family, test_loss, objects_list, mprobs_list)

et = time.time()

res = et-st
final_res = res/60

print("Total execution time:", final_res, "minutes")