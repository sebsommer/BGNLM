import time
import sys
import getopt

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
from dataloader import *
from utils import * 

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

arg_help = "{0} -d <dataset nr> -g <number of generations> -e <extra features> -c <max complexity> -v <verbose> -f <use_flows> -p <prior criterion> -nx <normalize x> -ny <normalize y> -out <print results>".format(sys.argv[0])
try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:g:v:e:c:b:z:y:s:f:p:nx:ny:out:", ["help", "dataset nr=", "generations=", "verbose=", "extra features=", "max complexity=","batch_size=","epochs=", "y=", "sigma=", "use_flows=","prior=", "normalize_x=", "normalize_y=","printout="])
except:
    print(arg_help)
    sys.exit(2)

for opt, arg in opts:
    if opt in ("-h", "--help"):
        print(arg_help)
        sys.exit(2)
    elif opt in ("-d", "--dataset_nr"):
        dataset_nr = arg
    elif opt in ("-g", "--n_generations"):
        N_GENERATIONS = int(arg)
    elif opt in ("-e", "--extra_features"):
        EXTRA_FEATURES = int(arg)
    elif opt in ("-c", "--max_complexity"):
        COMPLEXITY_MAX = int(arg)
    elif opt in ("-b", "--batch_size"):
        BATCH_SIZE = int(arg)
    elif opt in ("-z", "--epochs"):
        epochs = int(arg)
    elif opt in ("-v", "--verbose"):
        if arg == "True" or arg == "T":
            VERBOSE = True
        elif arg == "False" or arg == "F":
            VERBOSE = False
    elif opt in ("-y", "--sim2y"):
        sim2y = float(arg)
    elif opt in ("-s", "--sigma"):
        sigma = arg
    elif opt in ("-p", "--prior"):
        crit = arg
    elif opt in ("-nx", "--normalize_x"):
        norm_x = bool(arg)
    elif opt in ("-ny", "--normalize_y"):
        norm_y = bool(arg)    
    elif opt in ("-f", "--use_flows"):
        if arg == "True" or arg == "T":
            USE_FLOWS = True
        elif arg == "False" or arg == "F":
            USE_FLOWS = False
    elif opt in ("-o", "--printout"):
        printout = bool(arg)

try: dataset_nr
except: 
    print("Must specify dataset_nr. \n\t1: Breast cancer, binomial. \n\t2: Simulated, binomial. \n\t3: Abalone, normal. \n\t4: Kepler's third law, normal. \n\t5: Slice localization, normal \n\tsim1: simulated, normal ")
    print("See help:\n", arg_help)
    sys.exit(2)
try: N_GENERATIONS
except NameError: N_GENERATIONS = 0
try: EXTRA_FEATURES
except NameError: EXTRA_FEATURES = 0
try: VERBOSE
except NameError: VERBOSE = False
try: sigma
except NameError: sigma = None
try: BATCH_SIZE
except NameError: BATCH_SIZE = 500
try: epochs
except NameError: epochs = 600
try: COMPLEXITY_MAX
except NameError: COMPLEXITY_MAX = 10
try: crit
except NameError: crit = 'AIC'
try: norm_x
except NameError: norm_x = True
try: norm_y
except NameError: norm_y = False
try: USE_FLOWS
except NameError: USE_FLOWS = True
try: printout
except NameError: printout = True

if USE_FLOWS:
    from flow_models import *
    print("Flow approximations are used!")
else:
    from mf_models import *
    print("Mean-field approximations are used!")
if crit == 'AIC':
    def gamma_prior(comps,n,p):
        return torch.from_numpy(np.exp(-2*np.log1p(comps)-0.0001))
elif crit == 'BIC':
    def gamma_prior(comps,n,p):
        return torch.from_numpy(np.exp(-np.log(n)*np.log1p(comps)-0.0001))
elif crit == 'SIM':
    def gamma_prior(comps,n,p):
        return torch.from_numpy(np.exp(-2*np.log1p(comps+1)-0.0001))
elif crit == 'SIM2':
    def gamma_prior(comps, n, p):
        return torch.from_numpy(np.exp(-(1 + np.log(2**(comps)))*np.log(n)))
    
#Starting timer:    
st = time.time()

#Choosing device:
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

#Setting batch size, epochs and learning rate:
N_IAFLAYERS = 2
custom_loss = None
LEARNIG_RATE = 0.01
normalize = standardize

#Loading data:
x_train, x_test, y_train, y_test, family, x_cols = dataloader(dataset_nr)

TRAIN_SIZE = x_train.shape[0]
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE

# --------------------------------- MODEL ---------------------------------

class BayesianNetwork(nn.Module):
    def __init__(self, p, alpha_prior, family='gaussian', useflows = USE_FLOWS, num_transforms = N_IAFLAYERS, special_loss = custom_loss):
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

def train_batchnorm(net, optimizer, dtrain, p, batch_size=BATCH_SIZE, verbose=VERBOSE):
    family = net.family
    net.train()
    
    def norm(data):
        data_mean = data.mean(axis=0)
        data_std = data.std(axis=0)
        return torch.divide((data - data_mean), data_std)

    accs = []
    i = 0
    for i in range(dtrain.shape[0] // batch_size + 1):
        batch = dtrain[i*batch_size:(i+1)*batch_size,:]
        _x = batch[:, :-1]
        _y = batch[:, -1]
        _x = norm(_x)
        target = Variable(_y).to(DEVICE)
        data = Variable(_x).to(DEVICE)
        net.zero_grad()
        outputs = net(data, sample=True)
        target = target.unsqueeze(1).float()

        negative_log_likelihood = net.loss(outputs, target)
        loss = negative_log_likelihood + net.kl() / (dtrain.shape[0] // batch_size)
        loss.backward()
        optimizer.step()
    if dtrain.shape[0] % batch_size != 0:
        batch = dtrain[i*batch_size:dtrain.shape[0]]
        _x = batch[:, :- 1]
        _y = batch[:, -1]
        target = Variable(_y).to(DEVICE)
        data = Variable(_x).to(DEVICE)
    
        net.zero_grad()
        outputs = net(data, sample=True)
        target = target.unsqueeze(1).float()

        negative_log_likelihood = net.loss(outputs, target)
        loss = negative_log_likelihood + net.kl() / (dtrain.shape[0] // batch_size)
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

def test(net, dtest, p, samples = 30, lower_q=0.05, upper_q=0.95, verbose=VERBOSE):
    def norm(x):
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        return torch.divide((x - x_mean), x_std)
    
    family = net.family

    net.eval()
    data = dtest[:, 0:p - 1]

    data = norm(data)

    target = dtest[:, -1].squeeze().float()
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
    loss = negative_log_likelihood #+ net.kl() / NUM_BATCHES

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
        if family =='binomial':
            printProgressBar(epochs, epochs, suffix = f"Complete. \tValidation accuracy: {acc}")
        else:
            printProgressBar(epochs, epochs, suffix = f"Complete. \tValidation loss (RMSE): {np.sqrt(loss.item()/outputs.shape[0])}")
    if family == 'binomial':
        return loss.item(), negative_log_likelihood.item(), acc, pred, outputs_upper, outputs_lower
    else:
        return loss.item(), negative_log_likelihood.item(), pred, outputs_upper, outputs_lower

def var_inference(values, comps, iteration, family, net = None, prior_func_a = gamma_prior, verbose=VERBOSE, learnig_rate = LEARNIG_RATE):
    
    def norm(x):
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        return np.divide((x - x_mean), x_std)
    
    normed_vals = norm(values)
    
    tmpx = torch.tensor(np.column_stack((normed_vals, y_train)), dtype=torch.float32)#.to(DEVICE)
    
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
        else:
            printProgressBar(epoch, epochs, suffix = f"Complete")
    
    a = bayes_net.l1.alpha_q.data.detach().cpu().numpy().squeeze()

    return a, bayes_net

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
    idx = np.random.choice(v.shape[0], size = v.shape[0]//4, replace = False)
    for i in range(v.shape[1]):
        corr = np.corrcoef(v[idx,i], v_test[idx])[0][1]
        #corr = np.corrcoef(v[i,:], v_test[:])[0][1]
        if np.abs(corr) > 0.95:
            #print("NOPE", corr)
            return True
    #print("YES")
    return False

# nonlinear features. Should be decided by user.
transformations = [Gauss, Sigmoid, Sine, Cosine, Tanh, Atan, Ln, Exp, x72, x52, x13]
tprobs = np.array([1/len(transformations) for e in transformations])

# feature types
# mult, mod, proj, new:
fprobs = np.array([1/4, 1/4, 1/4, 1/4])

# --------------------------------- INITIALIZING ---------------------------------

# normailizing data:
if norm_x:
    x_train = normalize(x_train)
    x_test = normalize(x_test)

if norm_y:
    y_train = normalize(y_train)
    y_test = normalize(y_test)


if dataset_nr == '1':
    #transformations = [Sigmoid, Sine, Cosine, Ln, Exp]
    #transformations = [Gauss, Sine, Tanh, Atan]
    #tprobs = [1/4,1/4,1/4,1/4]
    def gamma_prior(comps, n, p):
        return torch.from_numpy(np.exp(-2*np.log1p(comps)-0.000001))
if dataset_nr == '3':
    transformations = [Sigmoid, Ln1p, Expm1, x72, x52, x13]
    tprobs = np.array([1/len(transformations) for e in transformations])
    fprobs = [1/4, 1/4, 1/4, 1/4]
if dataset_nr == '4':
    transformations = [x13, p2]
    tprobs = np.array([1/len(transformations) for e in transformations])
    fprobs = np.array([1/3,1/3,0,1/3])
    def generate_feature(pop, val, target, comps, fprobs, mprobs, tprobs, family, train_x, test_x, x_cols):
        rand = np.random.choice(4, size=1, p=fprobs)
        values = val.copy()
        # print("!:", values.shape)

        if rand == 0:
            idx = np.random.choice(pop.shape[0], size=2, replace=True)
            mult = Multiplication(pop[idx], values[:, idx], np.sum(comps[idx]))
            feat, val = mult.feature, mult.values
            if test_x is not None:
                test_val = mult.evaluate(test_x[:, idx])
            else:
                test_val = None
            obj = mult
            # print("0:",val.shape)

        elif rand == 1:
            idx = np.random.choice(pop.shape[0], size=1)
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
if dataset_nr == '5':
    transformations = [Gauss, Sigmoid, Sine, Cosine, Tanh, Atan, Ln1p, Expm1, Imean, percentile25, percentile75]
    tprobs = np.array([1/len(transformations) for e in transformations])
    fprobs = np.array([1/3,1/3,0,1/3])
if dataset_nr == '6':
    transformations = [Gauss, Sigmoid, Sine, Cosine, Tanh, Atan, Ln, Exp, x72, x52, x13, Imean]
    tprobs = np.array([1/len(transformations) for e in transformations])
if dataset_nr=='test':
    transformations = [Sigmoid, Sine, Cosine, Ln, Exp, x72, x52, x13]
    tprobs = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])
    fprobs = np.array([1/3, 1/3, 0, 1/3])
elif dataset_nr == 'sim2' or dataset_nr == 'sim2test' or dataset_nr == 'sim2_nonorm' or dataset_nr == 'sim2_precomputed':
    transformations = [p1, pm1, pm2, pm05, p05, p2, p3, p0p0, p0pm1, p0pm2, p0pm05,p0p05, p0p1, p0p2, p0p3]
    tprobs = [1./len(transformations) for _ in range(len(transformations))]
    fprobs = [1/2, 1/2]
    def generate_feature(pop, val, target, comps, fprobs, mprobs, tprobs, family, train_x, test_x, x_cols):
        rand = np.random.choice(2, size=1, p=fprobs)
        values = val.copy()
        
        if rand == 0: 
            idx = np.random.choice(np.argwhere(comps<0.5).ravel(), size=1)
            g = np.random.choice(transformations, p=tprobs)
            mod = Modification(pop[idx], values[:, idx], g, comps[idx])
            feat, val = mod.feature, mod.values.ravel()
            if test_x is not None:
                test_val = mod.evaluate(test_x[:, idx]).ravel()
            else:
                test_val = None
            obj = mod

            # print("1:",val.shape)

        elif rand == 1:
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
elif dataset_nr == 'sim2_original':
    transformations = [pm1, pm2, pm05, p05, p2, p3, xm15, Ln1p, p0p0, p0pm1, p0pm2, p0pm05, p0p1, p0p2, p0p3]
    tprobs = [1./len(transformations) for _ in range(len(transformations))]
    fprobs = [1/2, 1/2]
    def generate_feature(pop, val, target, comps, fprobs, mprobs, tprobs, family, train_x, test_x, x_cols):
        rand = np.random.choice(2, size=1, p=fprobs)
        values = val.copy()
        
        if rand == 0: 
            idx = np.random.choice(np.argwhere(comps<0.5).ravel(), size=1)
            g = np.random.choice(transformations, p=tprobs)
            mod = Modification(pop[idx], values[:, idx], g, comps[idx])
            feat, val = mod.feature, mod.values.ravel()
            if test_x is not None:
                test_val = mod.evaluate(test_x[:, idx]).ravel()
            else:
                test_val = None
            obj = mod

            # print("1:",val.shape)

        elif rand == 1:
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


#Filling first population. F0 is a dict of dicts

POPULATION_SIZE = x_train.shape[1] + EXTRA_FEATURES

F0 = {}
for j in range(x_train.shape[1]):
    F0[j] = {}
    F0[j]['feature'] = x_cols[j]
    F0[j]['mprob'] = 0
    F0[j]['complexity'] = 0
    F0[j]['values'] = x_train[:, j]
    F0[j]['type'] = Linear(x_cols[j], x_train[:,j])

# --------------------------------- TRAINING/TESTING ---------------------------------

predicted_vals, lower_vals, upper_vals, nets = [], [], [], []
if family == 'binomial':
    test_acc = []
test_loss = []

#Training generation 0:
print("\nTraining/testing generation 0:")
train_values = x_train
complexities = np.array([F0[k]['complexity'] for k in range(x_train.shape[1])])
mprobs, net = var_inference(train_values, complexities, 0, family, None, gamma_prior)
for id, _ in F0.items():
        F0[id]['mprob'] = mprobs[id]

objects_list = [[F0[k]['type'] for k in range(x_train.shape[1])]]
mprobs_list = [mprobs]

#Testing generation 0:
test_values = x_test
test_data = torch.tensor(np.column_stack((test_values, y_test)), dtype=torch.float32).to(DEVICE)

if family == 'binomial':
    loss, nll, accuracy, means, uppers, lowers = test(net, test_data, p=x_train.shape[1] + 1)
    test_acc.append(accuracy)
    test_loss.append(loss)
else:
    loss, nll, means, uppers, lowers = test(net, test_data, p=x_train.shape[1] + 1)
    test_loss.append(loss)

predicted_vals.append(means)
lower_vals.append(lowers)
upper_vals.append(uppers)
nets.append(net)


#Filling generation 1 with extra features:
if POPULATION_SIZE > x_test.shape[1]:
    test_values = np.zeros((x_test.shape[0], POPULATION_SIZE))
    for i in range(x_test.shape[1]):
        test_values[:, i] = x_test[:, i]
if N_GENERATIONS > 0:
    print("\nFilling population 1 with", EXTRA_FEATURES, "extra features:")
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
                printProgressBar(count, EXTRA_FEATURES, suffix = 'Complete')
                count += 1
                break
            continue

#Training generation 1, .... , N_GENERATIONS:
print("\nStarting training for", N_GENERATIONS, "generations:\n" )
for i in range(1, N_GENERATIONS + 1):
    # VI step:
    print(f"Training/testing for generation {i}:")
    train_values = np.array([F0[k]['values'] for k in range(POPULATION_SIZE)]).T
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
            if F0[id]['mprob'] < 0.3:
                if F0[id]['mprob'] > np.random.uniform(): #Keep bad features with some probability
                    stop = True
                else:
                    stop = False
                #Replacing features:
                while not stop:
                    feat, train_vals, test_vals, obj, comp = generate_feature(population, train_values, y_train, complexities, fprobs,
                                                                        mprobs / np.sum(mprobs), tprobs, family, x_train, test_values, x_cols)
                    
                    collinear = check_collinearity(train_vals, train_values)

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
    
    #Storing feature objects and network
    objects_list.append([F0[k]['type'] for k in range(POPULATION_SIZE)])
    nets.append(net)

    # Testing for generation i:
    test_data = torch.tensor(np.column_stack((test_values, y_test)), dtype=torch.float32).to(DEVICE)

    if family == 'binomial':
        loss, nll, accuracy, means, uppers, lowers = test(net, test_data, p=POPULATION_SIZE + 1)
        test_acc.append(accuracy)
        test_loss.append(loss)
    else:
        loss, nll, means, uppers, lowers = test(net, test_data, p=POPULATION_SIZE + 1)
        test_loss.append(loss)

    #Storing results
    predicted_vals.append(means)
    lower_vals.append(lowers)
    upper_vals.append(uppers)


# --------------------------------- RESULTS ---------------------------------

if VERBOSE:
    print("Loss evolution:", test_loss)


if printout:
    # for key, v in F0.items():
    #     f, m, c = v['feature'], v['mprob'], v['complexity']
    #     m = np.round(float(m), 3)
    #     t = type(v['type']).__name__
    #     print("ID: {:<3}, Feature: {:<60}, mprob:{:<15}, complexity: {:<5}, type: {:<10}".format(key, f, m, c, t))

    print('BEST GENERATION (kappa > 0.3): \n')
    for i, o in enumerate(objects_list[test_loss.index(min(test_loss))]):
            mprob = mprobs_list[test_loss.index(min(test_loss))][i]
            if mprob > 0.3:
                print(f'{o.feature}, mprob: {mprobs_list[test_loss.index(min(test_loss))][i]}')

    if family == 'binomial':
        print('\n\nTEST RESULTS:\n----------------------')
        print('FINAL GENERATION ACCURACY:', accuracy)
        print('MEAN TEST ACCURACY: ', np.mean(test_acc))
        print('BEST ACCURACY:', np.max(test_acc), f'(generation {test_acc.index(max(test_acc))})')
    elif family == 'gaussian':
        print('\n\nTEST RESULTS:\n----------------------')
        print('FINAL GENERATION LOSS:', np.sqrt(loss/x_test.shape[0]))
        print('MEAN TEST LOSS: ', np.sqrt(np.mean(test_loss)/x_test.shape[0]))
        print('BEST LOSS (RMSE):', np.sqrt(np.min(test_loss)/x_test.shape[0]), f'(generation {test_loss.index(min(test_loss))})')


# --------------------------------- STORING RESULTS FROM BEST POPULATION IN DICT 'best_gen' ---------------------------------

if family == 'binomial':
    best_id = test_acc.index(np.max(test_acc))
else:
    best_id = test_loss.index(min(test_loss))


beta_mu = nets[best_id].l1.weight_mu.data.detach().cpu().numpy().squeeze()
beta_sigma = nets[best_id].l1.weight_sigma.data.detach().cpu().numpy().squeeze()
alpha = nets[best_id].l1.alpha_q.data.detach().cpu().numpy().squeeze()
if USE_FLOWS:
    z, _ = nets[best_id].l1.sample_z(x_test.shape[0])
    z = z.data.detach().cpu().numpy().squeeze()
else:
    z = np.ones(x_test.shape[0])

posterior_gamma = np.where(alpha > 0.5, 1, 0)
posterior_mu_beta = beta_mu * alpha * z
posterior_var_beta = alpha * (beta_sigma ** 2 + (1 - alpha) * (beta_mu * z) ** 2)

best_gen = {}
best_gen['gen_nr'] = best_id
best_gen['RMSE'] = np.sqrt(test_loss[best_id]/x_test.shape[0])
best_gen['features'] = np.array([o.feature for o in objects_list[best_id]])
best_gen['mprobs'] = mprobs_list[best_id]
best_gen['pred'] = predicted_vals[best_id]
best_gen['ci'] = (lower_vals[best_id], upper_vals[best_id])
best_gen['net'] = nets[best_id]
best_gen['mu_beta'] = posterior_mu_beta
best_gen['sigma_beta'] = np.sqrt(posterior_var_beta)
best_gen['gamma'] = posterior_gamma
if family == 'binomial':
    best_gen['result'] = np.max(test_acc)
    best_gen['FNR'] = np.sum(np.logical_and(predicted_vals[best_id] == 0, y_test == 1))/np.sum(y_test == 1)
    best_gen['FPR'] = np.sum(np.logical_and(predicted_vals[best_id] == 1, y_test == 0))/np.sum(y_test == 0)
else:
    best_gen['result'] = np.sqrt(np.min(test_loss)/x_test.shape[0]) 
    best_gen['MAE'] = np.mean(np.abs(y_test - predicted_vals[best_id]))
    best_gen['corr'] = np.corrcoef(predicted_vals[best_id], y_test)[0][1]


#Print execution time:
et = time.time()
res = et-st
final_res = res/60
print("Total execution time:", final_res, "minutes")