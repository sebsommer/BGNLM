import math
import time
import sys
import getopt
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
from utils import printProgressBar

np.random.seed(100)

st = time.time()

arg_help = "{0} -d <dataset nr> -g <number of generations> -e <extra features> -c <max complexity> -v <verbose>".format(sys.argv[0])
try:
    opts, args = getopt.getopt(sys.argv[1:], "hd:g:v:e:c:s", ["help", "dataset nr=", "generations=", "extra features=", "max complexity", "verbose=", "sigma="])
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
    elif opt in ("-v", "--verbose"):
        if arg == "True" or arg == "T":
            VERBOSE = True
        elif arg == "False" or arg == "F":
            VERBOSE = False
        elif arg == " ": 
            VERBOSE = True
    elif opt in ("-s", "--sigma"):
        sigma = arg

try: dataset_nr
except: 
    print("Must specify dataset_nr. \n\t1: Breast cancer, binomial. \n\t2: Simulated, binomial. \n\t3: Abalone, normal. \n\t4: Kepler's third law, normal. \n\t5: Slice localization, normal \n\tsim1: simulated, normal ")
    print("See help:\n", arg_help)
    sys.exit(2)

# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")


# setting batch size and epochs
BATCH_SIZE = 500
epochs = 300
custom_loss = None
LEARNIG_RATE = 0.01

USE_FLOWS = True

if USE_FLOWS:
    from flow_models import *
    print("Flows are used!")
else:
    from mf_models import *
    print("Mean-field approximations are used!")


N_IAFLAYERS = 2

try: N_GENERATIONS
except NameError: N_GENERATIONS = 0

try: EXTRA_FEATURES
except NameError: EXTRA_FEATURES = 0

try: VERBOSE
except NameError: VERBOSE = False

try: sigma
except NameError: sigma = None

try: COMPLEXITY_MAX
except NameError: COMPLEXITY_MAX = 10

#dataset_nr = sys.argv[1]

norm = True

#binomial
if dataset_nr == '1':
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
    
    BATCH_SIZE = 20
    epochs = 300

    print("Data: Breast cancer")

#binomial
elif dataset_nr == '2':
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
    print("Data: Sim-data 1, binomial")

#gaussian
elif dataset_nr == '3':
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

    print("Data: Abalone")

# gaussian
elif dataset_nr == '4':

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
    print("Data: Kepler's third law")
    BATCH_SIZE = 500
    epochs = 300
    class KeplerLoss(nn.Module):
        def __init__(self):
            super(KeplerLoss, self).__init__()

        def forward(self, output, target):
            criterion = nn.MSELoss(reduction='sum')
            loss = criterion(output, target)
            return loss + 5*np.pi
    
    custom_loss = KeplerLoss() #Define this to avoid getting negative values

#gaussian
elif dataset_nr == '5':
    df = pd.read_csv(
        'slice_localization.txt',
        header=1)
    
    df = df.dropna(axis = 0)

    x_df = df.iloc[:, 1:-1]
    y_df = df.iloc[:, -1]

    # Split:
    x_train = x_df.sample(frac=0.8, random_state=100)
    y_train = y_df.sample(frac=0.8, random_state=100)

    x_test = x_df[~x_df.index.isin(x_train.index)]
    y_test = y_df[~y_df.index.isin(y_train.index)]

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    family = 'gaussian'
    
    print("Data: CT slice localization, normal")

    BATCH_SIZE = 2000
    epochs = 300
    custom_loss = None
    LEARNIG_RATE = 0.01

    norm = False

#gaussian
elif dataset_nr == 'sim1':
    X = np.random.normal(0,1, (10000, 10))
    beta = np.array([0,0,0,0,0, 1.5, -4, 3, -0.2, 1])

    sigma = sigma if sigma is not None else 1
    noise = np.random.normal(0,sigma, size = 10000)

    y = X @ beta.T + noise


    x_test = X
    x_train = X

    y_test = y
    y_train = y

    family = 'gaussian'
    print("Data: sim-data 1, Normal; mu = 0, sigma =", sigma)

    N_GENERATIONS = 0
    norm = False


def normalize(data: np.array):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return np.divide((data - data_mean), data_std)

#print("SHAPE TRAIN:", x_train.shape, "SHAPE TEST:", x_test.shape)

# normailizing data:
if norm:
    x_train = normalize(x_train)
    x_test = normalize(x_test)

#dtrain = torch.from_numpy(np.column_stack((x_train, y_train)))
#dtest = torch.from_numpy(np.column_stack((x_test, y_test)))

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
    if family == 'binomial':
        return loss.item(), negative_log_likelihood.item(), np.mean(accs)
    else:
        return loss.item(), negative_log_likelihood.item()

def test(net, dtest, p, samples = 30, lower_q=0.05, upper_q=0.95, verbose=VERBOSE):
    family = net.family

    net.eval()
    data = dtest[:, 0:p - 1]
    target = dtest[:, -1].squeeze().float()
    outputs = net(data).detach().cpu().numpy()
    for _ in range(samples - 1):
        outputs = np.column_stack((outputs, net(data).detach().cpu().numpy()))

    outputs_mean = torch.tensor(outputs.mean(axis=1), device=DEVICE)
    outputs_upper = torch.quantile(outputs_mean,upper_q)
    outputs_lower = torch.quantile(outputs_mean,lower_q)

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
        if family =='binomial':
            printProgressBar(epochs, epochs, suffix = f"Complete. \tValidation accuracy: {acc}")
        else:
            printProgressBar(epochs, epochs, suffix = f"Complete. \tValidation loss (RMSE): {np.sqrt(loss.item()/outputs.shape[0])}")
    if family == 'binomial':
        return loss.item(), negative_log_likelihood.item(), acc, pred, outputs_upper.cpu().detach().numpy(), outputs_lower.cpu().detach().numpy()
    else:
        return loss.item(), negative_log_likelihood.item(), pred, outputs_upper.cpu().detach().numpy(), outputs_lower.cpu().detach().numpy()

#Standard prior, all covariates will be included in model:
def gamma_prior(comps,n,p):
    return torch.from_numpy(np.exp(-2*np.log1p(comps)-0.0001))

def var_inference(values, comps, iteration, family, prior_a = gamma_prior, verbose=VERBOSE, learnig_rate = LEARNIG_RATE):
    tmpx = torch.tensor(np.column_stack((values, y_train)), dtype=torch.float32).to(DEVICE)
    
    n, p = tmpx.shape
    prior_a = gamma_prior(comps, n, p)

    #print("Prior vals", prior_a)

    net = BayesianNetwork(p, prior_a, family=family).to(DEVICE)
    #print([e for e in net.parameters()])
    optimizer = optim.Adam(net.parameters(), lr=learnig_rate)

    for epoch in range(epochs):
        if family == 'binomial':
            nll, loss, acc = train(net, optimizer, tmpx, p, verbose=verbose)
        else:
            nll, loss = train(net, optimizer, tmpx, p, verbose=verbose)
        if verbose:
            print('epoch =', epoch)
            print(f'generation: {iteration}')
        else:
            printProgressBar(epoch, epochs, suffix = f"Complete")
    
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
        new = Linear(f'x{idx}', x_train[:,idx])
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
transformations = [Sigmoid, Sine, Cosine, Ln, Exp, x72, x52, x13]
tprobs = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])

# feature types
# mult, mod, proj, new:
fprobs = np.array([1/4, 1/4, 1/4, 1/4])

if dataset_nr == '1':
    transformations = [Sigmoid, Sine, Cosine, Ln, Exp]
    tprobs = [1/5,1/5,1/5,1/5,1/5]
if dataset_nr == '3':
    transformations = [Sigmoid, Sine, Cosine, Ln, Exp, x72, x52, x13]
    tprobs = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8 , 1/8, 1/8])
    fprobs = [1/4, 1/4, 1/4, 1/4]
    def gamma_prior(comps, n, p):
        return torch.from_numpy(np.exp(-np.log(n)*np.log1p(comps)-0.000001))
if dataset_nr == '4':
    transformations = [x13]
    tprobs = np.array([1])
    fprobs = np.array([1/3,1/3,0,1/3])
if dataset_nr == '5':
    transformations = [Exp]
    tprobs = np.array([1])
    fprobs = np.array([1/3,1/3,0,1/3])
if dataset_nr=='test':
    transformations = [Sigmoid, Sine, Cosine, Ln, Exp, x72, x52, x13]
    tprobs = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])
    fprobs = np.array([1/3, 1/3, 0, 1/3])

POPULATION_SIZE = x_train.shape[1] + EXTRA_FEATURES

F0 = {}
for j in range(x_train.shape[1]):
    F0[j] = {}
    F0[j]['feature'] = f'x{j}'
    F0[j]['mprob'] = 0
    F0[j]['complexity'] = 0
    F0[j]['values'] = x_train[:, j]
    F0[j]['type'] = Linear(f'x{j}', x_train[:,j])

# TRAINING:
if family == 'binomial':
    test_acc = []
test_loss = []

print("\nTraining/testing generation 0")
#Generation 0:
train_values = x_train
complexities = np.array([F0[k]['complexity'] for k in range(x_train.shape[1])])
mprobs, net = var_inference(train_values, complexities, 0, family)
for id, _ in F0.items():
        F0[id]['mprob'] = mprobs[id]

objects_list = [[F0[k]['type'] for k in range(x_train.shape[1])]]
mprobs_list = [mprobs]

predicted_vals, lower_vals, upper_vals, nets = [], [], [], [net]

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
                                                                mprobs / np.sum(mprobs), tprobs, family, x_test)
            
            collinear = check_collinearity(train_vals, col)
            if comp < COMPLEXITY_MAX and np.all(np.isfinite(train_vals)) and not collinear:
                F0[j]['mprob'] = 0
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

print("\nStarting training for", N_GENERATIONS, "generations:" )
#Generation 1, .... , N_GENERATIONS:
for i in range(1, N_GENERATIONS + 1):
    # VI step:
    print(f"Training/testing for generation {i}:")
    train_values = np.array([F0[k]['values'] for k in range(POPULATION_SIZE)]).T
    if norm:
        train_values = normalize(train_values)
    complexities = np.array([F0[k]['complexity'] for k in range(POPULATION_SIZE)])
    mprobs, net = var_inference(train_values, complexities, i, family)
    population = np.array([F0[k]['feature'] for k in range(POPULATION_SIZE)])
    mprobs_list.append(mprobs)
    if i < N_GENERATIONS:
        for id, _ in F0.items():
            F0[id]['mprob'] = mprobs[id]
            if F0[id]['mprob'] < 0.3: #Replace low probalility things
                if F0[id]['mprob'] < np.random.uniform(): #But keep with some probability
                    continue
                while (True):
                    feat, train_vals, test_vals, obj, comp = generate_feature(population, train_values, y_train, complexities, fprobs,
                                                                        mprobs / np.sum(mprobs), tprobs, family, test_values)
                    
                    collinear = check_collinearity(train_vals, train_values)

                    if comp < COMPLEXITY_MAX and np.all(np.isfinite(train_vals)) and not collinear:
                        F0[id]['complexity'] = comp
                        F0[id]['feature'] = feat
                        F0[id]['values'] = normalize(train_vals)
                        F0[id]['type'] = obj
                        train_values[:, id] = train_vals
                        test_values[:, id] = test_vals
                        break
                    continue
    
    objects_list.append([F0[k]['type'] for k in range(POPULATION_SIZE)])
    nets.append(net)

    # TESTING FOR CURRENT POPULATION:
    if norm:
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
    print('BEST LOSS (RMSE):', np.sqrt(np.min(test_loss)/x_test.shape[0]), f'(generation {test_loss.index(min(test_loss))})')

    #Best features
if dataset_nr == '4':
    for i, o in enumerate(objects_list[test_loss.index(min(test_loss))]):
            mprob = mprobs_list[test_loss.index(min(test_loss))][i]
            if mprob > 0.5:
                print(f'{o.feature}, mprob: {mprobs_list[test_loss.index(min(test_loss))][i]}')

sns.set()
fig1, ax1 = plt.subplots()
ax1.plot(predicted_vals[test_loss.index(min(test_loss))], color = "red", label = 'predicted', linestyle = " ", marker = '.')
ax1.axhline(upper_vals[test_loss.index(min(test_loss))], color = "blue", label = '0.95 quantile' ,linestyle='dashed')
ax1.axhline(lower_vals[test_loss.index(min(test_loss))], color= "blue", label = '0.05 quantile' ,linestyle='dashed')
ax1.plot(y_test, label = 'target', color = "black", linestyle = " ", marker = '.')
ax1.legend()
fig1.savefig('test_fig1.png')
print("Prediction plot saved as 'test_fig1.png'")
plt.close(fig1)

if family=='gaussian':
    if USE_FLOWS:
        fig2, ax2 = plt.subplots()
        sns.kdeplot(ax = ax2, x = predicted_vals[test_loss.index(min(test_loss))], label = 'predicted')
        #sns.kdeplot(up[test_loss.index(min(test_loss))], label = '0.95 quantile' ,linestyle='dashed', alpha =0.5)
        #sns.kdeplot(low[test_loss.index(min(test_loss))], label = '0.05 quantile' ,linestyle='dashed', alpha =0.5)
        sns.kdeplot(ax = ax2, x = y_test, label = 'target')
        ax2.legend()
        fig2.savefig('test_fig2.png')
        print("Density plot saved as 'test_fig2.png'")
        plt.close(fig2)
    else:
        fig3, ax3 = plt.subplots()
        sns.kdeplot(ax = ax3, x = predicted_vals[test_loss.index(min(test_loss))], label = 'predicted', marker = 'o')
        #sns.kdeplot(up[test_loss.index(min(test_loss))], label = '0.95 quantile' ,linestyle='dashed', alpha =0.5)
        #sns.kdeplot(low[test_loss.index(min(test_loss))], label = '0.05 quantile' ,linestyle='dashed', alpha =0.5)
        sns.kdeplot(ax = ax3, x = y_test, label = 'target', marker = 'o')
        ax3.legend()
        fig3.savefig('test_fig3.png')
        print("MF density plot saved as 'test_fig3.png'")
        plt.close(fig3) 

#Best features
# final_features = []
# final_vals = []
# final_betas = []
# beta = nets[test_loss.index(min(test_loss))].l1.weight_mu.detach().cpu().numpy()
# print(beta.shape)
# for i, o in enumerate(objects_list[test_loss.index(min(test_loss))]):
#     mprob = mprobs_list[test_loss.index(min(test_loss))][i]
#     if mprob > 0.5:
#         final_features.append(o.feature)
#         final_vals.append(o.values)
#         final_betas.append(beta[:, i])

# print(len(final_vals), len(final_betas))
# print(y_train.shape)

# cols = np.array([final_betas[i]*final_vals[i].reshape(-1,1) for i in range(len(final_vals))])

# print(cols.shape)

        #print(f'{o.feature}, mprob: {mprobs_list[test_loss.index(min(test_loss)) + 1][i]}')

if VERBOSE:
    print("Loss evolution:", test_loss)

et = time.time()

res = et-st
final_res = res/60

print("Total execution time:", final_res, "minutes")