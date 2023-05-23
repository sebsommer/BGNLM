import time
import sys

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

np.random.seed(100)

st = time.time()

def normalize(data: np.array):
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    return np.divide((data - data_mean), data_std)


# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


# setting batch size and epochs
BATCH_SIZE = 500
epochs = 4000
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
VERBOSE = False
COMPLEXITY_MAX = 10
norm = False

if dataset_nr=='3':
    BATCH_SIZE = 500
    epochs = 300
    LEARNIG_RATE = 0.01
    
    N_IAFLAYERS = 2

    N_GENERATIONS = int(sys.argv[2])
    EXTRA_FEATURES = int(sys.argv[3])
    VERBOSE = False
    COMPLEXITY_MAX = 10
    norm = True

if dataset_nr == 'sim1':
    sigma = np.sqrt(float(sys.argv[6]))
if dataset_nr == 'sim1_corr':
    alpha = float(sys.argv[6])
if dataset_nr == 'art' or dataset_nr == 'art_original':
    std = np.sqrt(float(sys.argv[6]))

crit = sys.argv[4]

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
else:
    print('No gamma-prior privided')
    sys.exit(2)

#breast cancer (binomial)
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
    
    BATCH_SIZE = 500
    epochs = 1000


    print("Data: Breast cancer")

#simx (binomial)
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

#abalone (gaussian)
elif dataset_nr == '3':
    te_ids = pd.read_csv("https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/abalone%20age/teid.csv", header = 1, sep = ";").iloc[:,-1] -1
    df = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/abalone%20age/abalone.data', header=None)
    x_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]
    

    dummies = pd.get_dummies(x_df.iloc[:,0])
    res = pd.concat([x_df, dummies], axis = 1)
    x_df = res.drop([0], axis = 1)
    x_cols=x_df.columns = ['Length', 'Diameter', 'Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight', 'Female', 'Infant', 'Male']

    x_df = x_df.drop(['Infant'], axis =1)
    x_cols = ['Length', 'Diameter', 'Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight', 'Female', 'Male']
    
    x_test = x_df.iloc[te_ids,:]
    y_test = y_df.iloc[te_ids]

    x_train = x_df[~x_df.index.isin(te_ids)]
    y_train = y_df[~y_df.index.isin(te_ids)]

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    family = 'gaussian'

#kepler (gaussian)
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

    #true = (x_test[:,2]*x_test[:,2]*x_test[:,4])**1/3

    #x_test = x_train = np.column_stack((x_train, true))



    family = 'gaussian'
    print("Data: Kepler's third law")
    BATCH_SIZE = 500
    epochs = 500
    LEARNIG_RATE = 0.005
    
    class KeplerLoss(nn.Module):
        def __init__(self):
            super(KeplerLoss, self).__init__()

        def forward(self, output, target):
            criterion = nn.MSELoss(reduction='sum')
            loss = criterion(output, target)
            return loss + 5*np.pi
    
    custom_loss = KeplerLoss() #Define this to avoid getting negative values

#ct slice (gaussian)
elif dataset_nr == '5':
    df = pd.read_csv(
        'slice_localization.csv',
        header=1)
    
    #df = df.dropna()

    x_df = df.iloc[:, :-1]
    y_df = df.iloc[:, -1]

    x_df = x_df.loc[:, x_df.std(axis=0)>0.01]

    # Split:
    x_train = x_df.sample(frac=0.8, random_state=1104)
    y_train = y_df.sample(frac=0.8, random_state=1104)

    x_test = x_df[~x_df.index.isin(x_train.index)]
    y_test = y_df[~y_df.index.isin(y_train.index)]

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    #print(np.min(x_train[:, 276]), np.max(x_train[:, 276]), np.mean(x_train[:, 276]), np.std(x_train[:, 276]))
    #print(x_train[:, 276][x_train[:, 276] > -0.25])

    print(x_train.shape, x_test.shape)

    #x_train = x_train[:, x_train.std(axis=0)>0.01]
    #x_test = x_test[:, x_train.std(axis=0)>0.01]

    family = 'gaussian'
    
    print("Data: CT slice localization, normal")

    BATCH_SIZE = 1000
    epochs = 350
    #custom_loss = None
    LEARNIG_RATE = 0.01

    norm = True

#spam (binomial)
elif dataset_nr == '6':
    idx = pd.read_csv("spam_idx.txt", header = None).squeeze()
    df = pd.read_csv('spam.txt', header=None)
    x_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]


    x_test = x_df[idx==1]
    y_test = y_df[idx==1]

    x_train = x_df[idx==0]
    y_train = y_df[idx==0]

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

    family = 'binomial'
    
    BATCH_SIZE = 500
    epochs = 600

    print("Data: Spam")
    norm = True

#gaussian
elif dataset_nr == 'sim1':
    X = np.random.normal(0,1, (15000, 20))
    beta = np.array([0,0,0,0,0, 1.5, -4, 3, -0.2, 1,0,0,0,0,0, -2, 1.3, 0.3, -0.8, 3])

    noise = np.random.normal(0,sigma, size = 15000)

    #X[:, 2] = (1-alpha)*X[:, 2] + (alpha)*X[:, 5]

    y = X @ beta.T + noise

    x_test = X
    x_train = X

    y_test = y
    y_train = y

    family = 'gaussian'
    print("Data: sim-data 1, Normal; mu = 0, sigma =", sigma)
    BATCH_SIZE = 1000
    epochs = 300
    N_GENERATIONS = 0
    norm = False

elif dataset_nr == 'sim1_corr':
    X = np.random.RandomState(1104).normal(0,1, (15000, 20))
    beta = np.array([0,0,0,0,0, 1.5, -4, 3, -0.2, 1,0,0,0,0,0, -2, 1.3, 0.3, -0.8, 3])

    noise = np.random.RandomState(110495).normal(0,1, size = 15000)

    X[:, 2] = (1-alpha)*X[:, 2] + (alpha)*X[:, 5]

    y = X @ beta.T + noise


    x_test = X
    x_train = X

    y_test = y
    y_train = y

    family = 'gaussian'
    print("Data: sim-data 1, corr, alpha=", alpha)
    BATCH_SIZE = 1000
    epochs = 300
    N_GENERATIONS = 0
    N_IAFLAYERS = 10
    norm = False

#gaussian
elif dataset_nr == 'art':

    X = pd.read_csv('art.txt', header = 0)

    x_cols = X.iloc[:,2:14].columns.to_numpy()
    #Y = X.iloc[:,14:]

    x = normalize(X.iloc[:,2:14].to_numpy())

    mu = 0.1 + normalize(p05('x1').evaluate(X['x1'])) + x[:,0] + normalize(pm05('x3').evaluate(X['x3'])) + normalize(p0pm05('x3').evaluate(X['x3'])) + x[:,8] + normalize(pm1('x5').evaluate(X['x5'])) + normalize(p1('x6').evaluate(X['x6'])) + x[:,6] + x[:,7]

    y = np.random.normal(mu, scale = std)

    x_train = x
    x_test = x
    y_train = y
    y_test = y

    family = 'gaussian'

    print('Data: Sim-data 2, sigma =', std)

    BATCH_SIZE = 500
    epochs = 4000
    LEARNIG_RATE = 0.01

elif dataset_nr == 'art_original':
    X = pd.read_csv('art.txt', header = 0)
    

    x_cols = X.iloc[:,2:14].columns.to_numpy()
    #Y = X.iloc[:,14:]

    x = normalize(X.iloc[:,2:14].to_numpy())

    mu = 0.1 + normalize(p05('x1').evaluate(x[:,0])) + x[:,0] + x[:,2] + x[:,8] + normalize(xm15('x5').evaluate(x[:,3])) + normalize(Ln1p('x6').evaluate(x[:,4])) + x[:,6] + x[:,7]

    y = mu + np.random.RandomState(1104).normal(0, scale = std, size = X.shape[0])

    x_train = x_test = x
    y_train = y_test = normalize(y)


    family = 'gaussian'

    BATCH_SIZE = 250
    epochs = 1000
    LEARNIG_RATE = 0.01
    print('Data: Sim-data 2, sigma =', std)

elif dataset_nr == 'sim2test':
    try: sim2y
    except NameError:
        print("Need to specify y! Options are:\n y_50\n y_25\n y_10 \n y_5 \n y_1 \n y_0.1\n y_0.01\n y_0.001 \n y_1e-4 \n y_1e-5 \n y_1e-6\n y_1e-7\n y_1e-7 \n y_1e-8 \n y_1e-9 \n y_1e-10 \n")
        sim2y = input("Provide y for sim study 2: ")
    X = pd.read_csv('art.txt', header = 0)
    
    mu = 0.1 + normalize(p05('x1').evaluate(X['x1'])) + X['x1'] + normalize(pm05('x3').evaluate(X['x3'])) + normalize(p0pm05('x3').evaluate(X['x3'])) + normalize(X['x4a']) + normalize(pm1('x5').evaluate(X['x5'])) + normalize(p1('x6').evaluate(X['x6'])) + normalize(X['x8']) + normalize(X['x10'])

    x_cols = X.iloc[:,2:14].columns.to_numpy()
    Y = X.iloc[:,14:]
    
    X = np.column_stack((X.iloc[:,2:14].to_numpy(), normalize(p05('x1').evaluate(X['x1'])), normalize(pm05('x3').evaluate(X['x3'])), normalize(p0pm05('x3').evaluate(X['x3'])), normalize(pm1('x5').evaluate(X['x5'])), normalize(p1('x6').evaluate(X['x6']))))
    

    sds = float(sim2y)
    y = np.random.normal(loc = mu, scale = sds, size = X.shape[0])

    x_train = x_test = X
    y_train = y_test = y

    x_cols = np.array(["x1","x2","x3","x5","x6","x7","x8","x10","x4a","x4b","x9a","x9b", "p05_1", "pm05_3", "p0pm05_3", "pm1_5", "p1_6"])


    family = 'gaussian'

    BATCH_SIZE = 50
    epochs = 2000
    LEARNIG_RATE = 0.01
    N_IAFLAYERS = 5
    COMPLEXITY_MAX = 2

try: x_cols
except NameError: x_cols = [f'x{j}' for j in range(x_train.shape[1])]


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
    if family == 'binomial':
        return loss.item(), negative_log_likelihood.item(), acc, pred, outputs_upper.cpu().detach().numpy(), outputs_lower.cpu().detach().numpy()
    else:
        return loss.item(), negative_log_likelihood.item(), pred, outputs_upper.cpu().detach().numpy(), outputs_lower.cpu().detach().numpy()

def var_inference(values, comps, iteration, family, prior_func_a = gamma_prior, verbose=VERBOSE, learnig_rate = LEARNIG_RATE):
    def norm(x):
        x_mean = x.mean(axis=0)
        x_std = x.std(axis=0)
        return np.divide((x - x_mean), x_std)
    
    normed_vals = norm(values)

    tmpx = torch.tensor(np.column_stack((normed_vals, y_train)), dtype=torch.float32).to(DEVICE)

    n, p = tmpx.shape
    prior_a = prior_func_a(comps, n, p - 1)

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
    
    a = net.l1.alpha_q.data.detach().cpu().numpy().squeeze()

    return a, net


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
    idx = np.random.choice(v.shape[0], size = v.shape[0]//2, replace = False)
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

if dataset_nr == '1':
    #transformations = [Sigmoid, Sine, Cosine, Ln, Exp]
    #transformations = [Gauss, Sine, Tanh, Atan]
    #tprobs = [1/4,1/4,1/4,1/4]
    def gamma_prior(comps, n, p):
        return torch.from_numpy(np.exp(-2*np.log1p(comps)-0.000001))
elif dataset_nr == '3':
    transformations = [Sigmoid, Ln1p, Expm1, x72, x52, x13]
    tprobs = np.array([1/len(transformations) for e in transformations])
    fprobs = [1/4, 1/4, 1/4, 1/4]
elif dataset_nr == '4':
    transformations = [x13]
    tprobs = np.array([1])
    fprobs = np.array([1/3,1/3,0,1/3])
elif dataset_nr == '6':
    transformations = [Gauss, Sigmoid, Sine, Cosine, Tanh, Atan, Ln, Exp, x72, x52, x13, Imean]
    tprobs = np.array([1/len(transformations) for e in transformations])
    def gamma_prior(comps, n, p):
        return torch.from_numpy(np.exp(-2*np.log1p(comps)))
elif dataset_nr=='test':
    transformations = [Sigmoid, Sine, Cosine, Ln, Exp, x72, x52, x13]
    tprobs = np.array([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8])
    fprobs = np.array([1/3, 1/3, 0, 1/3])
elif dataset_nr == 'art' or dataset_nr == 'sim2test' or dataset_nr == 'art_original':
    transformations = [p1,pm1, pm2, pm05, p05, p2, p3, p0p0, p0pm1, p0pm2, p0p05, p0pm05, p0p1, p0p2, p0p3]
    tprobs = [1./len(transformations) for _ in range(len(transformations))]
    fprobs = [1/2, 1/2]
    def generate_feature(pop, values, target, comps, fprobs, mprobs, tprobs, family, train_x, test_x,x_cols):
        rand = np.random.choice(2, size=1, p=fprobs)
        
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

POPULATION_SIZE = x_train.shape[1] + EXTRA_FEATURES

F0 = {}
for j in range(x_train.shape[1]):
    F0[j] = {}
    F0[j]['feature'] = x_cols[j]
    F0[j]['mprob'] = 0
    F0[j]['complexity'] = 0
    F0[j]['values'] = x_train[:, j]
    F0[j]['type'] = Linear(x_cols[j], x_train[:,j])

if dataset_nr == 'sim2test':
    F0 = {}
    for j in range(x_train.shape[1]-5):
        F0[j] = {}
        F0[j]['feature'] = x_cols[j]
        F0[j]['mprob'] = 0
        F0[j]['complexity'] = 0
        F0[j]['values'] = x_train[:, j]
        F0[j]['type'] = Linear(x_cols[j], x_train[:,j])
    for j in range(x_train.shape[1]-5, x_train.shape[1]):
        F0[j] = {}
        F0[j]['feature'] = x_cols[j]
        F0[j]['mprob'] = 0
        F0[j]['complexity'] = 1
        F0[j]['values'] = x_train[:, j]
        F0[j]['type'] = Linear(x_cols[j], x_train[:,j])

# TRAINING:
if family == 'binomial':
    test_acc = []
test_loss = []

#Generation 0:
train_values = x_train
complexities = np.array([F0[k]['complexity'] for k in range(x_train.shape[1])])
mprobs, net = var_inference(train_values, complexities, 0, family, gamma_prior)
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
                if norm:
                    train_vals = normalize(train_vals)
                    test_vals = normalize(test_vals)
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
for i in range(1, N_GENERATIONS + 1):
    # VI step:
    train_values = np.array([F0[k]['values'] for k in range(POPULATION_SIZE)]).T
    population = np.array([F0[k]['feature'] for k in range(POPULATION_SIZE)])
    complexities = np.array([F0[k]['complexity'] for k in range(POPULATION_SIZE)])
    
    #Training current population:
    mprobs, net = var_inference(train_values, complexities, i, family, gamma_prior)
    mprobs_list.append(mprobs)
    
    #Storing mprobs:
    for id, _ in F0.items():
        F0[id]['mprob'] = mprobs[id]
    if i < N_GENERATIONS:
        #Replacing all low probability things for next generation: 
        for id, _ in F0.items():
            #print("ID:", id)
            if F0[id]['mprob'] < 0.3:
                stop = False
                if F0[id]['mprob'] > np.random.uniform(): #Keep bad features with some probability
                    stop = True
                while not stop:
                    #print(id, "mprob", F0[id]['mprob'], "OLD: ", F0[id]['feature'])
                    
                    #Finding new feature:
                    feat, train_vals, test_vals, obj, comp = generate_feature(population, train_values, y_train, complexities, fprobs,
                                                                        mprobs / np.sum(mprobs), tprobs, family, x_train, test_values, x_cols)
                    
                    collinear = check_collinearity(train_vals, train_values)
                    #print(id, "mprob", F0[id]['mprob'], "NEW: ", feat, "\n----------\n")

                    #Replace if it is not already in population:
                    if comp <= COMPLEXITY_MAX and not collinear and np.all(np.isfinite(train_vals)):
                        if norm:
                            train_vals = normalize(train_vals)
                            test_vals = normalize(test_vals)
                        F0[id]['mprob'] = None
                        F0[id]['complexity'] = comp
                        F0[id]['feature'] = feat
                        F0[id]['values'] = train_vals
                        F0[id]['type'] = obj
                        train_values[:, id] = train_vals
                        test_values[:, id] = test_vals
                        stop = True
            else:
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

if VERBOSE:
    print("Loss evolution:", test_loss)

best_id = test_loss.index(min(test_loss))


best_gen = {}
best_gen['gen_nr'] = best_id
best_gen['RMSE'] = np.sqrt(test_loss[best_id]/x_test.shape[0])
best_gen['features'] = np.array([o.feature for o in objects_list[best_id]])
best_gen['mprobs'] = mprobs_list[best_id]
best_gen['pred'] = predicted_vals[best_id]
best_gen['ci'] = (lower_vals[best_id], upper_vals[best_id])
best_gen['net'] = nets[best_id]
if family == 'binomial':
    best_gen['result'] = np.max(test_acc)
    best_gen['FNR'] = np.sum(np.logical_and(predicted_vals[best_id] == 0, y_test == 1))/np.sum(y_test == 1)
    best_gen['FPR'] = np.sum(np.logical_and(predicted_vals[best_id] == 1, y_test == 0))/np.sum(y_test == 0)
else:
    best_gen['result'] = np.sqrt(np.min(test_loss)/x_test.shape[0]) 
    best_gen['MAE'] = np.mean(np.abs(y_test - predicted_vals[best_id]))
    best_gen['corr'] = np.corrcoef(predicted_vals[best_id], y_test)[0][1]

print(best_gen['result'])

et = time.time()

res = et-st
final_res = res/60

print("Total execution time:", final_res, "minutes")