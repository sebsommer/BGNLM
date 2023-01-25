#matplotlib inline
#import math
#import matplotlib.pyplot as plt
#import seaborn as sns
#import torch.nn.functional as F
# from tensorboardx import SummaryWriter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd

np.random.seed(1)

#select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

if (torch.cuda.is_available()):
    print("GPUs are used!")
else:
    print("CPUs are used!")

#setting batch size and epochs
BATCH_SIZE = 400
epochs = 200



dataset_nr = 2

# import the data
# taken from https://github.com/aliaksah/EMJMCMC2016/tree/master/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)

if dataset_nr == 1:
    x_df = pd.read_csv(
        'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/train.csv').iloc[:,1:-1]

    y_df = pd.read_csv(
        'https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/train.csv').iloc[:,-1]

    x_test = np.array(x_df)
    y_test = np.array(y_df)
    
    x_train = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/test.csv').iloc[:,1:-1].to_numpy()
    y_train = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/test.csv').iloc[:,-1].to_numpy()
    
    data_mean = x_train.mean(axis=0)
    data_std = x_train.std(axis=0)
    x_train = (x_train - data_mean) / data_std

elif dataset_nr == 2:
    x_df = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-X.txt',
    header=None)

    y_df = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/examples/Simulated%20Logistic%20Data%20With%20Multiple%20Modes%20(Example%203)/sim3-Y.txt',
    header=None)
    x = np.array(x_df)
    y = np.array(y_df)

    #Split:
    x_train = x_df.sample(frac=0.8, random_state=100)
    y_train = y_df.sample(frac=0.8, random_state=100)
    
    x_test = x_df[~x_df.index.isin(x_train.index)]
    y_test = y_df[~y_df.index.isin(y_train.index)]

    x_train, x_test, y_train, y_test = x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

# normailizing data:
data_mean = x_train.mean(axis=0)
data_std = x_train.std(axis=0)
x_train = (x_train - data_mean) / data_std

test_mean = x_test.mean(axis=0)
test_std = x_test.std(axis=0)
x_test = (x_test - test_mean) / test_std

dtrain = torch.tensor(np.column_stack((x_train, y_train)), dtype=torch.float32)

TRAIN_SIZE = len(dtrain)
NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, alpha_prior):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # weight variational parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight_sigma = torch.empty(size=self.weight_rho.shape)

        # weight priors = N(0,1)
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE)
        self.sigma_prior = (self.mu_prior + 1.).to(DEVICE)

        # model variational parameters
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(1.5, 2.5))
        self.alpha_q = torch.empty(size=self.lambdal.shape)

        # prior inclusion probability
        self.alpha_prior = alpha_prior.to(DEVICE)

        # bias variational parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.01, 0.01))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias_sigma = torch.empty(self.bias_rho.shape)

        # bias priors = N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.).to(DEVICE)

        # scalars
        self.kl = 0

    # forward path
    def forward(self, input, sample=False, calculate_log_probs=False):
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))  # -0.5)) #Added -0.5 in attempt to make stable
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if self.training or sample:
            e_w = self.weight_mu * self.alpha_q
            var_w = self.alpha_q * (self.weight_sigma ** 2 + (1 - self.alpha_q) * self.weight_mu ** 2)
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps

        else:  # posterior mean
            e_w = self.weight_mu * self.alpha_q
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            activations = e_b

        if self.training or calculate_log_probs:

            kl_bias = (torch.log(self.bias_sigma_prior / self.bias_sigma) - 0.5
                            + (self.bias_sigma ** 2 
                            + (self.bias_mu - self.bias_mu_prior) ** 2) / (2 * self.bias_sigma_prior ** 2)).sum()

            kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight_sigma) - 0.5 
                            + torch.log(self.alpha_q / self.alpha_prior)
                            + (self.weight_sigma ** 2
                            + (self.weight_mu - self.mu_prior) ** 2) / (2 * self.sigma_prior ** 2))
                            + (1 - self.alpha_q) * torch.log((1 - self.alpha_q) / (1 - self.alpha_prior))).sum()

            self.kl = kl_bias + kl_weight
        else:
            self.kl = 0

        return activations

class BayesianNetwork(nn.Module):
    def __init__(self, p, alpha_prior):
        super().__init__()
        # set the architecture
        self.l1 = BayesianLinear(p - 1, 1, alpha_prior)  # one layer with one neuron i.e. logistic regression
        self.loss = nn.BCELoss(reduction='sum')  # output is 0 or 1

    def forward(self, x, sample=False):
        x = self.l1(x, sample)
        x = torch.sigmoid(x)
        return x

    def kl(self):
        return self.l1.kl

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


#SEB CODE:

class Projection:
    def __init__(self, features, values, g, strategy = 1):
        self.alphas = np.ones(features.shape[0]+1)
        self.g = g
        values = np.column_stack([np.ones([values.shape[0], 1], dtype=np.float32), values])
        self.optimize(features, values, strategy)
            
    def optimize(self, features, values, strategy):
        if strategy == None:
            self.get_feature(features, g)
            self.get_values(values)
        if strategy == 1:
            tmpx = torch.tensor(np.column_stack((values, y_train)), dtype=torch.float32).to(DEVICE)

            data_mean = tmpx.mean(axis=0)[1:-1]
            data_std = tmpx.std(axis=0)[1:-1]
            tmpx[:,1:-1] = (tmpx[:,1:-1] - data_mean) / data_std

            p = tmpx.shape[1]

            model = LogisticRegression(p - 1, 1).to(DEVICE)
            criterion = nn.BCELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            
            # Gradient Descent
            for _ in range(200):
                _x = tmpx[:,:-1]
                _y = tmpx[:,-1]
                target = Variable(_y).to(DEVICE)
                data = Variable(_x).to(DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(torch.squeeze(outputs), target).to(DEVICE)
                loss.backward()
                optimizer.step()
            
            parm = {}
            for name, param in model.named_parameters():
                parm[name]=param.detach().cpu().numpy()

            alphas_out = parm['linear.weight'][0]
            self.alphas = alphas_out
            self.feature = self.get_feature(features, alphas_out)
            self.values = self.get_values(values)
        elif strategy == 2:
            #Not implemented
            return
        elif strategy == 3:
            #Not implemented
            """
            tmpx = torch.tensor(np.column_stack((values, y_train)), dtype=torch.float32).to(DEVICE)

            data_mean = tmpx.mean(axis=0)[1:-1]
            data_std = tmpx.std(axis=0)[1:-1]
            tmpx[:,1:-1] = (tmpx[:,1:-1] - data_mean) / data_std

            p = tmpx.shape[1]  # correct to an argument later

            model = LogisticRegression(p - 1, 1).to(DEVICE)
            criterion = nn.BCELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            for _ in range(100):
                _x = tmpx[:,:-1]
                _y = tmpx[:,-1]
                target = Variable(_y).to(DEVICE)
                data = Variable(_x).to(DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(torch.squeeze(outputs), target).to(DEVICE)
                loss.backward()
                optimizer.step()

            parm = {}
            for name, param in model.named_parameters():
                parm[name]=param.detach().cpu().numpy()
            alphas_out = parm['linear.weight'][0]
            self.alphas = alphas_out
            self.feature = self.get_feature(features, alphas_out)
            self.values = self.get_values(values)
            """
            return

    def get_feature(self, features, alphas_in):
        formula = f'({np.round(float(alphas_in[0]),4)}'
        for i, a in enumerate(alphas_in[1:]):
            if np.sign(a) > 0:
                formula += f'+{np.round(float(a),4)}*{features[i]}'
            else: 
                formula += f'{np.round(float(a),4)}*{features[i]}'
        formula += f')'
        self.f = self.g(formula)
        feature = str(formula)
        return feature


    def get_values(self, values):
        val = np.sum(self.alphas*values, axis = 1)
        return val

    def evaluate(self, values):
        values = np.column_stack([np.ones([values.shape[0], 1], dtype=np.float32), values])
        return np.sum(self.alphas*values, axis = 1)
    
    def __name__(self):
        return 'Projection'

class Modification:
    def __init__(self, feature, values, g):
        self.f = g(feature[0])
        self.feature = str(self.f)
        self.values = self.f.evaluate(values)
    
    def evaluate(self, values):
        return self.f.evaluate(values)

    def __name__(self):
        return 'Modification'

class Multiplication:
    def __init__(self, features, values):
        self.feature = f'{features[0]}*{features[1]}'
        self.values = values[:, 0] * values[:, 1]
    
    def evaluate(self, values):
        return values[:, 0] * values[:, 1]
    
    def __name__(self):
        return 'Multiplication'

def generate_feature(pop, val, fprobs, mprobs, tprobs, test_x):
    rand = np.random.choice(4, size=1, p=fprobs)
    values = val.copy()
    #print("!:", values.shape)

    if rand == 0:
        idx = np.random.choice(pop.shape[0], size=2, p=mprobs, replace=True)
        mult = Multiplication(pop[idx], values[:, idx])
        feat, val = mult.feature, mult.values
        if test_x is not None:
            test_val = mult.evaluate(test_x[:, idx])
        else:
            test_val = None
        obj = mult
        #print("0:",val.shape)

    elif rand == 1:
        idx = np.random.choice(pop.shape[0], size=1, p=mprobs)
        g = np.random.choice(transformations, p=tprobs)
        mod = Modification(pop[idx], values[:, idx], g)
        feat, val = mod.feature, mod.values.ravel()
        if test_x is not None:
            test_val = mod.evaluate(test_x[:, idx]).ravel()
        else:
            test_val = None
        obj = mod
        #print("1:",val.shape)

    elif rand == 2:
        g = np.random.choice(transformations, p=tprobs)
        s = np.random.choice([2,3,4], size=1)
        idx = np.random.choice(pop.shape[0], size=s, p=mprobs, replace=False)
        proj = Projection(pop[idx], values[:, idx], g)
        feat, val = proj.feature, proj.values
        if test_x is not None:
            test_val = proj.evaluate(test_x[:, idx])
        else:
            test_val = None
        obj = proj
        #print("2:",val.shape)

    elif rand == 3:
        idx = np.random.choice(x_train.shape[1])
        new = Linear(f'x{idx}')
        feat, val = str(new), new.evaluate(x_train[:, idx])
        if test_x is not None:
            test_val = test_x[:, idx]
        else:
            test_val = None
        obj = new
        #print("3:",val.shape)

    return feat, val, test_val, obj

#TODO: fix this:
def get_complexity(feat):
    complexity = 0
    for c in feat[1:]:
        if c == '(' or c == '*' or c == '+' or c== '-':
            complexity += 1
    return complexity

def check_collinearity(val, values):
    return np.any(np.any(val == values, axis=0))

class Non_linear():

    def __init__(self, feature):
        self.feature = feature

class Linear(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)
    def __str__(self):
        return str(self.feature)
    def evaluate(self, data):
        return data
    def __name__(self):
        return 'Linear'

class Sigmoid(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'sigma(' + str(self.feature) + ')'

    def evaluate(self, data):
        return 1 / 1 + np.exp(-data)

class Sine(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'sin(' + str(self.feature) + ')'

    def evaluate(self, data):
        return np.sin(data)

class Cosine(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'cos(' + str(self.feature) + ')'

    def evaluate(self, data):
        return np.cos(data)

class Exp(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'exp(-abs(' + str(self.feature) + '))'

    def evaluate(self, data):
        return np.exp(-np.abs(data))

class Ln(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return 'ln(abs(' + str(self.feature) + ') + 1)'

    def evaluate(self, data):
        return np.log(np.abs(data) + 1)

class x72(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return '(' + str(self.feature) + ')^7/2'

    def evaluate(self, data):
        return np.power(np.abs(data), 7 / 2)

class x52(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return '(' + str(self.feature) + ')^5/2'

    def evaluate(self, data):
        return np.power(np.abs(data), 5 / 2)

class x13(Non_linear):
    def __init__(self, feature):
        super().__init__(feature)

    def __str__(self):
        return '(' + str(self.feature) + ')^1/3'

    def evaluate(self, data):
        return np.power(np.abs(data), 1 / 3)

# nonlinear features. Should be decided by user.
transformations =  [Sigmoid, Sine, Cosine, Ln, Exp]
tprobs = np.array([1/5, 1/5, 1/5, 1/5, 1/5])

# feature types
# mult, mod, proj, new:
fprobs = np.array([1/4, 1/4, 1/4, 1/4])

# deciding population size and number og generations:
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