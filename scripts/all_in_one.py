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
Z_FLOW_TYPE = 'IAF'
R_FLOW_TYPE = 'IAF'

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out

class PropagateFlow(nn.Module):
    def __init__(self, transform, dim, num_transforms):
        super().__init__()

        if transform == 'IAF':
            self.transforms = nn.ModuleList([IAF(dim, h_sizes=[75,75,75,75]) for i in range(num_transforms)])
        else:
            print('Transform not implemented')

    def forward(self, z):
        logdet = 0
        for f in self.transforms:
            z = f(z)
            logdet += f.log_det()
        return z, logdet

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.LeakyReLU(0.1),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1: return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)

class IAF(nn.Module):
    ### for the variable selection simulation it works best with smaller networks
    def __init__(self, dim, h_sizes=[75, 75]):
        super().__init__()
        self.net = MADE(nin=dim, hidden_sizes=h_sizes, nout=2 * dim)
    
    def forward(self, x):  # x -> z
        out = self.net(x)
        first_half = int(out.shape[-1] / 2)

        if out.dim() == 2:  # if we have a minibatch of z
            shift = out[:, :first_half]
            scale = out[:, first_half:]
        else:  # otherwise if we are estimating r
            shift = out[:first_half]
            scale = out[first_half:]
        self.gate = torch.sigmoid(scale)
        z = x * self.gate + (1 - self.gate) * shift

        return z

    def log_det(self):
        return (torch.log(self.gate + 1e-8)).sum(-1)  # avoid log (0)

class BayesianLinearFlow(nn.Module):
    def __init__(self, in_features, out_features, num_transforms, alpha_prior):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # weight mu and rho initialization
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))

        # weight prior is N(0,1) for all the weights
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE) + 0.0
        self.sigma_prior = (self.mu_prior + 1.).to(DEVICE)

        # posterior inclusion initialization
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(1.5, 2.5))
        self.alpha_q = torch.empty(size=self.lambdal.shape)

        # inclusion prior
        self.alpha_prior = alpha_prior.to(DEVICE)

        # bias mu and rho initialization
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.01, 0.01))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        # bias prior is also N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.0).to(DEVICE)

        # initialization of the flow parameters
        # read MNF paper for more about what this means
        # https://arxiv.org/abs/1703.01961
        self.q0_mean = nn.Parameter(0.1 * torch.randn(in_features))
        self.q0_log_var = nn.Parameter(-9 + 0.1 * torch.randn(in_features))
        self.r0_c = nn.Parameter(0.1 * torch.randn(in_features))
        self.r0_b1 = nn.Parameter(0.1 * torch.randn(in_features))
        self.r0_b2 = nn.Parameter(0.1 * torch.randn(in_features))

        # one flow for z and one for r(z|w,gamma)
        self.z_flow = PropagateFlow(Z_FLOW_TYPE, in_features, num_transforms)
        self.r_flow = PropagateFlow(R_FLOW_TYPE, in_features, num_transforms)

        self.kl = 0
        self.z = 0

    def sample_z(self, batch_size=1):
        q0_std = self.q0_log_var.exp().sqrt().repeat(batch_size, 1)
        epsilon_z = torch.randn_like(q0_std)
        self.z = self.q0_mean + q0_std * epsilon_z
        zs, log_det_q = self.z_flow(self.z)
        return zs[-1], log_det_q.squeeze()

        # forward path

    def forward(self, input, sample=False, calculate_log_probs=False):
        ### perform the forward pass
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        z_k, _ = self.sample_z(input.size(0))
        e_w = self.weight_mu * self.alpha_q * z_k
        var_w = self.alpha_q * (self.weight_sigma ** 2 + (1 - self.alpha_q) * (self.weight_mu * z_k) ** 2)
        e_b = torch.mm(input, e_w.T) + self.bias_mu
        var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
        eps = torch.randn(size=(var_b.size()), device=DEVICE)
        activations = e_b + torch.sqrt(var_b) * eps

        ### compute the ELBO
        z2, log_det_q = self.sample_z()
        W_mean = z2 * self.weight_mu * self.alpha_q
        W_var = self.alpha_q * (self.weight_sigma ** 2 + (1 - self.alpha_q) * (self.weight_mu * z2) ** 2)
        log_q0 = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * self.q0_log_var
                  - 0.5 * ((self.z - self.q0_mean) ** 2 / self.q0_log_var.exp())).sum()
        log_q = -log_det_q + log_q0

        act_mu = self.r0_c @ W_mean.T
        act_var = self.r0_c ** 2 @ W_var.T
        act_inner = act_mu + act_var.sqrt() * torch.randn_like(act_var)
        act = torch.tanh(act_inner)
        mean_r = self.r0_b1.outer(act).mean(-1)  # eq (9) from MNF paper
        log_var_r = self.r0_b2.outer(act).mean(-1)  # eq (10) from MNF paper
        z_b, log_det_r = self.r_flow(z2)
        log_rb = (-0.5 * torch.log(torch.tensor(math.pi)) - 0.5 * log_var_r - 0.5 * ((z_b[-1] - mean_r) ** 2 / log_var_r.exp())).sum()
        log_r = log_det_r + log_rb

        kl_bias = (torch.log(self.bias_sigma_prior / self.bias_sigma) - 0.5 
                    + (self.bias_sigma ** 2 + (self.bias_mu - self.bias_mu_prior) ** 2) / (2 * self.bias_sigma_prior ** 2)).sum()

        kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight_sigma)- 0.5 
                    + torch.log(self.alpha_q / self.alpha_prior) 
                    + (self.weight_sigma ** 2 + (self.weight_mu * z2 - self.mu_prior) ** 2) / (2 * self.sigma_prior ** 2))
                    + (1 - self.alpha_q) * torch.log((1 - self.alpha_q) / (1 - self.alpha_prior))).sum()

        self.kl = kl_bias + kl_weight + log_q - log_r

        return activations

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
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
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
                       + (self.bias_sigma ** 2 + (self.bias_mu - self.bias_mu_prior) ** 2) / (2 * self.bias_sigma_prior ** 2)).sum()

            kl_weight = (self.alpha_q * (torch.log(self.sigma_prior / self.weight_sigma) - 0.5
                        + torch.log(self.alpha_q / self.alpha_prior)
                        + (self.weight_sigma ** 2 + (self.weight_mu - self.mu_prior) ** 2) / (2 * self.sigma_prior ** 2))
                        + (1 - self.alpha_q) * torch.log((1 - self.alpha_q) / (1 - self.alpha_prior))).sum()

            self.kl = kl_bias + kl_weight
        else:
            self.kl = 0

        return activations

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

class Projection:
    def __init__(self, features, values, target, g, family, comp, strategy=1, epochs=200):
        self.alphas = np.ones(features.shape[0] + 1)
        self.g = g
        self.family = family
        self.complexity = comp
        values = np.column_stack([np.ones([values.shape[0], 1], dtype=np.float32), values])
        self.optimize(features, values, target, epochs, strategy)

    def optimize(self, features, values, target, epochs, strategy):
        if strategy == 1:
            tmpx = torch.tensor(np.column_stack((values, target)), dtype=torch.float32).to(DEVICE)
            p = tmpx.shape[1]
            if self.family == 'binomial':
                model = LogisticRegression(p - 1, 1).to(DEVICE)
                criterion = nn.BCELoss()
            elif self.family == 'gaussian':
                model = LinearRegression(p - 1, 1).to(DEVICE)
                criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # Gradient Descent
            for _ in range(epochs):
                _x = tmpx[:, :-1]
                _y = tmpx[:, -1]
                target = Variable(_y).to(DEVICE)
                data = Variable(_x).to(DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(torch.squeeze(outputs), target).to(DEVICE)
                loss.backward()
                optimizer.step()

            parm = {}
            for name, param in model.named_parameters():
                parm[name] = param.detach().cpu().numpy()
            alphas_out = parm['linear.weight'][0]
            self.alphas = alphas_out
            self.feature = self.get_feature(features, alphas_out)
            self.values = self.get_values(values)
        elif strategy == 2:
            # Not implemented
            return
        elif strategy == 3:
            # Not implemented
            return

    def get_feature(self, features, alphas_in):
        formula = f'({np.round(float(alphas_in[0]), 4)}'
        for i, a in enumerate(alphas_in[1:]):
            if np.sign(a) > 0:
                formula += f'+{np.round(float(a), 4)}*{features[i]}'
            else:
                formula += f'{np.round(float(a), 4)}*{features[i]}'
        formula += f')'
        self.f = self.g(formula)
        feature = str(formula)
        return feature

    def get_values(self, values):
        val = np.sum(self.alphas * values, axis=1)
        return val

    def evaluate(self, values):
        values = np.column_stack([np.ones([values.shape[0], 1], dtype=np.float32), values])
        return np.sum(self.alphas * values, axis=1)

    def __name__(self):
        return 'Projection'

class Modification:
    def __init__(self, feature, values, g, comp):
        self.f = g(feature[0])
        self.feature = str(self.f)
        self.values = self.f.evaluate(values)
        self.complexity = comp + 1

    def evaluate(self, values):
        return self.f.evaluate(values)

    def __name__(self):
        return 'Modification'

class Multiplication:
    def __init__(self, features, values, comp):
        self.feature = f'{features[0]}*{features[1]}'
        self.values = values[:, 0] * values[:, 1]
        self.complexity = comp + 1

    def evaluate(self, values):
        return values[:, 0] * values[:, 1]

    def __name__(self):
        return 'Multiplication'

class Linear:
    def __init__(self, feature):
        self.feature = feature
        self.complexity = 0

    def __str__(self):
        return str(self.feature)

    def evaluate(self, data):
        return data

    def __name__(self):
        return 'Linear'

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

class Non_linear():
    def __init__(self, feature):
        self.feature = feature

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
        return np.log1p(np.abs(data))

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
