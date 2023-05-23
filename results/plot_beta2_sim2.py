import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from nonlinear import *
from features import *

N_GENERATIONS = 0
EXTRA_FEATURES = 0
VERBOSE = False
sigma = None
COMPLEXITY_MAX = 10
crit = 'SIM'
norm_y = False
USE_FLOWS = False
if USE_FLOWS:
    from flow_models import *
else:
    from mf_models import *

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

# select the device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

# setting batch size and epochs
N_IAFLAYERS = 2
BATCH_SIZE = 1000
epochs = 300
custom_loss = None
LEARNIG_RATE = 0.01



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
        if family == 'binomial':
            print(f'Validation Accuracy {acc}\n')
        else:
            print(f'Validation loss (RMSE): {np.sqrt(loss.item()/outputs.shape[0])}\n')
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
    
    #tmpx = torch.tensor(np.column_stack((normed_vals, y_train)), dtype=torch.float32)#.to(DEVICE)
    tmpx = torch.tensor(np.column_stack((normed_vals, y_train)), dtype=torch.float32)

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


beta_pred = np.zeros(9)
beta_upper = np.zeros(9)
beta_lower = np.zeros(9)
gamma = np.zeros(9)
for i, alp in enumerate([0.9, 0.8, 0.7 ,0.6, 0.5, 0.4, 0.3, 0.2, 0.1]):
    X = np.random.RandomState(1104).normal(0,1, (15000, 20))
    beta = np.array([0,0,0,0,0, 1.5, -4, 3, -0.2, 1,0,0,0,0,0, -2, 1.3, 0.3, -0.8, 3])
    noise = np.random.RandomState(110495).normal(0,1, size = 15000)
    X[:, 2] = (1-alp)*X[:, 2] + (alp)*X[:, 5]
    y = X @ beta.T + noise

    x_test = X
    x_train = X

    y_test = y
    y_train = y

    family = 'gaussian'

    TRAIN_SIZE = x_train.shape[0]
    NUM_BATCHES = TRAIN_SIZE / BATCH_SIZE

    try: x_cols
    except NameError: x_cols = [f'x{j}' for j in range(x_train.shape[1])]

    # --------------------------------- MODEL ---------------------------------

    F0 = {}
    for j in range(x_train.shape[1]):
        F0[j] = {}
        F0[j]['feature'] = x_cols[j]
        F0[j]['mprob'] = 0
        F0[j]['complexity'] = 0
        F0[j]['values'] = x_train[:, j]
        F0[j]['type'] = Linear(x_cols[j], x_train[:,j])

    # --------------------------------- TRAINING ---------------------------------

    if family == 'binomial':
        test_acc = []
    test_loss = []

    #Generation 0:
    #print("\nTraining/testing generation 0:")
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
        loss, nll, accuracy, means, uppers, lowers = test(net, test_data, p=x_train.shape[1] + 1)
        test_acc.append(accuracy)
        test_loss.append(loss)
    else:
        loss, nll, means, uppers, lowers = test(net, test_data, p=x_train.shape[1] + 1)
        test_loss.append(loss)

    predicted_vals.append(means)
    lower_vals.append(lowers)
    upper_vals.append(uppers)

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
        z = np.ones(beta_mu.shape[0])

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

    u = posterior_mu_beta[2] + 1.96*np.sqrt(beta_sigma[2])
    l = posterior_mu_beta[2] - 1.96*np.sqrt(beta_sigma[2])

    beta_pred[i] = posterior_mu_beta[2]
    print(posterior_mu_beta[2])
    beta_lower[i] = l
    beta_upper[i] = u
    gamma[i] = posterior_gamma[2]
    print(posterior_gamma[2])




sns.set()
plt.rcParams['figure.constrained_layout.use'] = True
fig, ax = plt.subplots()
true = [0 for _ in range(9)]
#ax.plot([1.5 for _ in range(9)], label = r'$\beta_5$', color = 'green')
ax.plot(true, label = 'truth', color = 'black')
#ax.plot(gamma, label = r'$\gamma$', color = 'blue')
ax.plot(beta_pred, label = rf'$\beta_{i}$', color = 'red')
ax.plot(beta_lower, color = 'red', linestyle= 'dashed', alpha=0.5)
ax.plot(beta_upper, color = 'red', linestyle= 'dashed', alpha=0.5)
ax.set_title(r'$\beta_3$')

plt.sca(ax)
plt.xticks(range(9), ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1'], rotation = 70, fontsize=9)
plt.ylim([-1, 1])
fig.suptitle(r'$\beta_3$ for simulation study 2')
fig.savefig('beta2_sim1_mf.png')
print("Prediction plot saved as 'beta2_sim1_mf.png'")
plt.close(fig)