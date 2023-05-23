import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import pandas as pd

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class NormalizingFlow(nn.Module):
    def __init__(self, latent, flows):
      super(NormalizingFlow, self).__init__()
      self.latent = latent
      self.flows = flows

    def latent_log_prob(self, z):
      return self.latent.log_prob(z)

    def latent_sample(self, num_samples: int = 1):
      return self.latent.sample((num_samples,))

    def sample(self, num_samples: int = 1):
      '''Sample a new observation x by sampling z from
      the latent distribution and pass through g.'''
      return self.g(self.latent_sample(num_samples))

    def f(self, x):
      '''Maps observation x to latent variable z.
      Additionally, computes the log determinant
      of the Jacobian for this transformation.
      Inveres of g.'''
      z, sum_log_abs_det = x, torch.ones(x.size(0)).to(x.device)
      for flow in self.flows:
        z, log_abs_det = flow.f(z)
        sum_log_abs_det += log_abs_det

      return z, sum_log_abs_det

    def g(self, z):
      '''Maps latent variable z to observation x.
      Inverse of f.'''
      with torch.no_grad():
        x = z
        for flow in reversed(self.flows):
               x = flow.g(x)

        return x

    def g_steps(self, z):
      '''Maps latent variable z to observation x
         and stores intermediate results.'''
      xs = [z]
      for flow in reversed(self.flows):
        xs.append(flow.g(xs[-1]))

      return xs

    def log_prob(self, x):
      '''Computes log p(x) using the change of variable formula.'''
      z, log_abs_det = self.f(x)
      return self.latent_log_prob(z) + log_abs_det

    def __len__(self) -> int:
      return len(self.flows)
    
class AffineCouplingLayer(nn.Module):
    def __init__(self,theta,split):
        super(AffineCouplingLayer, self).__init__()
        self.theta = theta
        self.split = split

    def f(self, x):
        '''f : x -> z. The inverse of g.'''
        x2, x1 = self.split(x)
        t, s = self.theta(x1)
        z1, z2 = x1, x2 * torch.exp(s) + t
        log_det = s.sum(-1)
        return torch.cat((z1, z2), dim=-1), log_det

    def g(self, z):
        '''g : z -> x. The inverse of f.'''
        z1, z2 = self.split(z)
        t, s = self.theta(z1)
        x1, x2 = z1, (z2 - t) * torch.exp(-s)
        return torch.cat((x2, x1), dim=-1)
  
class Conditioner(nn.Module):
    def __init__(self, in_dim: int, out_dim: int,num_hidden: int, hidden_dim: int, num_params: int):
        super(Conditioner, self).__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden)])

        self.num_params = num_params
        self.out_dim = out_dim
        self.dims = nn.Linear(hidden_dim, out_dim*num_params)

    def forward(self, x):
        x = F.leaky_relu(self.input(x))
        for h in self.hidden:
            x = F.leaky_relu(h(x))

        batch_params = self.dims(x).reshape(x.size(0), self.out_dim, -1)
        params = batch_params.chunk(self.num_params, dim=-1)
        return [p.squeeze(-1) for p in params]



def affine_coupling_flows(data_dim: int,hidden_dim: int,num_hidden: int,num_params: int,num_flows: int,device: str) -> nn.Module:

    def flow():
        split = partial(torch.chunk, chunks=2, dim=-1)
        theta = Conditioner(in_dim=data_dim // 2,out_dim=data_dim // 2,num_params=num_params,hidden_dim=hidden_dim,num_hidden=num_hidden)
        return AffineCouplingLayer(theta, split)

    latent = torch.distributions.MultivariateNormal(torch.zeros(data_dim).to(device),torch.eye(data_dim).to(device))
    flows = nn.ModuleList([flow() for _ in range(num_flows)])
    return NormalizingFlow(latent, flows)

def simple_iris_model(device='cuda'):
    return affine_coupling_flows(data_dim=x_train.shape[-1],hidden_dim=100,num_hidden=1,num_flows=5,num_params=2,device=device).to(device)

def train(model,train_loader,num_epochs, args) -> torch.Tensor:
    
    def _train(epoch, log_interval=50):
        model.train()
        losses = torch.zeros(len(train_loader))
        for i, x in enumerate(train_loader):
            optimizer.zero_grad()
            loss = -model.log_prob(x.to(device)).mean()
            loss.backward()
            losses[i] = loss.item()
            optimizer.step()

        return losses.mean().item()

    def log_training_results(loss, epoch, log_interval):
        if epoch % log_interval == 0:
            print("Training Results - Epoch: {}  Avg train loss: {:.2f}".format(epoch, loss))

            optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
            num_steps = len(train_loader)*num_epochs
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=args['lr_decay_interval'],gamma=args['lr_decay_rate'])

            train_losses = []
            for epoch in range(num_epochs+1):
                loss = _train(epoch)
                log_training_results(loss, epoch, log_interval=100)
                train_losses.append(loss)
                scheduler.step()

        return train_losses


def train_new_model(data, num_epochs, args):
    np.random.seed(1)
    model = simple_iris_model()
    loader = partial(torch.utils.data.DataLoader, batch_size=32, shuffle=True)
    loss = train(model=model,train_loader=loader(data),num_epochs=num_epochs,args=args)

    return model, loss

args={'lr_decay_interval': 400,'lr_decay_rate': .3,'lr': 1e-3}

def target_class(species: str):
    x = iris[iris['Species'] == species].drop(columns='Species')
    return torch.from_numpy(x.values).float()


def class_probs(x, class_models):
    with torch.no_grad(): 
        log_probs = torch.stack([m.log_prob(x.float().cuda()).cpu() for m in class_models])

    log_probs[torch.isnan(log_probs)] = -math.inf
    probs = log_probs.exp()
    probs /= probs.sum(0)
    return probs

def predict(class_probs):
    return class_probs.argmax(0)

def accuracy(y, y_hat):
    return y.eq(y_hat).float().mean().item()


x_test = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/test.csv').iloc[:, 1:-1].to_numpy()
y_test = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/test.csv').iloc[:, -1].to_numpy()

x_train = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/train.csv').iloc[:, 1:-1].to_numpy()
y_train = pd.read_csv('https://raw.githubusercontent.com/aliaksah/EMJMCMC2016/master/supplementaries/BGNLM/breast%20cancer/train.csv').iloc[:, -1].to_numpy()




labels = {0: 'No_C', 1: 'C'}

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
classes = list(labels.values())
train_with_args = partial(train_new_model, num_epochs=2000, args=args)
model_no, loss_no = train_with_args(torch.tensor(y_train))
model_yes, loss_yes = train_with_args(torch.tensor(y_train))

x = x_train
probs = class_probs(x, class_models=[model_no, model_yes])
pred_species = predict(probs)
print(f'Model accuracy: {accuracy(y_train, pred_species)}')