import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

Z_FLOW_TYPE = 'IAF'
R_FLOW_TYPE = 'IAF'



class PropagateFlow(nn.Module):
    def __init__(self, transform, dim, num_transforms):
        super().__init__()

        if transform == 'IAF':
            self.transforms = nn.ModuleList([IAF(dim, h_sizes=[dim, dim]) for _ in range(num_transforms)])
        elif transform == 'RealNVP':
            self.transforms = nn.ModuleList([RealNVP(dim, h_sizes=[512,512]) for _ in range(num_transforms)])
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
    def __init__(self, dim, h_sizes=[75,75]):
        super().__init__()
        self.net = MADE(nin=dim, hidden_sizes=h_sizes, nout=2*dim)
    
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
        self.lambdal = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-1, 1))
        self.alpha_q = torch.empty(size=self.lambdal.shape)

        # inclusion prior
        self.alpha_prior = alpha_prior.to(DEVICE)

        # bias mu and rho initialization
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.01, 0.01))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))

        # bias prior is N(0,1)
        self.bias_mu_prior = torch.zeros(out_features, device=DEVICE)
        self.bias_sigma_prior = (self.bias_mu_prior + 1.0).to(DEVICE)

        # initialization of the flow parameters
        # read MNF paper for more about what this means
        # https://arxiv.org/abs/1703.01961
        self.q0_mean = nn.Parameter(0.1 * torch.randn(in_features))
        self.q0_log_var = nn.Parameter(-9 + 0.1 * torch.randn(in_features)) #nn.Parameter(0.5 + 0.1 * torch.randn(in_features))
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
    def forward(self, input, sample=True, calculate_log_probs=False):
        ### perform the forward pass
        self.alpha_q = 1 / (1 + torch.exp(-self.lambdal))
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        if self.training or sample:
            z_k, _ = self.sample_z(input.size(0))
            e_w = self.weight_mu * self.alpha_q * z_k
            var_w = self.alpha_q * (self.weight_sigma ** 2 + (1 - self.alpha_q) * (self.weight_mu * z_k) ** 2)
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps  
        
        else: # posterior mean
            z_k, _ = self.sample_z(input.size(0))
            e_w = self.weight_mu * self.alpha_q * z_k
            var_w = self.alpha_q * (self.weight_sigma ** 2 + (1 - self.alpha_q) * (self.weight_mu * z_k) ** 2)
            e_b = torch.mm(input, e_w.T) + self.bias_mu
            var_b = torch.mm(input ** 2, var_w.T) + self.bias_sigma ** 2
            eps = torch.randn(size=(var_b.size()), device=DEVICE)
            activations = e_b + torch.sqrt(var_b) * eps

        if self.training or calculate_log_probs:
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
        else:
            self.kl = 0

        return activations
