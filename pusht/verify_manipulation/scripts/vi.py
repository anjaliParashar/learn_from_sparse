from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.models.gplvm import VariationalLatentVariable
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
import matplotlib.pylab as plt
import torch
import os
import numpy as np
from pathlib import Path
import sys
sys.path.append('/home/anjali/learn_from_sparse')
import pickle

# gpytorch imports
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior


device = torch.device('cpu')
#Collect experimental data
obs_np = np.zeros((1,2))
reward_np = np.zeros((1,1))
for i in range(7):
    file = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/gpr/experiments/'+str(i)+'.pkl','rb')
    data = pickle.load(file)
    file.close()
    obs_np = np.vstack((obs_np,data['obs'][:,2:4]))
    score = np.repeat(max(data['reward']),data['obs'].shape[0]).reshape((-1,1)) #+ 0.3*np.random.rand()
    reward_np = np.vstack((reward_np,score))
reward_np = 1-reward_np #reward to risk

#Collect data from simulations
file_sim = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/gpr/sim/pusht_gpr_train.pkl','rb')
data_sim = pickle.load(file_sim)
file_sim.close()

#Combine datasets from both sources
obs_full = torch.tensor(np.vstack((obs_np,np.array(data_sim['seed'])))).float().to(device)
reward_full = torch.tensor(np.vstack((reward_np,np.array(data_sim['risk']).reshape((-1,1))))).float().to(device)
Y = obs_full
# Setting manual seed for reproducibility
torch.manual_seed(73)
np.random.seed(73)


def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q = latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:,:latent_dim]))


class bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False):
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        # Initialise X with PCA or randn
        if pca == True:
             X_init = _init_pca(Y, latent_dim) # Initialise X to PCA
        else:
             X_init = torch.nn.Parameter(torch.randn(n, latent_dim))

        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

N = len(Y)
data_dim = Y.shape[1]
latent_dim = data_dim
n_inducing = 25
pca = False

# Model
model = bGPLVM(N, data_dim, latent_dim, n_inducing, pca=pca)

# Likelihood
likelihood = GaussianLikelihood(batch_shape=model.batch_shape)

# Declaring the objective to be optimised along with optimiser
# (see models/latent_variable.py for how the additional loss terms are accounted for)
mll = VariationalELBO(likelihood, model, num_data=len(Y))

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()}
], lr=0.01)


# Training loop - optimises the objective wrt kernel hypers, variational params and inducing inputs
# using the optimizer provided.

loss_list = []
batch_size = 100
for i in range(10000):
    batch_index = model._get_batch_idx(batch_size)
    optimizer.zero_grad()
    sample = model.sample_latent_variable()  # a full sample returns latent x across all N
    sample_batch = sample[batch_index]
    output_batch = model(sample_batch)
    loss = -mll(output_batch, reward_full[batch_index].T).sum()
    loss_list.append(loss.item())
    print(loss.item())
    loss.backward()
    optimizer.step()