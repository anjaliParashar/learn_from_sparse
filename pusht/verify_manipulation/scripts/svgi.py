import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
import sys
from gpytorch.variational import VariationalStrategy
import torch
from matplotlib import pyplot as plt
import numpy as np
import ipdb
import pickle
import tqdm

sys.path.append('/home/anjali/learn_from_sparse')

device = torch.device('cuda')
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

train_dataset = TensorDataset(obs_full, reward_full)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

test_dataset = TensorDataset(obs_full, reward_full)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

inducing_points = obs_full[::2, :]
model = GPModel(inducing_points=inducing_points)
likelihood = gpytorch.likelihoods.GaussianLikelihood()

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


num_epochs = 1 
model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.0001)

# Our loss object. We're using the VariationalELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=reward_full.size(0))


epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        minibatch_iter.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()

model.eval()
likelihood.eval()
means = torch.tensor([0.])
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        preds = model(x_batch)
        means = torch.cat([means, preds.mean.cpu()])
means = means[1:]
print('Test MAE: {}'.format(torch.mean(torch.abs(means - reward_np.cpu()))))