import sys
sys.path.append('/home/anjali/learn_from_sparse')
import numpy as np
from pusht.verify_manipulation.utils.metropolis_hastings import gaussian_proposal,gaussian_proposal_prob,mcmc_mh,get_means
import scipy.stats as st
from pusht.verify_manipulation.utils.inference_final import get_trajectory,normalize_data
import argparse
import ipdb
import pickle
import torch
import matplotlib.pyplot as plt

from flowgmm.flow_ssl.realnvp.realnvp import RealNVPTabular
from flowgmm.flow_ssl.distributions import SSLGaussMixture
from flowgmm.flow_ssl import FlowLoss

#import data from Normalizing flows to prepare prior
model = RealNVPTabular(num_coupling_layers=20, in_dim=2, num_layers=1, hidden_dim=32)
PATH = "/home/anjali/learn_from_sparse/flowgmm/experiments/synthetic_data/pusht_noise.pt"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

r = 2
n_classes = 2
means = get_means(n_classes,r)
prior = SSLGaussMixture(means=means)
loss_fn = FlowLoss(prior)
grid_points = 150
grid_freq = 10
z_lims = np.array([-8, 8])
x_lims = np.array([0, 600])
line_z = np.linspace(*z_lims, grid_points)
line_x = np.linspace(*x_lims, grid_points)
xx_z, yy_z = np.meshgrid(line_z, line_z)
xx_x, yy_x = np.meshgrid(line_x, line_x)

#Demonstrate the sampling from normal isotropic distribution to the clustered search space
""" plt.figure()
zs = []
for i in range(len(means)):
    z = loss_fn.prior.sample((1000,), gaussian_id=i).numpy()
    print(loss_fn.prior.log_prob_(torch.tensor(z),gaussian_id=i))
    zs.append(z)
    plt.scatter(z[:, 0], z[:, 1], cmap=plt.cm.rainbow)

plt.figure()
for z in zs:
    x = model.inverse(torch.from_numpy(z).float()).detach().numpy()
    plt.scatter(x[:, 0], x[:, 1], cmap=plt.cm.rainbow)
    plt.xlim(x_lims)
    plt.ylim(x_lims)
plt.show() """

#Returns p(Z|X) using custom defined likelihood and custom prior p(Z)
def custom_posterior(Z): 
    # Assumes gaussian log likelihood and gaussian prior
    #loglik = np.sum(np.log(st.norm(loc=Z, scale=1).pdf(X)))
    #logprior = np.log(st.norm(loc=0, scale=1).pdf(Z))
    Zz = model.inverse(torch.from_numpy(Z.reshape(1,-1)).float()).detach().numpy()
    _,obs_list,reward = get_trajectory(Zz[:,0][0],Zz[:,1][0])
    X = np.array(obs_list)[:,2:4]
    loglike = get_loglike(X)
    #logprior = loss_fn.prior.log_prob_(torch.tensor(z),gaussian_id=0)#np.log(st.uniform(loc=np.array([100,100]),scale=np.array([400,400])).pdf(Z))
    return loglike,reward #+ logprior

#Returns loglikelihood p(X|Z)
def get_loglike(X):
    dispersion = np.var(X)
    return dispersion

def mcmc_mh_posterior_pusht(args, func, proposal_func, proposal_func_prob, n_iter=1000): #custom_posterior, gaussian_proposal, gaussian_proposal_prob,
    # Executes Metropolis-Hastings in a loop for n_iters
    Zs = []
    Z_x = args['Z_x']
    Z_y = args['Z_y']
    Z_curr = np.array([Z_x,Z_y])
    Zz_curr = model.inverse(torch.from_numpy(Z_curr.reshape(1,-1)).float()).detach().numpy()
    score,obs_list,reward = get_trajectory(Zz_curr[:,0][0],Zz_curr[:,1][0])
    
    X = np.array(obs_list)[:,2:4]
    prob_curr = get_loglike(X)

    accept_rates = []
    dispersion_list =[]
    rewards_list = []
    accept_cum = 0
    
    for i in range(1, n_iter+1):
        Z_new,accept,prob_new,reward_new = mcmc_mh(Z_curr,prob_curr,func, proposal_func, proposal_func_prob)
        accept_cum +=accept
        accept_rates.append(accept_cum/i)
        Zs.append(model.inverse(torch.from_numpy(Z_new.reshape(1,-1)).float()).detach().numpy().squeeze())

        if accept==1:
            Z_curr =  Z_new
            dispersion_list.append(prob_new)
            rewards_np = np.array(reward_new)
            rewards_list.append(rewards_np)
            prob_curr=prob_new
    return Zs, accept_rates, dispersion_list, rewards_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--Z_x", type=int)
    parser.add_argument("--Z_y", type=int)
    args = vars(parser.parse_args())
    with ipdb.launch_ipdb_on_exception():
        Zs, accept_rates, dispersion_list, rewards_list = mcmc_mh_posterior_pusht(args, custom_posterior, gaussian_proposal, gaussian_proposal_prob, n_iter=500)  
    data = {'Z':Zs,'disp':dispersion_list,'accept':accept_rates,'reward':rewards_list}
    file = open(f"/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/MH/MH_{args['Z_x']}_{args['Z_y']}.pkl", 'wb')
    
    # dump information to that file
    pickle.dump(data, file)
    file.close()