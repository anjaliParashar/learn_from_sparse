import sys
sys.path.append('/home/anjali/learn_from_sparse')
import numpy as np
from f1tenth.utils.metropolis_hastings import gaussian_proposal,gaussian_proposal_prob,mcmc_mh,get_means
import scipy.stats as st
from f1tenth.utils.simulation import do_simulation
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
PATH = "/home/anjali/learn_from_sparse/flowgmm/experiments/synthetic_data/f1tenth_noise.pt"
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
x_lims = np.array([0, 7])
line_z = np.linspace(*z_lims, grid_points)
line_x = np.linspace(*x_lims, grid_points)
xx_z, yy_z = np.meshgrid(line_z, line_z)
xx_x, yy_x = np.meshgrid(line_x, line_x)

#Returns p(Z|X) using custom defined likelihood and custom prior p(Z)
def custom_posterior(Z): 
    # Assumes gaussian log likelihood and gaussian prior
    #loglik = np.sum(np.log(st.norm(loc=Z, scale=1).pdf(X)))
    #logprior = np.log(st.norm(loc=0, scale=1).pdf(Z))
    Zz = model.inverse(torch.from_numpy(Z.reshape(1,-1)).float()).detach().numpy()
    #yaw,risk,_,_,_,_ = do_simulation(Zz[:,1][0],Zz[:,0][0]) 
    yaw, risk, mean_risk, max_risk, final_risk, _,_,_,_  = do_simulation(Zz[:,1][0],Zz[:,0][0]) 
    loglike = get_loglike(yaw)
    #logprior = loss_fn.prior.log_prob_(torch.tensor(z),gaussian_id=0)#np.log(st.uniform(loc=np.array([100,100]),scale=np.array([400,400])).pdf(Z))
    return loglike,risk,mean_risk, max_risk, final_risk #+ logprior

#Returns loglikelihood p(X|Z)
def get_loglike(X):
    dispersion = np.max(X) - np.min(X)#np.var(X)
    return dispersion

def mcmc_mh_posterior_f1tenth(args, func, proposal_func, proposal_func_prob, n_iter=1000): #custom_posterior, gaussian_proposal, gaussian_proposal_prob,
    # Executes Metropolis-Hastings in a loop for n_iters
    Zs = []
    Z_x = args['Z_x']
    Z_y = args['Z_y']
    Z_curr = np.array([Z_x,Z_y])
    Zz_curr = model.inverse(torch.from_numpy(Z_curr.reshape(1,-1)).float()).detach().numpy()
    yaw, risk, mean_risk, max_risk, final_risk, _,_,_,_ = do_simulation(Zz_curr[:,1][0],Zz_curr[:,0][0])
    #yaw,risk,_,_,_,_ = do_simulation(Zz_curr[:,1][0],Zz_curr[:,0][0])
    prob_curr = get_loglike(yaw)

    accept_rates = []
    dispersion_list =[]
    risk_list = []
    mean_list = []
    max_list = []
    final_list = []
    accept_cum = 0
    
    for i in range(1, n_iter+1):
        Z_new,accept,prob_new,risk_new, mean_risk,max_risk,final_risk = mcmc_mh(Z_curr,prob_curr,func, proposal_func, proposal_func_prob)
        accept_cum +=accept
        accept_rates.append(accept_cum/i)
        Zz_new = model.inverse(torch.from_numpy(Z_new.reshape(1,-1)).float()).detach().numpy().squeeze()
        Zs.append(Zz_new)
        print("Z_new:",Z_new, "Zz_new:",Zz_new,'prob:',prob_new)

        if accept==1:
            Z_curr =  Z_new
            dispersion_list.append(prob_new)
            risk_np = np.array(risk_new)
            risk_list.append(risk_np)
            mean_np = np.array(mean_risk)
            mean_list.append(risk_np)
            max_np = np.array(max_risk)
            max_list.append(risk_np)
            final_np = np.array(final_risk)
            final_list.append(risk_np)
            prob_curr=prob_new
    return Zs, accept_rates, dispersion_list, risk_list, mean_list,max_list,final_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--Z_x", type=float)
    parser.add_argument("--Z_y", type=float)
    args = vars(parser.parse_args())
    with ipdb.launch_ipdb_on_exception():
        Zs, accept_rates, dispersion_list, risk_list,mean_list,max_list,final_list = mcmc_mh_posterior_f1tenth(args, custom_posterior, gaussian_proposal, gaussian_proposal_prob, n_iter=500)  
    data = {'Z':Zs,'disp':dispersion_list,'accept':accept_rates,'risk':risk_list,'mean':mean_list,'max':max_list,'final':final_list}
    file = open(f"/home/anjali/learn_from_sparse/f1tenth/data/MH_circle_proj/MH_{args['Z_x']}_{args['Z_y']}.pkl", 'wb')
    
    # dump information to that file
    pickle.dump(data, file)
    file.close()