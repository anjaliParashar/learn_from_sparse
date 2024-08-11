import sys
sys.path.append('/home/anjali/learn_from_sparse')
import numpy as np
from pusht.verify_manipulation.utils.metropolis_hastings import gaussian_proposal,gaussian_proposal_prob,mcmc_mh
import scipy.stats as st
from pusht.verify_manipulation.utils.inference_final import get_trajectory,normalize_data
import argparse
import ipdb
import pickle

#Returns p(Z|X)
def custom_posterior(X, Z): 
    # Assumes gaussian log likelihood and gaussian prior
    #loglik = np.sum(np.log(st.norm(loc=Z, scale=1).pdf(X)))
    #logprior = np.log(st.norm(loc=0, scale=1).pdf(Z))
    loglike = get_loglike(X)
    #logprior = np.log(st.uniform(loc=np.array([100,100]),scale=np.array([400,400])).pdf(Z))
    return loglike #+ logprior

#Returns loglikelihood p(X|Z)
def get_loglike(X):
    dispersion = np.var(X)
    return dispersion

def mcmc_mh_posterior_pusht(args, func, proposal_func, proposal_func_prob, n_iter=1000): #gaussian_posterior, gaussian_proposal, gaussian_proposal_prob,
    # Executes Metropolis-Hastings in a loop for n_iters
    Zs = []
    Z_x = args['Z_x']
    Z_y = args['Z_y']
    Z_curr = np.array([Z_x,Z_y])
    score,obs_list,reward = get_trajectory(Z_curr[0],Z_curr[1])
    X = np.array(obs_list)[:,2:4]

    accept_rates = []
    dispersion_list =[]
    rewards_list = []
    accept_cum = 0
    
    for i in range(1, n_iter+1):
        Z_new,accept = mcmc_mh(X,Z_curr,func, proposal_func, proposal_func_prob)
        accept_cum +=accept
        accept_rates.append(accept_cum/i)
        Zs.append(Z_new)

        if accept==1:
            #run inference on the new initial seed
            _,obs_list,rewards = get_trajectory(Z_new[0],Z_new[1])
            rewards_np = np.array(rewards)
            X = np.array(obs_list)[:,2:4]
            Z_curr =  Z_new
            dispersion_list.append(get_loglike(X))
            rewards_list.append(rewards_np)
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