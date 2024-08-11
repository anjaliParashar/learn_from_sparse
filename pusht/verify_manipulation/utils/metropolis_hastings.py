#Python implementation of MH algorithm, adopted from:
# https://boyangzhao.github.io/posts/mcmc-bayesian-inference

import numpy as np
import scipy.stats as st
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az

#Returns p(Z|X)
def gaussian_posterior(X, Z): 
    # Assumes gaussian log likelihood and gaussian prior
    loglik = np.sum(np.log(st.norm(loc=Z, scale=1).pdf(X)))
    logprior = np.log(st.norm(loc=0, scale=1).pdf(Z))
    return loglik + logprior

#Returns Z^* given Z_t
def gaussian_proposal(Z_curr):
    # proposal based on Gaussian
    Z_new = st.multivariate_normal(mean=Z_curr, cov=100).rvs()
    if Z_new[0]<100 or Z_new[1]<100 or Z_new[0]>500 or Z_new[1]>500:
        Z_new = st.multivariate_normal(mean=Z_curr, cov=500).rvs()
    return Z_new

#Returns q(Z^*|Z_t). Assumes gaussian distribution for q(.)
def gaussian_proposal_prob(x1, x2):
    # calculate proposal probability q(x2|x1), based on Gaussian
    q = st.multivariate_normal(mean=x1, cov=500).pdf(x2)
    return q

#Implements one iteration of Metropolis-Hastings 
def mcmc_mh(X,Z_curr,func,proposal_func,proposal_func_prob): 
    Z_new = proposal_func(Z_curr) #Returns Z^* given Z_t
        
    prob_curr = func(X, Z_curr) #Calculate likelihood of Z_t
    prob_new = func(X, Z_new) #Calculate likelihood of Z^*
    
    # we calculate the prob=exp(x) only when prob<1 so the exp(x) will not overflow for large x
    if prob_new > prob_curr:
        acceptance_ratio = 1
    else:
        qr = proposal_func_prob(Z_curr, Z_new)/proposal_func_prob(Z_curr, Z_new)
        acceptance_ratio = np.exp(prob_new - prob_curr) * qr
    acceptance_prob = min(1, acceptance_ratio)
    
    if acceptance_prob > st.uniform(0,1).rvs():
        accept = 1
        Z_t_plus = Z_new
 
    else:
        Z_t_plus = Z_curr
        accept = 0
    return Z_t_plus, accept
    
        
def mcmc_mh_posterior_toy(X, Z_init, func, proposal_func, proposal_func_prob, n_iter=1000): #gaussian_posterior, gaussian_proposal, gaussian_proposal_prob,
    # Executes Metropolis-Hastings in a loop for n_iters
    Zs = []
    Z_curr = Z_init
    accept_rates = []
    accept_cum = 0
    
    for i in range(1, n_iter+1):
        Z_new,accept = mcmc_mh(X,Z_curr,func, proposal_func, proposal_func_prob)
        accept_cum +=accept
        accept_rates.append(accept_cum/i)
        Zs.append(Z_new)
        Z_curr =  Z_new
    return Zs, accept_rates

def plot_res(xs, burn_in, x_name):
    # plot trace (based on xs), distribution, and autocorrelation

    xs_kept = xs[burn_in:]
    
    # plot trace full
    fig, ax = plt.subplots(2,2, figsize=(15,5))
    ax[0,0].plot(xs)
    ax[0,0].set_title('Trace, full')
    
    # plot trace, after burn-in
    ax[0,1].plot(xs_kept)
    ax[0,1].set_title('Trace, after discarding burn-in')

    # plot distribution, after burn-in
    sns.histplot(xs_kept, ax=ax[1,0])
    ax[1,0].set_xlabel(f'{x_name} (after burn-in)')
    
    # plot autocorrelation, after burn-in
    plot_acf(np.array(xs_kept), lags=100, ax=ax[1,1], title='')
    ax[1,1].set_xlabel('Lag (after burn-in)')
    ax[1,1].set_ylabel('Autocorrelation')
    plt.show()


"""import pymc as pm
X = st.norm(loc=3, scale=1).rvs(size=1000)
with pm.Model() as model:

    prior = pm.Normal('mu', mu=0, sigma=1)  # prior
    obs = pm.Normal('obs', mu=prior, sigma=1, observed=X)  # likelihood
    step1 = pm.Metropolis()
    step2 = pm.hmc.NUTS()

    # sample with 3 independent Markov chains
    trace = pm.sample(draws=50000, chains=5, step=step2, return_inferencedata=True)  

pm.plot_trace(trace)
pm.plot_posterior(trace)
#x = az.extract(trace,group='posterior')"""
if __name__=='__main__':
    # run MH-MCMC
    X = st.norm(loc=3, scale=1).rvs(size=1000)
    Zs, accept_rates = mcmc_mh_posterior_toy(X, 1, 
                                            gaussian_posterior, gaussian_proposal, gaussian_proposal_prob, 
                                            n_iter=10000)
    plot_res(Zs, 500, 'theta')
    print(f"Mean acceptance rate: {np.mean(accept_rates[500:]): .3f}")
