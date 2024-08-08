import gpytorch
import sys
sys.path.append('/home/anjali/learn_from_sparse')
import torch
from matplotlib import pyplot as plt
import numpy as np
import ipdb
import pickle

device=torch.device('cuda')

#Open all clustered datasets created using "gpr_dataset_gen.py"
file = open('/home/anjalip/learn_from_sparse/pusht/verify_manipulation/data/gpr/score_clusters.pkl', 'rb')
data = pickle.load(file)
file.close()

#Unpack data from pkl. 
score_estimate1 = data['score_estimate1']
score_true1 = data['score_true1']
seed1 = np.array(data['seed1'])
seed_torch = torch.tensor(seed1)

score_estimate2 = data['score_estimate2']
score_true2 = data['score_true2']
seed2 = np.array(data['seed2'])

score_estimate3 = data['score_estimate3']
score_true3 = data['score_true3']
seed3 = np.array(data['seed3'])

"""score_estimate4 = data['score_estimate4']
score_true4 = data['score_true4']
seed4 = np.array(data['seed4'])"""

fig, axs = plt.subplots(1,3,figsize=(12,4))

#fig.suptitle('Difference in true and estimated risk')
axs[0].plot(score_estimate1,'o',label='Estimate')
axs[0].plot(score_true1,'o',label='True')
axs[0].set_title('Cluster-1')

axs[1].plot(score_estimate2,'o',label='Estimate')
axs[1].plot(score_true2,'o',label='True')
axs[1].set_title('Cluster-2')

axs[2].plot(score_estimate3,'o',label='Estimate')
axs[2].plot(score_true3,'o',label='True')
axs[2].set_title('Cluster-3 ')
axs[2].legend()
"""axs[1,1].plot(score_estimate4,'o',label='Estimate')
axs[1,1].plot(score_true4,'o',label='True')
axs[1,1].legend()
axs[1,1].set_title('Cluster-4 (HR,LV)')"""
plt.savefig('Clusters.png')

#Normalize seeds to be within [0,1]. Not sure if it makes a difference?
seed1 = (seed1-100)/400
seed2=(seed2-100)/400
seed3 = (seed3-100)/400
#seed4=(seed4-100)/400

def distance(model_list,likelihood_list,seed,y_train):
    error_list = []
    for model,likelihood in zip(model_list,likelihood_list):
        model.eval()
        likelihood.eval()
        observed_pred = likelihood(model(seed.reshape((1,2))))
        error_list.append(torch.linalg.norm(observed_pred.mean-y_train))
    return min(error_list),error_list.index(min(error_list))

def train(model,likelihood,xx_train,yy_train):
    # Use the negative marginal log-likelihood as the loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model.to(device))

    # Put the model into training mode
    model.train()
    likelihood.train()

    # Set the number of training iterations
    n_iter = 100
    for i in range(n_iter):
        # Set the gradients from previous iteration to zero
        optimizer.zero_grad()

        # Output from model
        output = model(xx_train.to(device))

        # Compute loss and backprop gradients
        loss = -mll(output, yy_train.to(device))
        
        loss.backward()

        print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))

        optimizer.step()
    return model,likelihood

#This piece was adopted from GPytorch tutorial
# https://docs.gpytorch.ai/en/v1.6.0/examples/01_Exact_GPs/Simple_GP_Regression.html
class SpectralMixtureGP(gpytorch.models.ExactGP):
    def __init__(self, x_train,y_train,likelihood):
        super(SpectralMixtureGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean().to(device) # Construct the mean function
        #self.cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.cov = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=100,ard_num_dims=2).to(device) # Construct the kernel function
        self.cov.initialize_from_data(x_train, y_train) # Initialize the hyperparameters from data
        
    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x) 

model_list = []
likelihood_list = []
#Train
for i in range(25):        
    #Initialize a model using a very small chunk of data
    x_train,y_train = torch.tensor(seed1[i*25:i*25+25]).to(device), torch.tensor(score_estimate1[i*25:i*25+25]).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = SpectralMixtureGP(x_train, y_train,likelihood)
    model_list.append(model)
    likelihood_list.append(likelihood)

    # Use the Adam optimizer, with learning rate set to 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    #Train an ensemble of initial models
    train(model,likelihood,x_train,y_train)
    print(i)

#Eval
fig, axs = plt.subplots(5,5)
i=0
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for k in range(5):   
        for j in range(5):
            # Obtain the predictive mean and covariance matrix
            model = model_list[i]     
            likelihood = likelihood_list[i]
            model.eval()
            likelihood.eval()
            observed_pred = likelihood(model(torch.tensor(seed1[i*25:i*25+25]).float().to(device)))
            lower, upper = observed_pred.confidence_region()
            axs[k,j].plot(range(25),score_true1[i*25:i*25+25],'o',color='red',label='True Risk')
            axs[k,j].plot(range(25),score_estimate1[i*25:i*25+25],'o',color='black',label='Observed Risk')
            axs[k,j].plot(range(25),observed_pred.mean.cpu().numpy(),label='GP Mean')
            axs[k,j].fill_between(range(25),lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5,label='Confidence')
            i+=1
plt.show()

#Retrain this model:
i=7
xnew_train = torch.tensor(seed1[i*25:i*25+25]).to(device)
ynew_train = torch.hstack((torch.tensor(score_estimate1[i*25:i*25+10]),torch.tensor(score_true1[i*25+10]),torch.tensor(score_estimate1[i*25+11:i*25+20]),torch.tensor(score_true1[i*25+20]),torch.tensor(score_estimate1[i*25+21:i*25+25]))).to(device)

likelihood_update = gpytorch.likelihoods.GaussianLikelihood().to(device)
model_update = SpectralMixtureGP(xnew_train, ynew_train,likelihood)
train(model_update,likelihood_update,xnew_train,ynew_train)

model_update.eval()
likelihood_update.eval()
observed_pred_update = likelihood_update(model_update(torch.tensor(seed1[i*25:i*25+25]).float().to(device)))
lower_update, upper_update = observed_pred_update.confidence_region()

fig, axs = plt.subplots(1,2)
axs[0].plot(range(25),score_true1[i*25:i*25+25],'o',color='red',label='True Risk')
axs[0].plot(range(25),score_estimate1[i*25:i*25+25],'o',color='black',label='Observed Risk')
axs[0].plot(range(25),observed_pred_update.mean.detach().cpu().numpy(),label='GP Mean Updated')
axs[0].fill_between(range(25),lower_update.detach().cpu().numpy(), upper_update.detach().cpu().numpy(), alpha=0.5,label='Confidence')
axs[0].set_title('Updated Prediction')
axs[0].legend(fontsize=15)


model = model_list[i]
likelihood = likelihood_list[i]
model.eval()
likelihood.eval()
observed_pred = likelihood(model(torch.tensor(seed1[i*25:i*25+25]).float().to(device)))
lower, upper = observed_pred.confidence_region()
axs[1].plot(range(25),score_true1[i*25:i*25+25],'o',color='red',label='True Risk')
axs[1].plot(range(25),score_estimate1[i*25:i*25+25],'o',color='black',label='Observed Risk')
axs[1].plot(range(25),observed_pred.mean.detach().cpu().numpy(),label='GP Mean True')
axs[1].fill_between(range(25),lower.detach().cpu().numpy(), upper.detach().cpu().numpy(), alpha=0.5,label='Confidence')
axs[1].set_title('Old Prediction')
axs[1].legend(fontsize=15)
plt.show()
"""model.eval()
likelihood.eval()
observed_pred = likelihood(model(x_train))
print(observed_pred.variance)
breakpoint()"""

"""
k_idx=0
model_current = model
likelihood_current = likelihood
dist_threshold = 0.2

model_list = [model_current]
likelihood_list = [likelihood_current]
for xx_train,yy_train in zip(x_train_plus,y_train_plus):
    dist,idx = distance(model_list,likelihood_list,xx_train,yy_train)
    if dist>dist_threshold:
        #Initialize a new model
        model_current = SpectralMixtureGP(xx_train, yy_train,likelihood)
        likelihood_current = gpytorch.likelihoods.GaussianLikelihood()
        likelihood_list.append(likelihood_current)
        model_list.append(model_current)
    else:
        model_current = model_list[idx]
        likelihood_current = likelihood_list[idx]
    
    # Put the model into training mode
    model_current.train()
    likelihood.train()
    train(model_current,likelihood_current,xx_train,yy_train)
        
breakpoint()   


model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Obtain the predictive mean and covariance matrix
    observed_pred1 = likelihood(model(torch.tensor(seed1).float().to(device)))
    lower1, upper1 = observed_pred1.confidence_region()

    observed_pred2 = likelihood(model(torch.tensor(seed2).float().to(device)))
    lower2, upper2 = observed_pred2.confidence_region()

fig, axs = plt.subplots(2,1)
axs[0].plot(range(int(score_true1.shape[0])),score_true1,'o',color='red',label='True Risk')
axs[0].plot(range(int(score_true1.shape[0])),score_estimate1,'o',color='black',label='Observed Risk')
axs[0].plot(range(int(score_true1.shape[0])),observed_pred1.mean.cpu().numpy(),label='GP Mean')
axs[0].fill_between(range(int(score_true1.shape[0])),lower1.cpu().numpy(), upper1.cpu().numpy(), alpha=0.5,label='Confidence')
axs[0].set_title('Cluster-1 (HR,HV)')
axs[0].legend()

axs[1].plot(range(int(score_true2.shape[0])),score_true2,'o',color='red')
axs[1].plot(range(int(score_true2.shape[0])),score_estimate2,'o',color='black',label='Observed Risk')
axs[1].plot(range(int(score_true2.shape[0])),observed_pred2.mean.cpu().numpy())
axs[1].fill_between(range(int(score_true2.shape[0])),lower2.cpu().numpy(), upper2.cpu().numpy(), alpha=0.5,)
axs[1].set_title('Cluster-2 (LR,HV)')

plt.show()
plt.savefig('Clusters_primitive_num_iters_500.png')
"""