import gpytorch
import sys
sys.path.append('/home/anjali/learn_from_sparse')
import torch
from matplotlib import pyplot as plt
import numpy as np
import ipdb
import pickle

device = torch.device('cuda')
#Collect experimental data
obs_np = np.zeros((1,2))
reward_np = np.zeros((1,1))
for i in range(10):
    file = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/gpr/experiments/'+str(i)+'.pkl','rb')
    data = pickle.load(file)
    file.close()
    obs_np = np.vstack((obs_np,data['obs'][:,2:4]))
    score = np.repeat(max(data['reward']),data['obs'].shape[0]).reshape((-1,1)) #+ 0.3*np.random.rand()
    reward_np = np.vstack((reward_np,score))
reward_np = 1-reward_np #reward to risk

#data_ = torch.tensor(np.hstack((obs_np,reward_np)))
#data_=data_[torch.randperm(data_.size()[0])]

#Collect data from simulations
file_sim = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/gpr/sim/pusht_gpr_train.pkl','rb')
data_sim = pickle.load(file_sim)
file_sim.close()

#Combine datasets from both sources
obs_full = np.vstack((np.array(data_sim['seed']),obs_np))
reward_full = np.vstack((np.array(data_sim['risk']).reshape((-1,1)),reward_np))
obs_full = obs_full/500
print('sim:',len(np.array(data_sim['seed'])))
print("obs_full:",obs_full.shape)
print("reward_full:",reward_full.shape)
plt.scatter(obs_full[:,0],obs_full[:,1])
plt.title('Sim+Experiment Data')
plt.show()

class SpectralMixtureGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(SpectralMixtureGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean() # Construct the mean function
        #self.mean = gpytorch.means.LinearMeanGrad(2)# Construct the mean function
        self.cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.cov = gpytorch.kernels.ProductStructureKernel(gpytorch.kernels.RBFKernel(),num_dims=2)
        #self.cov = gpytorch.kernels.RBFKernel()
        #self.cov = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=100,ard_num_dims=2).to(device) # Construct the kernel function
        #self.cov.initialize_from_data(x_train, y_train) # Initialize the hyperparameters from data
        
    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x) 
        
# Initialize the likelihood and model
#x_train,y_train = data_[:,2:4].to(device),data_[:,-1].to(device)
x_train,y_train = torch.tensor(obs_full).float().to(device), torch.tensor(reward_full).float().squeeze().to(device)
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = SpectralMixtureGP(x_train, y_train,likelihood).to(device)


# Put the model into training mode
model.train()
likelihood.train()

# Use the Adam optimizer, with learning rate set to 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Use the negative marginal log-likelihood as the loss function
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)

# Set the number of training iterations
n_iter = 300
weight=0.3
N_Sim = len(np.array(data_sim['seed']))
for i in range(n_iter):
    # Set the gradients from previous iteration to zero
    optimizer.zero_grad()
    # Output from model
    output = model(x_train)
    # Compute loss and backprop gradients
    loss_sim = -mll(output[0:N_Sim], y_train[0:N_Sim])
    loss_exp = -mll(output[N_Sim:], y_train[N_Sim:])
    loss = (weight)*loss_sim + (1-weight)*loss_exp
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    optimizer.step()

x_test = x_train
y_test = y_train
model.eval()
likelihood.eval()

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Obtain the predictive mean and covariance matrix
    X = np.linspace(0,1,30)
    Y = np.linspace(0,1,30)
    mean_preds = torch.zeros((30,30))
    i=0
    for x in X:
        j=0
        for y in Y:
            #x = (x)/500
            #y = (y)/500
            mean_preds[i,j] = torch.sigmoid(likelihood(model(torch.tensor([x,y]).float().reshape((1,2)).to(device))).mean)
            j+=1
        i+=1
    
    #f_preds = model(x_test)
    #f_mean = f_preds.mean
    #f_cov = f_preds.covariance_matrix
    
    # Make predictions by feeding model through likelihood
    #input_ = np.linspace(0,100,100)
    #observed_pred = likelihood(model(x_test))
    #lower, upper = observed_pred.confidence_region()
    #breakpoint()
    #plt.figure()
    #plt.plot(input_,score_,'o',color='red',label='Observed Data')
    #plt.plot(input_,observed_pred.mean.numpy(),label='GP Mean')
    #plt.fill_between(input_,lower.numpy(), upper.numpy(), alpha=0.5,label='Confidence')
    #plt.ylim([-3, 3])
    #plt.legend()
    #plt.savefig('GPR_predict.png')

    plt.figure()
    plt.contourf(X*500,Y*500,mean_preds.detach().cpu(),levels=10)
    plt.colorbar()
    plt.title('Predicted risk, Weight=0.3',fontsize=20)
    plt.show()
    #plt.savefig('GPR_pred.png')

    plt.figure()
    input_ = np.linspace(0,x_test.shape[0],x_test.shape[0])
    observed_pred = likelihood(model(x_test))
    lower, upper = observed_pred.confidence_region()
    plt.plot(input_,observed_pred.mean.detach().cpu(),label='GP Mean')
    plt.plot(input_,reward_full,'o',color='red',label='Observed Data')
    plt.fill_between(input_,lower.cpu(), upper.cpu(), alpha=0.5,label='Confidence')
    plt.legend()
    plt.show()
    """
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x_train.numpy(), y_train.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(x_test.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(x_test.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    """
plt.show()
