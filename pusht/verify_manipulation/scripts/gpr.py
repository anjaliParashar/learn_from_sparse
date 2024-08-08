import gpytorch
import sys
sys.path.append('/home/anjali/learn_from_sparse')
import torch
from matplotlib import pyplot as plt
import numpy as np
import ipdb
import pickle

file = open('/home/anjalip/learn_from_sparse/pusht/verify_manipulation/data/gpr/score_estimate.pkl', 'rb')
# dump information to that file
data = pickle.load(file)
file.close()

score_estimate1 = data['score_estimate1']
score_true1 = data['score_true1']
seed1 = np.array(data['seed1'])

score_estimate2 = data['score_estimate2']
score_true2 = data['score_true2']
seed2 = np.array(data['seed2'])

plt.plot(score_estimate1,'o',label='Estimate')
plt.plot(score_true1,'o',label='True')
plt.legend()
plt.title('Difference in true and estimated risk for Cluster-1')
plt.show()

plt.figure()
plt.plot(score_estimate2,'o',label='Estimate')
plt.plot(score_true2,'o',label='True')
plt.legend()
plt.title('Difference in true and estimated risk for Cluster-2')
plt.show()

seed1 = (seed1-100)/400
seed2=(seed2-100)/400
# Plot training data as black stars
#plt.plot(seed1[], score_true, 'k*',label='True')
#plt.plot(seed1, score_estimate, 'r^',label='Predicted')
#plt.legend()
#plt.show()

class SpectralMixtureGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(SpectralMixtureGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean() # Construct the mean function
        #self.mean = gpytorch.means.LinearMeanGrad(2)# Construct the mean function
        #self.cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.cov = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=100,ard_num_dims=2) # Construct the kernel function
        self.cov.initialize_from_data(x_train, y_train) # Initialize the hyperparameters from data
        
    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x) 
        
# Initialize the likelihood and model
#x_train,y_train = torch.tensor(seed1[0:60]).float(), torch.tensor(score_true1[0:60]).float()
x_train,y_train = torch.vstack((torch.tensor(seed1[0:10]),torch.tensor(seed2[0:15]))).float(), torch.tensor(np.hstack((score_true1[0:10],score_true2[0:15]))).float()
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = SpectralMixtureGP(x_train, y_train,likelihood)


# Put the model into training mode
model.train()
likelihood.train()

# Use the Adam optimizer, with learning rate set to 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Use the negative marginal log-likelihood as the loss function
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# Set the number of training iterations
n_iter = 500

for i in range(n_iter):
    # Set the gradients from previous iteration to zero
    optimizer.zero_grad()
    # Output from model
    output = model(x_train)
    # Compute loss and backprop gradients
    loss = -mll(output, y_train)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
    optimizer.step()

# The test data is 50 equally-spaced points from [0,5]
x_test = torch.vstack((torch.tensor(seed1[0:10]),torch.tensor(seed2[0:20]))).float()
#x_test = torch.tensor(seed1[20:60]).float()
score_ = np.hstack((score_true1[0:10],score_true2[0:20]))
#score_ = torch.tensor(score_true1[20:60]).double()
# Put the model into evaluation mode
model.eval()
likelihood.eval()

# The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
# See https://arxiv.org/abs/1803.06058
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Obtain the predictive mean and covariance matrix
    X = np.linspace(100,500,10)
    Y = np.linspace(100,500,10)
    mean_preds = torch.zeros((10,10))
    i=0
    for x in X:
        j=0
        for y in Y:
            x = (x-100)/400
            y = (y-100)/400
            mean_preds[i,j] = likelihood(model(torch.tensor([x,y]).float().reshape((1,2)))).mean
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
    plt.contourf(X,Y,mean_preds,levels=10)
    plt.colorbar()
    plt.title('Predicted risk',fontsize=20)
    plt.savefig('GPR_pred.png')

    plt.figure()
    input_ = np.linspace(0,x_test.shape[0],x_test.shape[0])
    observed_pred = likelihood(model(x_test))
    lower, upper = observed_pred.confidence_region()
    plt.plot(input_,observed_pred.mean.detach().numpy(),label='GP Mean')
    plt.plot(input_,score_,'o',color='red',label='Observed Data')
    plt.fill_between(input_,lower.numpy(), upper.numpy(), alpha=0.5,label='Confidence')
    plt.legend()
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
