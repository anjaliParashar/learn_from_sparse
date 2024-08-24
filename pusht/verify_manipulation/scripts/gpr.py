#Script for training a Gaussian Process regression for learning Risk from Sim+Experimental data. 
# GPR framework adopted from https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
import gpytorch
import sys
sys.path.append('/home/anjali/learn_from_sparse')
import torch
from matplotlib import pyplot as plt
import numpy as np
import ipdb
import pickle
from pusht.verify_manipulation.scripts.coverage_analyze import generate_exp_data
device = torch.device('cuda')

#Define the GP Mean and covariance. Compositional kernel of RBF + Scale gives good results. 
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

def generate_exp_data(N):
    #Collect experimental data from "N" epxeriments
    obs_np = np.zeros((1,2))
    Z_initial=np.zeros((1,2))
    reward_np = np.zeros((1,1))
    for i in range(N):
        file = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/gpr/experiments/'+str(i)+'.pkl','rb')
        data = pickle.load(file)
        file.close()
        obs_np = np.vstack((obs_np,data['obs'][:,2:4])) #obs[:,2:4] corresponds to [x_T,y_T]
        Z_initial = np.vstack((obs_np,data['obs'][0,2:4])) #obs[:,2:4] corresponds to initial seeds for each experiment
        score = np.repeat(max(data['reward']),data['obs'].shape[0]).reshape((-1,1)) #+ 0.3*np.random.rand()
        reward_np = np.vstack((reward_np,score))   
    reward_np = 1-reward_np #reward to risk
    Z_initial = Z_initial[1:,:]
    reward_np =  reward_np[1:,:]
    obs_np =  obs_np[1:,:]
    return obs_np,reward_np,Z_initial

#Function for organizing data from various sources for initial training
def data_collect(N,visualize=True):

    #Collect experimental data from "N" epxeriments
    obs_np,reward_np,_ = generate_exp_data(N)

    #Collect data from simulations: Consists of algorithmic+noise failure regions that fail with confidence
    file_sim = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/gpr/sim/pusht_gpr_train.pkl','rb')
    data_sim = pickle.load(file_sim)
    file_sim.close()

    #Combine datasets from both sources
    N_Sim=134
    obs_full = np.vstack((np.array(data_sim['seed'][0:N_Sim]),obs_np))
    reward_full = np.vstack((np.array(data_sim['risk'][0:N_Sim]).reshape((-1,1)),reward_np))

    #Tried randomizing data for generalization, didn't work, idk why that's a problem with Gaussian Processes
    #data_ = torch.tensor(np.hstack((obs_np,reward_np)))
    #data_=data_[torch.randperm(data_.size()[0])] 

    #Normalize data: gives good generalization
    obs_full = obs_full/500

    #Check if sizes make sense
    print('sim:',len(np.array(data_sim['seed'])))
    print("obs_full:",obs_full.shape)
    print("reward_full:",reward_full.shape)

    #Visualize the sim+experiment data:Optional
    if visualize==True:
        plt.scatter(obs_full[:,0],obs_full[:,1])
        plt.title('Sim+Experiment Data')
        plt.show()
    return torch.tensor(obs_full).float().to(device), torch.tensor(reward_full).float().squeeze().to(device),N_Sim

#Function for training the GPR with collected data
def train(obs,reward,N_Sim,weight):
    # Initialize the likelihood and model
    x_train,y_train = obs,reward#torch.tensor(obs_full).float().to(device), torch.tensor(reward_full).float().squeeze().to(device)
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
    n_iter = 200

    #Choose some initial weight for mixing sim and exp data
    weight=0.3
    
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

    return  model,likelihood, mll

def weight_update(weight,model,mll,x_test,y_test,N_Sim):
    model.eval()

    output = model(x_test)
    loss_sim = -mll(output[0:N_Sim], y_test[0:N_Sim])
    loss_exp = -mll(output[N_Sim:], y_test[N_Sim:])
    
    lr =torch.tensor(0.01)
    n_iter=10

    loss = lambda weight: (weight)*loss_sim + (1-weight)*loss_exp
    weight =torch.tensor(weight).requires_grad_()
    
    for i in range(n_iter):
        weight_grad = torch.autograd.grad(loss,weight)
        weight += -weight_grad*lr 
    return weight
        
#For a given set of data-points, we check the prediction error and uncertainity 
def eval(x_test,y_test,model,likelihood):
    model.eval()
    likelihood.eval()
    observed_pred = likelihood(model(x_test))
    lower, upper = observed_pred.confidence_region()
    variance = upper.numpy()-lower.numpy()
    y_pred = observed_pred.mean.numpy()
    error_sq = np.power((y_pred-y_test),2) 
    score = error_sq + variance
    return score

#Visualize the results of a trained GPR
def visualize_gpr(x_test,y_test,model,likelihood):
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
    plt.plot(input_,y_test,'o',color='red',label='Observed Data')
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

if __name__=="__main__":

    N_final=20
    N=N_final/2
    N_Sim=134
    weight=0.5
    obs_init,reward_init,Z_init = generate_exp_data(N)
    for i in range(N_final-N):
        #Zip data from sim+exp for training GPR
        obs,reward,N_sim= data_collect(N,visualize=True)

        #Train GPR using collected data
        model,likelihood,mll = train(obs,reward,N_Sim,weight)
        
        #Evaluate performance on the initially chosen set of points. This tells you where to sample from and how to update the weight
        score = eval(Z_init,reward_init,model,likelihood)
        weight = weight_update(weight,model,mll,obs,reward,N_Sim) #Interweaved optimization of the weight

        #Get index corresponding to the largest score in Z_init, sample from the corresponding cluster
        search_idx = np.argmax(score)
        Z_new = get_Z_new(search_idx)

        # Remove corresponding element from Z_init
        Z_init= np.delete(Z_init,search_idx,axis=0) #Z_init shape needs to be NX 2 for this, else axis=1

        #This step should happen in experiment, with updating the data corresponding to Z_new
        N+=1































# Sample from the region
import sys
sys.path.append('/home/anjali/learn_from_sparse')
import numpy as np
from pusht.verify_manipulation.utils.metropolis_hastings import gaussian_proposal,gaussian_proposal_prob,mcmc_mh,get_means
from flowgmm.flow_ssl.realnvp.realnvp import RealNVPTabular
from flowgmm.flow_ssl.distributions import SSLGaussMixture
from flowgmm.flow_ssl import FlowLoss

#import data from Normalizing flows to prepare prior
flow = RealNVPTabular(num_coupling_layers=20, in_dim=2, num_layers=1, hidden_dim=32)
PATH = "/home/anjali/learn_from_sparse/flowgmm/experiments/synthetic_data/pusht_noise.pt"
checkpoint = torch.load(PATH)
flow.load_state_dict(checkpoint['model_state_dict'])
flow.eval()

r = 2
n_classes = 2
means = get_means(n_classes,r)
prior = SSLGaussMixture(means=means)
loss_fn = FlowLoss(prior)

zs = []
for i in range(len(means)):
    z = loss_fn.prior.sample((1000,), gaussian_id=i).numpy()
    zs.append(z)
#   plt.scatter(z[:, 0], z[:, 1], cmap=plt.cm.rainbow)
#
#Gaussian sampling
for z in zs:
    x = flow.inverse(torch.from_numpy(z).float()).detach().numpy()
    print(x)