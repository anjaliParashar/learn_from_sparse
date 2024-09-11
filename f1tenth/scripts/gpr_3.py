#Script for training a Gaussian Process regression for learning Risk from Sim+Experimental data. 
# GPR framework adopted from https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html
import gpytorch
import sys
sys.path.append('/home/anjali/learn_from_sparse')
import torch
from matplotlib import pyplot as plt
import numpy as np
import ipdb
import scipy
import pickle
import ipdb
from flowgmm.flow_ssl.realnvp.realnvp import RealNVPTabular
from flowgmm.flow_ssl.distributions import SSLGaussMixture
from flowgmm.flow_ssl import FlowLoss
from pusht.verify_manipulation.utils.gpr_utils import get_Z_new
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from f1tenth.utils import cubic_spline
from f1tenth.utils.simulation import do_simulation
from f1tenth.utils.ref_curve import lane_change_trajectory
import matplotlib 
device = torch.device('cuda')
seed = 100
torch.manual_seed(seed)
#import data from Normalizing flows to prepare weights
flow_model = RealNVPTabular(num_coupling_layers=20, in_dim=2, num_layers=1, hidden_dim=32)
PATH = "/home/anjali/learn_from_sparse/flowgmm/experiments/synthetic_data/pusht_noise.pt"
checkpoint = torch.load(PATH)
flow_model.load_state_dict(checkpoint['model_state_dict'])
flow_model.eval()

def get_means(n_classes, r):
    phis = np.linspace(0, 2 * np.pi, n_classes+1)[:-1]
    mean_x = np.cos(phis) * r
    mean_y = np.sin(phis) * r
    means = np.hstack([mean_x[:, None], mean_y[:, None]])
    means = torch.from_numpy(means).float()
    return means


def grid_image(mapping, xx, yy, extradim=False):
    lines = np.hstack([xx.reshape([-1, 1]), yy.reshape([-1, 1])])
    if extradim:
        lines = lines[:, None, :]
    lines = torch.from_numpy(lines).float()
    img_lines = mapping(lines).detach().numpy()
    
    if extradim:
        img_xx, img_yy = img_lines[:, 0, 0], img_lines[:, 0, 1]
    else:
        img_xx, img_yy = img_lines[:, 0], img_lines[:, 1]
    img_xx = img_xx.reshape(xx.shape)
    img_yy = img_yy.reshape(yy.shape)
    return img_xx, img_yy

def get_decision_boundary(f_xx, f_yy, prior):
    f_points = np.hstack([f_xx.reshape([-1, 1]), f_yy.reshape([-1, 1])])
    classes = prior.classify(torch.from_numpy(f_points).float()).detach().numpy()
    return classes

def visualize_boundary():
    grid_points = 150
    x_lims = np.array([0, 600])
    line_x = np.linspace(*x_lims, grid_points)
    xx_x, yy_x = np.meshgrid(line_x, line_x)
    f_xx, f_yy = grid_image(flow_model, xx_x, yy_x)
    r = 2
    n_classes = 2
    means = get_means(n_classes,r)
    prior = SSLGaussMixture(means=means)
    classes = get_decision_boundary(f_xx, f_yy, prior)
    plt.contourf(xx_x, yy_x, classes.reshape(xx_x.shape), cmap="RdBu_r", alpha=0.5)
    plt.show()

def get_weight(Z):
    r = 2
    n_classes = 2
    means = get_means(n_classes,r)
    prior = SSLGaussMixture(means=means)
   
    img_Z = flow_model(Z.cpu()).detach()
    #failure region is label=1, the region we wish to get from is label=0
    weight = prior.classify(img_Z.float()) #.detach().numpy()
    return weight

#Define the GP Mean and covariance. Compositional kernel of RBF + Scale gives good results. 
class SpectralMixtureGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(SpectralMixtureGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean() # Construct the mean function
        #self.mean = gpytorch.means.LinearMeanGrad(2)# Construct the mean function
        self.cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.cov = gpytorch.kernels.ProductStructureKernel(gpytorch.kernels.RBFKernel(),num_dims=2)
        #self.cov = gpytorch.kernels.RBFKernel()
        #self.cov = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=20,ard_num_dims=2).to(device) # Construct the kernel function
        #self.cov.initialize_from_data(x_train, y_train) # Initialize the hyperparameters from data
        
    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x) 

def find_closest(x,y,y_ckpt):
    dist_actual = np.abs(y-y_ckpt)
    y_actual = np.argmin(dist_actual)
    
    if np.linalg.norm(y[y_actual]-y_ckpt)>0.01 and y_actual+1<len(y):
        y_ref = (y[y_actual]+y[y_actual+1])/2
        x_ref = (x[y_actual]+x[y_actual+1])/2
    else:
        y_ref= y[y_actual]
        x_ref = x[y_actual]    
    return x_ref,y_ref

def mean_dist_(x,y,cx,cy):
    y_ckpt = np.linspace(0,5,10)
    mean_dist = np.zeros((10,))
    for i,y_ in zip(range(10),y_ckpt):
        x_a,y_a = find_closest(x,y,y_)
        x_r,y_r = find_closest(cx,cy,y_)
        mean_dist[i]= np.linalg.norm(np.array([x_a-x_r,y_a-y_r]))
    return np.mean(mean_dist)

def risk(x,y,cx,cy):
    X = np.array([x,y]).squeeze()
    CX = np.array([cx,cy])#[:,0:X.shape[1]]
    dist_mat = scipy.spatial.distance.cdist(X.T,CX.T)
    min_dist = np.min(dist_mat,axis=1)
    mean_dist = mean_dist_(x,y,cx,cy)
    #mean_dist = min_dist.mean()
    max_dist = min_dist.max()
    final = np.min(np.linalg.norm(X-CX[:,-1][:,None],axis=0))
    risk_ = 20*mean_dist + 15*max_dist+10*final
    print("Mean:",mean_dist,"Max:",max_dist,"Final",final)
    return risk_,mean_dist,max_dist,final#np.linalg.norm(X-CX)

def generate_exp_data(N,visualize=False):
    #Collect experimental data from "N" epxeriments
    Z_np = np.zeros((1,2))
    risk_np = np.zeros((1,1))
    mean_np = np.zeros((1,1))
    max_np = np.zeros((1,1))
    final_np = np.zeros((1,1))
    for i in range(N):
        file_demo = open('/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/initial_'+str(i)+'.pkl','rb')
        data_demo = pickle.load(file_demo)
        file_exp = open('/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/initial_experiments/data_initial_'+str(i)+'.pkl','rb')
        data_exp = pickle.load(file_exp)
        file_exp.close()
        file_demo.close()
        risk_exp,mean_exp,max_exp,final_exp = risk(np.array(data_exp['states']['x'])+1.44,np.array(data_exp['states']['y'])+3.7,data_demo['X'],data_demo['Y'])
        risk_sim,_,_,_ = risk(data_demo['Y_sim'],data_demo['X_sim'],data_demo['X'],data_demo['Y'])
        print('Risk_exp:',risk_exp,'Risk_sim:',risk_sim,'ID:',i)
        if visualize==True:
            plt.scatter(np.array(data_exp['states']['x'])+1.44,np.array(data_exp['states']['y'])+3.7,label='Exp')
            plt.scatter(data_demo['Y_sim'],data_demo['X_sim'],label='Sim')
            plt.scatter(data_demo['X'],data_demo['Y'],label='reference')
            plt.legend()
            plt.show()
        Z_exp = np.array([data_demo['lw'],data_demo['V_ref']])
        Z_np = np.vstack((Z_np,Z_exp)) 
        risk_np = np.vstack((risk_np,risk_exp))  
        mean_np = np.vstack((mean_np,mean_exp))   
        max_np = np.vstack((max_np,max_exp))  
        final_np = np.vstack((final_np,final_exp))  
    Z_np = Z_np[1:,:]
    risk_np =  risk_np[1:,:]
    mean_np =  mean_np[1:,:]
    max_np =  max_np[1:,:]
    final_np =  final_np[1:,:]
    return Z_np, risk_np, mean_np, max_np, final_np

def generate_ref_data(N):
    #Collect experimental data from "N" UNIFORM experiments
    Z_np = np.zeros((1,2))
    risk_np = np.zeros((1,1))
    for i in range(N):
        file_demo = open('/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/uniform_'+str(i)+'.pkl','rb')
        data_demo = pickle.load(file_demo)
        file_exp = open('/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/uniform_experiments/data_initial_'+str(i)+'.pkl','rb')
        data_exp = pickle.load(file_exp)
        file_exp.close()
        file_demo.close()
        breakpoint()
        risk_exp = risk(data_exp['states']['x'],data_exp['states']['y'],data_demo['X'],data_demo['Y'])
        risk_sim = risk(data_demo['X_sim'],data_demo['Y_sim'],data_demo['X'],data_demo['Y'])
        print('Risk_exp:',risk_exp,'Risk_sim:',risk_sim,'ID:',i)
        Z_exp = np.array([data_demo['V_ref'],data_demo['lw']])
        Z_np = np.vstack((Z_np,Z_exp)) 
        risk_np = np.vstack((risk_np,risk_exp))   
    Z_np = Z_np[1:,:]
    risk_np =  risk_np[1:,:]
    return Z_np, risk_np

def generate_test_data(N=10):
    #Collect experimental data from "N" RANDOM experiments
    Z_np = np.zeros((1,2))
    risk_np = np.zeros((1,1))
    for i in range(N):
        file_demo = open('/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/test_'+str(i)+'.pkl','rb')
        data_demo = pickle.load(file_demo)
        file_exp = open('/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/test_experiments/data_test_'+str(i)+'.pkl','rb')
        data_exp = pickle.load(file_exp)
        file_exp.close()
        file_demo.close()
        #breakpoint()
        risk_exp = risk(data_exp['states']['x'],data_exp['states']['y'],data_demo['X'],data_demo['Y'])
        risk_sim = risk(data_demo['X_sim'],data_demo['Y_sim'],data_demo['X'],data_demo['Y'])
        print('Risk_exp:',risk_exp,'Risk_sim:',risk_sim,'ID:',i)
        Z_exp = np.array([data_demo['V_ref'],data_demo['lw']])
        Z_np = np.vstack((Z_np,Z_exp)) 
        risk_np = np.vstack((risk_np,risk_exp))   
    Z_np = Z_np[1:,:]
    risk_np =  risk_np[1:,:]
    return Z_np, risk_np

#Function for organizing data from various sources for initial training
def data_collect(N,N_Sim,visualize=True):

    #Collect experimental data from "N" epxeriments
    Z_np, risk_np, mean_np,max_np,final_np = generate_exp_data(N)

    #Collect data from simulations: Consists of algorithmic+noise failure regions that fail with confidence
    file_sim = open('/home/anjali/learn_from_sparse/f1tenth/data/gpr/sim/f1tenth_gpr_train.pkl','rb')
    data_sim = pickle.load(file_sim)
    file_sim.close()
    #Combine datasets from both sources
    #N_Sim= 327#900 #134
    #Z_full = np.vstack((np.array(data_sim['seed'][0:300:5]),np.array(data_sim['seed'][300:900:100]),Z_np))
    #risk_full = np.vstack((np.array(data_sim['risk'][0:300:5]).reshape((-1,1)),np.array(data_sim['risk'][300:900:100]).reshape((-1,1)),risk_np))
    Z_full = np.vstack((np.array(data_sim['seed'][0:N_Sim:1]),Z_np))
    risk_full = np.vstack((np.array(data_sim['risk'][0:N_Sim:1]).reshape((-1,1)),risk_np))
    mean_full = np.vstack((np.array(data_sim['mean'][0:N_Sim:1]).reshape((-1,1)),mean_np))
    max_full = np.vstack((np.array(data_sim['max'][0:N_Sim:1]).reshape((-1,1)),max_np))
    final_full = np.vstack((np.array(data_sim['final'][0:N_Sim:1]).reshape((-1,1)),final_np))
    risk_full = (torch.sigmoid(torch.tensor(risk_full-15)))
    mean_full = (torch.sigmoid(torch.tensor(mean_full-0.4)*5))
    max_full = (torch.sigmoid(torch.tensor(max_full-0.4)*5))
    final_full = (torch.sigmoid(torch.tensor(final_full-0.4)*5))
    #mean_full = torch.tensor(mean_full)
    #max_full = torch.tensor(max_full)
    #final_full = torch.tensor(final_full)
    #risk_full = risk_full/10
    
    #risk_full = np.clip(risk_full, a_min=None, a_max=20)
    #Tried randomizing data for generalization, didn't work, idk why that's a problem with Gaussian Processes
    #data_ = torch.tensor(np.hstack((obs_np,reward_np)))
    #data_=data_[torch.randperm(data_.size()[0])] 

    #Normalize data: gives good generalization
    #
    Z_full[:,1] = Z_full[:,1]/6.5
    Z_full[:,0] = (Z_full[:,0]-0.5)/4.0
 
    #Check if sizes make sense
    print('sim:',len(np.array(data_sim['seed'])))
    print("Z_full:",Z_full.shape)
    print("risk_full:",risk_full.shape)

    #Visualize the sim+experiment data:Optional
    if visualize==True:
        #plt.scatter(Z_full[:,0]*4.5,Z_full[:,1]*6.5,c=risk_full)
        plt.scatter(Z_full[:,0],Z_full[:,1],c=risk_full)
        plt.colorbar()
        plt.title('Sim+Experiment Data')
        plt.show()

    return torch.tensor(Z_full).float().to(device), risk_full.float().squeeze().to(device), mean_full.float().squeeze().to(device), max_full.float().squeeze().to(device), final_full.float().squeeze().to(device)

#Function for training the GPR with collected data
def train(obs,reward,N_Sim):
    # Initialize the likelihood and model
    x_train,y_train = obs,reward#torch.tensor(obs_full).float().to(device), torch.tensor(reward_full).float().squeeze().to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = SpectralMixtureGP(x_train, y_train,likelihood).to(device)

    # Put the model into training mode
    model.train()
    likelihood.train()

    # Use the Adam optimizer, with learning rate set to 0.1
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Use the negative marginal log-likelihood as the loss function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)

    # Set the number of training iterations
    n_iter = 2000
    lr_ = 0.001
    for i in range(n_iter):
        # Set the gradients from previous iteration to zero
        optimizer.zero_grad()
        # Output from model
        output = model(x_train)
        # Compute loss and backprop gradients
        #loss_sim = -mll(output[0:N_Sim], y_train[0:N_Sim])
        #loss_exp = -mll(output[N_Sim:], y_train[N_Sim:])
        #loss = loss_sim + loss_exp
        loss = -mll(output, y_train)
        loss.backward()
        #weight+= (loss_sim - loss_exp).detach().cpu()*lr_
        print('Iter %d/%d - Loss: %.3f' % (i + 1, n_iter, loss.item()))
        #print(loss_sim,loss_exp)
        optimizer.step()

    return  model,likelihood, mll


        
#For a given set of data-points, we check the prediction error and uncertainity 
def eval(x_test,y_test,model,likelihood):
    model.eval()
    likelihood.eval()
    x_test[7,:] = np.array([285,355])
    x_test = x_test/500
    #x_test = np.delete(x_test,[3,4,9],axis=0)
    #y_test = np.delete(y_test,[3,4,9],axis=0)
    y_test[7] = 0
    y_test = 1-y_test
    #breakpoint()
    observed_pred = likelihood(model(torch.tensor(x_test).to(device).float()))
    lower, upper = observed_pred.confidence_region()
    #variance = upper.numpy()-lower.numpy()
    y_pred = observed_pred.mean.detach().cpu().numpy()
    #breakpoint()
    error_sq = np.abs(y_pred-y_test.squeeze()) 
    fontsize = 20
    parameters = {
        'font.family': 'Times New Roman',
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'legend.fontsize': fontsize
    }
    plt.rcParams.update(parameters)
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 10)) 

    plt.plot(y_pred,c='r',linewidth=2,marker='P',label='Prediction')
    plt.plot(y_test,linestyle='--', linewidth=2,c='b',marker='P',label='Ground Truth')
    plt.axhline(0.3,0,10,linestyle='--',linewidth=2,color='k',label='$R_{th}$')
    plt.xlabel('Demonstration no.',fontsize=50)
    plt.ylabel('Risk',fontsize=50)
    plt.yticks(fontsize=40)
    plt.xticks(fontsize=40)
    #plt.title('title',**csfont)
    #plt.legend(fontsize=15)
    plt.show()
    
    breakpoint()
    print("Test Error:",error_sq)
    print("mean test error: ", error_sq.mean())
    print("max test error: ", error_sq.max())
    print("max test error: ", error_sq.std())
    breakpoint()
    return error_sq

#Visualize the results of a trained GPR
def visualize_gpr(Z_initial,y_test,model,likelihood):#model1,likelihood1,model2,likelihood2,model3,likelihood3):
    #model1.eval()
    #likelihood1.eval()

    #model2.eval()
    #likelihood2.eval()

    model.eval()
    likelihood.eval()

    # The gpytorch.settings.fast_pred_var flag activates LOVE (for fast variances)
    # See https://arxiv.org/abs/1803.06058
    #Z_search = np.zeros((30,2))
    Z_search = np.zeros((1,2))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Obtain the predictive mean and covariance matrix
        Y = np.linspace(0,6.5,30)
        X = np.linspace(0.5,4.5,30)
        #X = np.linspace(0.0,1.0,30)
        #Y = np.linspace(0.0,1.0,30)
        mean_1 = np.zeros((30,30))
        mean_2 = np.zeros((30,30))
        mean_3 = np.zeros((30,30))
        mean_preds = np.zeros((30,30))
        # min_mean = np.zeros((100))
        i=0
        for x in X:
            j=0
            # for y in Y:
                #x = (x)/500
                #y = (y)/500
            #mean_1[i,:] = likelihood1(model1(torch.tensor([x.repeat(Y.shape[0])[:, None],Y[:, None]]).squeeze().T.float().to(device))).mean.squeeze().cpu().detach().numpy()
            #mean_2[i,:] = likelihood2(model2(torch.tensor([x.repeat(Y.shape[0])[:, None],Y[:, None]]).squeeze().T.float().to(device))).mean.squeeze().cpu().detach().numpy()
            #mean_3[i,:] = likelihood3(model3(torch.tensor([x.repeat(Y.shape[0])[:, None],Y[:, None]]).squeeze().T.float().to(device))).mean.squeeze().cpu().detach().numpy()
            mean_preds[i,:] = likelihood(model(torch.tensor([x.repeat(Y.shape[0])[:, None],Y[:, None]]).squeeze().T.float().to(device))).mean.squeeze().cpu().detach().numpy()
            #mean_preds[i,:] = 20*mean_1[i,:] + 15*mean_2[i,:] + 10*mean_3[i,:]
            Y_arg = np.where(mean_preds[i,:]<12.5)
            #Y_arg =np.where( np.logical_or(mean_1[i,:]<0.5, np.logical_or(mean_2[i,:]<0.5, mean_3[i,:]<0.5))==True)
            repeated_array = np.full(len(Y_arg[0]), X[i])
            combined_array = np.column_stack((repeated_array, Y[Y_arg]))
            Z_search = np.vstack((Z_search,combined_array))
            i+=1
            
    #breakpoint()
    print("Z_initial",Z_initial.shape, Z_initial[-1,:])
    N_shape = Z_initial.shape[0]-10
    n_exp = 20-N_shape
    print("No. of remaining experiments",20-N_shape)
    
    Z_search = Z_search[1:,:]
    
    #kmeans = KMeans(n_clusters=10-N_shape, random_state=0, n_init="auto").fit(Z_search)
    kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(Z_search)
    labels_ = kmeans.predict(Z_search)
    Z_means1 = kmeans.cluster_centers_
    
    #breakpoint()
    if N_shape==0:
        Z_means = Z_means1
    elif N_shape==1:
        label_remove = kmeans.predict(Z_initial[-1,:].reshape(1,-1))
        Z_means = np.delete(Z_means1,label_remove,axis=0)
        print("Label Removed",label_remove)
    else:
        label_remove = kmeans.predict(Z_initial[-N_shape:,:])
        Z_means = np.delete(Z_means1,label_remove,axis=0)
        print("Label Removed",label_remove)
    

    min_dists = scipy.spatial.distance.cdist(Z_means,Z_initial)
    mean_dists = np.min(min_dists, axis=1)
    argmax_dist = np.argmax(mean_dists)
    Z_new = Z_means[argmax_dist]
    print(Z_new)
    print(argmax_dist)
    x_ref,y_ref,_ = lane_change_trajectory(v=1.5,lane_width=Z_new[0])
    steering, risk_, mean_dist, max_dist, final, x,y,cx,cy  = do_simulation(v=Z_new[1],lane_width=Z_new[0])
    file_i = open("/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/sequential_"+str(n_exp)+".pkl",'wb')
    data_i = {'X':y_ref,"Y":x_ref,"V_ref":Z_new[1],'lw':Z_new[0],'X_sim':x,'Y_sim':y}
    pickle.dump(data_i,file_i)
    file_i.close()
    #print("Predicted Risk:",likelihood(model(torch.tensor(Z_new.reshape(1,2)).float().to(device))).mean)
    # breakpoint()
    #print("Z_new",Z_new)
    plt.figure()
    plt.scatter(Z_search[:, 0], Z_search[:, 1], c=labels_, s=40, cmap='viridis')
    plt.scatter(Z_new[0],Z_new[1],marker="P",s=100,color='black')
    plt.scatter(Z_initial[:,0],Z_initial[:,1], s=50,color='red',label='Collected data')
    plt.legend()
    plt.title('GMM clustering')
    plt.show()
    #breakpoint()risk_init

    N = Z_initial.shape[0]
    plt.figure()
    #plt.contourf((X*4.0)+0.5,Y*6.5,mean_preds.T,levels=10)
    fig,axs= plt.subplots(4)
    axs[0].contourf(X,Y,mean_1.T,levels=10)
    axs[1].contourf(X,Y,mean_2.T,levels=10)
    axs[2].contourf(X,Y,mean_3.T,levels=10)
    axs[3].contourf(X,Y,mean_preds.T,levels=10)
    
    #plt.scatter(Z_initial[:,0],Z_initial[:,1],color='black',marker='P',s=60,label='$Z_{exp}$')
    #plt.colorbar()
    plt.ylim([0,6.5])
    plt.xlim([0.5,4.5])
    plt.title('Predicted Risk',fontsize=20)
    plt.ylabel('Y',fontsize=15)
    plt.xlabel('X',fontsize=15)

    #plt.savefig('/home/anjali/learn_from_sparse/f1tenth/media/GPR_pred'+str(N)+'.png')
    plt.show()
    breakpoint()
    """
    plt.figure()
    input_ = np.linspace(0,x_test.shape[0],x_test.shape[0])
    observed_pred = likelihood(model(x_test))
    lower, upper = observed_pred.confidence_region()
    plt.plot(input_,observed_pred.mean.detach().cpu(),label='GP Mean')
    plt.plot(input_,y_test,'o',color='red',label='Observed Data')
    plt.fill_between(input_,lower.cpu(), upper.cpu(), alpha=0.5,label='Confidence')
    plt.legend()
    plt.show()
    
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
    return Z_new

def main():
    N_final=40
    N= int(N_final/2)  #+ 10 #+5 #+12
    N_Sim= 476
    weight=0.9
    Z_init,risk_init,_,_,_ = generate_exp_data(N)
    print("Initial seeds",Z_init)
    for i in range(1):#range(N_final-N):
        #Zip data from sim+exp for training GPR
        Z,risk,mean,max_,final= data_collect(N,N_Sim,visualize=True)
        #breakpoint()
        #Train GPR using collected data

        # obs = torch.vstack((obs,x_test[3:19:4,:]
        model1,likelihood1,mll1 = train(Z,risk,N_Sim)
        #model1,likelihood1,mll1 = train(Z,mean,N_Sim)
        #model2,likelihood2,mll2 = train(Z,max_,N_Sim)
        #model3,likelihood3,mll3 = train(Z,final,N_Sim)

        #Visualize the predicted risk
        
        #breakpoint()
        #Evaluate performance on the initially chosen set of points. This tells you where to sample from and how to update the weight
        #score = eval(Z_init,reward_Z,model,likelihood)
        #weight = weight_update(weight,model,likelihood,obs,reward,N_Sim)
        Z_new = visualize_gpr(Z_init,risk_init,model1,likelihood1) #Interweaved optimization of the weight
        #Z_new = visualize_gpr(Z_init,mean,model1,likelihood1,model2,likelihood2,model3,likelihood3) #Interweaved optimization of the weight
        #Z_new = visualize_gpr(Z_init,max_,model2,likelihood2)
        #Z_new = visualize_gpr(Z_init,final,model3,likelihood3)
        #_,_,x_test,y_test = generate_ref_data(N=20)
        _,_,x_test,y_test = generate_test_data(N=10)
        error_sq = eval(x_test,y_test,model,likelihood)
        breakpoint()
        #Get index corresponding to the largest score in Z_init, sample from the corresponding cluster
        #search_idx = np.argmax(score)
        #Z_new = get_Z_new(search_idx)

        # Remove corresponding element from Z_init
        #Z_init= np.delete(Z_init,search_idx,axis=0) #Z_init shape needs to be NX 2 for this, else axis=1

        #This step should happen in experiment, with updating the data corresponding to Z_new
        N+=1

if __name__=="__main__":
    with ipdb.launch_ipdb_on_exception():
        main()

