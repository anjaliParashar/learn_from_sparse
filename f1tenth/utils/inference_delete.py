import torch
import sys
sys.path.append('/home/anjali/work/SQP/nla_falsifier/LQR')
from network import AutoEncoder,TrajectoryDataset
import pickle
from helpers import get_trajectory,get_control
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoEncoder().to(device)
optimizer = torch.optim.Adam(model.parameters())
nn_loss = torch.nn.MSELoss()


checkpoint = torch.load('/home/anjali/work/SQP/nla_falsifier/LQR/checkpoints/1024_500k_n1_500/vae_1024_500k_500_n1.pt')
#checkpoint = torch.load('/home/anjali/work/results/dump_vae_st_1024_80k_500_n1.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

file = open('/home/anjali/work/SQP/nla_falsifier/LQR/checkpoints/vae_train_data_80k.pkl', 'rb')
#file = open('vae_train_data_st_80k.pkl','rb')
data = pickle.load(file)
file.close()
train_dataset = TrajectoryDataset(data,device)
dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)
#def decode(theta):
#theta = torch.ones((1,4)).to(device)
#print(model.decoder(theta))
"""
def decode(theta):
    decoded = model.decoder(theta).to(device)
    model.eval()
    batch_size =1
    phi_max = torch.pi/20
    a_max = 30
    phi_learn = torch.cat((decoded[:,0:5].reshape((batch_size,5,1)),decoded[:,5:10].reshape((batch_size,5,1))),-1)*torch.pi/2
    a_learn = torch.cat(((2*decoded[:,10:15]-1).reshape((batch_size,5,1))*phi_max,(2*decoded[:,15:20]-1).reshape((batch_size,5,1))*a_max),-1)
    p_learn = decoded[:,-10:].reshape((batch_size,5,2))*200
"""
def get_reference_vae(theta):
    #data = next(iter(dataloader))
    model.eval()
    batch_size =1
    phi_max = torch.pi/20
    a_max = 30
    decoded = model.decoder(theta).to(device)
    phi_learn = torch.cat((decoded[:,0:5].reshape((batch_size,5,1)),decoded[:,5:10].reshape((batch_size,5,1))),-1)*torch.pi/2
    a_learn = torch.cat(((2*decoded[:,10:15]-1).reshape((batch_size,5,1))*phi_max,(2*decoded[:,15:20]-1).reshape((batch_size,5,1))*a_max),-1)
    p_learn = decoded[:,-10:].reshape((batch_size,5,2))*200
    state_learn = get_trajectory(phi_learn,a_learn,p_learn)[:,:,0:2,:].reshape((100,2))
    x_ = list(state_learn[:,0].detach().cpu())
    y_ = list(state_learn[:,1].detach().cpu())
    return x_,y_,state_learn[:,0],state_learn[:,1]
#x_,y_ = get_reference_vae(theta = torch.ones((1,4)).to(device))
#print(x_,y_)
#get_reference_vae()
if False:
    print(loss,epoch)
    model.eval()
    batch_size =1
    
    phi_max = torch.pi/20
    a_max = 30
    """
    v_max = 10
    omega_max = 0.1
    """
    state_list = []
    state_learn_list = []
    error_list = []
    encodings = []
    action_list = []
    for batch_idx, data in enumerate(dataloader):
        encoded, decoded = model(data['data'].to(device))
        
        phi_learn = torch.cat((decoded[:,0:5].reshape((batch_size,5,1)),decoded[:,5:10].reshape((batch_size,5,1))),-1)*torch.pi/2
        a_learn = torch.cat(((2*decoded[:,10:15]-1).reshape((batch_size,5,1))*phi_max,(2*decoded[:,15:20]-1).reshape((batch_size,5,1))*a_max),-1)
        p_learn = decoded[:,-10:].reshape((batch_size,5,2))*200
        """
        phi_learn = torch.cat((decoded[:,0:5].reshape((batch_size,5,1)),decoded[:,5:10].reshape((batch_size,5,1))),-1)*torch.pi/2
        a_learn = torch.cat(((decoded[:,10:15]).reshape((batch_size,5,1))*v_max,(2*decoded[:,15:20]-1).reshape((batch_size,5,1))*omega_max),-1)
        p_learn = decoded[:,-10:].reshape((batch_size,5,2))*0.1
        """
        encodings.append(encoded.reshape((4,)).detach().cpu())
        #print(encoded[:,1])
        action_learn = get_control(phi_learn,a_learn,p_learn).reshape((100,2))
        #print(action_learn.shape)
        action_list.append(action_learn.detach().cpu().numpy())
        state_learn = get_trajectory(phi_learn,a_learn,p_learn)[:,:,0:2,:]
        state_learn = state_learn.transpose(0,1).reshape((100,2))
        #state = data['state'][:,:,0:2,:].reshape((100,2)).to(device)
        #error_list.append(nn_loss(state,state_learn).detach().cpu())
        #state_list.append(state.detach().cpu())
        state_learn_list.append(state_learn.detach().cpu().numpy())
        if batch_idx==1000:
            break
    encodings = np.array(encodings)
    n_bins = 100
    print(encodings.shape)
if False:
    """
    fig, axs = plt.subplots(3, 1)
    axs[0, 0].scatter(encodings[:,0],encodings[:,1],linewidth=0.1)
    axs[0, 0].set_title('Correlation 1 & 2')
    axs[0, 1].scatter(encodings[:,1],encodings[:,2],linewidth=0.1)
    axs[0, 1].set_title('Correlation 2 & 3')
    axs[1, 0].scatter(encodings[:,2],encodings[:,3],linewidth=0.1)
    axs[1, 0].set_title('Correlation 3 & 4')
    axs[1, 1].scatter(encodings[:,1],encodings[:,3],linewidth=0.1)
    axs[1, 1].set_title('Correlation 2 & 4')
    axs[2, 0].scatter(encodings[:,0],encodings[:,2],linewidth=0.1)
    axs[2, 0].set_title('Correlation 1 & 3')
    axs[2, 1].scatter(encodings[:,0],encodings[:,3],linewidth=0.1)
    axs[2, 1].set_title('Correlation 1 & 4')
    plt.show()
    """
    fig, axs = plt.subplots(4,1)
    axs[0].hist(encodings[:,0],bins = n_bins)
    axs[0].set_title('Encoding 1')
    axs[1].hist(encodings[:,1],bins = n_bins)
    axs[1].set_title('Encoding 2')
    axs[2].hist(encodings[:,2],bins = n_bins)
    axs[2].set_title('Encoding 3')
    axs[3].hist(encodings[:,3],bins = n_bins)
    axs[3].set_title('Encoding 3')
    plt.show()
    #axs[1, 1].hist(encodings[:,3],bins=n_bins)
    #axs[1, 1].set_title('Encoding 4')
    #plt.savefig('1024_500k_500_n1_Encoding_Dist.png')

if False:
    plt.plot(error_list)
    plt.show()
#Need to modify this script for LQR and LQR_ST
if False:
    m = 5
    n = 5
    fig1, axs1 = plt.subplots(nrows=m, ncols=n, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig1.suptitle("Velocity curves", fontsize=18, y=0.95)
    for action, ax in zip(action_list, axs1.ravel()):
        # filter df for ticker and plot on specified axes
        ax.plot(action[:,0])
    #plt.savefig('Steering_angle_1024_500k_500_n1.png')

    fig2, axs2 = plt.subplots(nrows=m, ncols=n, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig2.suptitle("Omega curves", fontsize=18, y=0.95)
    for action, ax in zip(action_list, axs2.ravel()):
        # filter df for ticker and plot on specified axes
        ax.plot(action[:,1]*57.3)
    #plt.savefig('Acceleration_1024_500k_500_n1.png')

    fig2, axs2 = plt.subplots(nrows=m, ncols=n, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    fig2.suptitle("State curves", fontsize=18, y=0.95)
    for state_learn,ax in zip(state_learn_list, axs2.ravel()):
        # filter df for ticker and plot on specified axes
        #ax.plot(state[:,0],state[:,1],label='Actual')
        ax.plot(state_learn[:,0],state_learn[:,1])#,label='Prediction')
        #ax.legend()
    plt.show()
    #plt.savefig('State_curve_1024_500k_500_n1.png')
