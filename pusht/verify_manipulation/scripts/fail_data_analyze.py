#This script analyzes the risk data obtained by inserting noise in dynamics and state estimation
import numpy as np
import pickle
import torch
import pickle
import matplotlib.pyplot as plt
# open a file, where you stored the pickled data

#def dataset(plot=False):
plot=True
score_list = []
seed_list = []
for i in range(30):
    file_i = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/I_30_seeds_3_delete_3/pusht_4'+str(i)+'.pkl', 'rb')
    # load information from file
    data_i = pickle.load(file_i)
    file_i.close()
    score_list.append(data_i['score'])
    seed_list+=data_i['seed']

score_np = np.array(score_list)

#score_np = score_np.reshape(-1, *score_np.shape[1:3]).swapaxes(0,2)
score_np = 1-score_np #convert score to risk
print('Score value',score_np.shape)
seed1 = []
seed2 = []
seed3 = []
var_np1 = np.zeros((30,30))
var_np2 = np.zeros((30,30))
risk_np = np.zeros((30,30))
seed1 = []
seed2 = []
seed3 =[]
score_1 = []
score_2 = []
score_3 = []

label_1 = []
label_2 = []
label_3 = []
X = np.linspace(100,500,30)
Y = np.linspace(100,500,30)
fail = []
for i in range(score_np.shape[1]): #X
    for j in range(score_np.shape[0]): #Y
        var_sigma1 = np.max(score_np[j,i,:,0])-np.min(score_np[j,i,:,0])#np.cov(score_np[j,3:7,i])#np.average(score_np[j,:,i])#np.max(score_np[j,3:7,i])-np.min(score_np[j,3:7,i])##np.cov(score_np[j,:,i])#np.max(score_np[j,:,i])-np.min(score_np[j,:,i])#np.cov(score_np[:,i,j])#np.cov(score_np[j,:,i])
        var_sigma2 = np.max(score_np[j,i,0,:])-np.min(score_np[j,i,0,:])

        var_np1[i,j] = var_sigma1
        var_np2[i,j] = var_sigma2
        risk_np[i,j] = score_np[j,i,0,0]
        #if var_sigma1>0.25 or var_sigma2>0.25: #High variance region
        #    score_1.append(score_np[j,i,0,0])
        #    seed1.append([X[i],Y[j]])
        #    label_1.append(0)
        if (score_np[j,i,1:5,1:5]>=0.3).all():#score_np[j,i,0,0]>=0.7 and var_sigma1<0.1 and var_sigma2<0.1: #Fail with certainity
            score_2.append(score_np[j,i,0,0])
            seed2.append([X[i],Y[j]])
            label_2.append(1)
        else:
            score_1.append(score_np[j,i,0,0])
            seed1.append([X[i],Y[j]])
            label_1.append(0)
        
        if (score_np[j,i,1:5,1:5]<=0.3).all():#score_np[j,i,0,0]<0.3 and var_sigma1<0.1 and var_sigma2<0.1: #Pass with certainity
            score_3.append(score_np[j,i,0,0])
            seed3.append([X[i],Y[j]])
            label_3.append(2)
        
        if score_np[j,i,0,0]>=0.7:
            fail.append([X[i],Y[j]])
seed2np = np.array(seed2)
seed1np = np.array(seed1)
seed3np = np.array(seed3)

plt.scatter(seed2np[:,0],seed2np[:,1],label='seed2')
plt.scatter(seed1np[:,0],seed1np[:,1],label='seed1')
plt.legend()

data = seed1[::10]+seed2#+seed3
labels=label_1[::10]+label_2#+label_3

data={'data':np.array(data),'labels':np.array(labels)}
file = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/pusht_cluster_train_2.pkl','wb')
pickle.dump(data,file)
file.close()

data_gpr={'seed':np.array(seed2+seed1),'risk':np.array(score_2+seed1)}
#data_gpr={'seed':np.array(seed2+seed1[::50]),'risk':np.array(score_2+score_1[::50])}
file = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/gpr/sim/pusht_gpr_train.pkl','wb')
pickle.dump(data_gpr,file)
file.close()
#breakpoint()
if plot==True:
    plt.contourf(X,Y,var_np2.T,levels=3)
    plt.colorbar()
    plt.title('Risk variance across 10 values of length for sigma_1')
    plt.scatter(256,256,s=50,color='red')
    plt.xlabel('X',fontsize=15)
    plt.ylabel('Y',fontsize=15)
    plt.show()
    #plt.savefig('contour_risk_2_seed.png')

    fig,axs=plt.subplots(5,5)
    for i in range(5):
        for j in range(5):
            axs[i,j].contourf(X,Y,score_np[:,:,i,j].T,levels=2)
    plt.show()
    
    """plt.figure()
    d = np.ma.array(score_np[:,5,:], mask=var_np <0.3)
    plt.contourf(X[7:25],Y[7:25],risk_np[7:25,7:25].T,levels=10)
    
    plt.colorbar()
    plt.scatter(256,256,s=200,color='red')
    plt.title('Risk for l=3.5',fontsize=20)

    plt.figure()
    #plt.contourf(X,Y,d,levels=10)
    plt.contourf(X,Y,score_np[:,5,:],levels=10)
    plt.colorbar()
    plt.title('Risk for l=4.0',fontsize=20)

    plt.figure()
    plt.contourf(X[7:25],Y[7:25],score_np[7:25,7,7:25],levels=10)
    plt.colorbar()
    plt.title('Risk for l=4.2',fontsize=20)
    plt.show()"""
    #plt.show()
        #plt.savefig('contour_risk_35_2.png')
        #breakpoint()
    
#    return seed1,seed2,score_1,score_2, fail

