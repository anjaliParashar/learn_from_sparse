import numpy as np
import pickle
import torch
import pickle
import matplotlib.pyplot as plt
# open a file, where you stored the pickled data
file_0 = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/pusht_train_2.pkl', 'rb')
# dump information to that file
data_0 = pickle.load(file_0)
file_0.close()

score_np = data_0['score']#[-1]
print(score_np.shape)
length = np.linspace(4,10,5)
friction = np.linspace(0.1,1,5)

X,Y = np.meshgrid(length,friction)
#length_ = data_0['length']+data_1['length']
#friction_ = data_0['friction']+data_1['friction']

#Visualize 10 seeds (initial conditions)
fig, axs = plt.subplots(2,5, sharex=True, sharey=True, gridspec_kw={'hspace': 0},figsize=(15, 5))
for ax, i in zip(axs.flat,range(10)):
    cs = ax.contourf(friction,length,score_np[:,:,i])
    cbar = fig.colorbar(cs)
fig.supxlabel('Friction Coefficient',fontsize=15)
fig.supylabel('Length',fontsize=15)
plt.savefig('data_visualize.png')

for i in range(10):
    var_mu = np.cov(score_np[0,:,i]) #reference length value=4
    var_length = np.cov(score_np[:,-1,i]) #reference friction value = 1
    #cov = np.cov(score_np[:,:,i])
    print(var_mu,var_length)