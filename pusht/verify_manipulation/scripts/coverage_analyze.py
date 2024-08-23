import numpy as np
import pickle
import matplotlib.pyplot as plt
import copy
Z_xr = [2,1,3,2,2]
Z_yr = [0,0,0,1,-1]
Z_list =[]
dispersion_list = []
reward_list = []
for Z_x,Z_y in zip(Z_xr, Z_yr):
    print(Z_x,Z_y)
    file = open("/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/MH/MH_"+str(Z_x)+ "_"+str(Z_y)+".pkl", 'rb')
    # dump information to that file
    data = pickle.load(file)
    file.close()
    dispersion_list+=data['disp']
    reward_list +=data['reward']
    Z_list+=data['Z']
    

Z_np = np.array(Z_list).squeeze()
dispersion_np = np.array(dispersion_list).squeeze()

plt.scatter(Z_np[:,0],Z_np[:,1])
plt.show()

plt.figure()
plt.plot(dispersion_np,'o')
plt.show()

#Analysis of results
N_idx = np.where(dispersion_np>0.29)
print(N_idx)
print('Z:',Z_np[N_idx,:])
print('Disperson:',dispersion_np[N_idx])
Z_plt = Z_np[N_idx,:].squeeze()
plt.scatter(Z_plt[:,0],Z_plt[:,1])
plt.show()
#plt.plot(reward_list[N_idx],'o')