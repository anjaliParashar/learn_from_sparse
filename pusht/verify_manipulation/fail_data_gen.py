import numpy as np
import pickle
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# open a file, where you stored the pickled data


score_list = []
seed_list = []
N = [400,3200]
score_np = np.zeros((3,3,20,40))
j=0
for i in N:
    file_i = open('/home/anjalip/learn_from_sparse/pusht/verify_manipulation/data/pusht_train_'+str(i)+'.pkl', 'rb')
    # dump information to that file
    data_i = pickle.load(file_i)
    file_i.close()
    score_np[:,:,:,j:j+20] = data_i['score']
    seed_list+=data_i['seed']
    
    j+=20
score_np = 1 - score_np
label_list = []
seed1 = []
seed2 = []
seed3 = []
I=100
var_list = []
for i in range(score_np.shape[2]): #x
    J=100
    for j in range(score_np.shape[3]): #y
        #var_mu = np.cov(score_np[:,0,i,j]) #reference length value=4
        #print(score_np[0,:,i,j])
        var_length = np.cov(score_np[:,-1,i,j]) #reference friction value = 1
        var_list.append(var_length)
        #cond_num = var_length/var_mu
        #print(var_length,var_mu)
        if  var_length>0.2:
            label = 0
            seed1.append([I,J])
        elif var_length>0.1 and var_length<0.2:
            label=1
            seed2.append([I,J])
        else:
            label=2
            seed3.append([I,J])
        label_list.append(label)
        J+=10
    I+=10

#var_list.sort()
#print(var_list[0:10],var_list[-10:])
print('label-0:',len(seed1),'label-1:',len(seed2),'label-2:',len(seed3))
#Unlabeled data

#file_un = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/pusht_train_11000.pkl', 'rb')
#data_un = pickle.load(file_un)

#file_un.close()
#label_un = [-1]*10

#Pack data into a pickle file
print(len(label_list))
seed_list = seed1+seed2+seed3
seed1 =  np.array(seed1)
seed2 =  np.array(seed2)
seed3 =  np.array(seed3)
seed_list = np.array(seed_list)
label_list = np.array(label_list)
data = {'labels':label_list,'data':seed_list}
filename='data/pusht_fail_dist_2d.pkl'
file = open(filename, 'wb')

# dump information to that file
pickle.dump(data, file)
file.close()

def visualize_dist():
    plt.scatter(seed1[:,0],seed1[:,1],label='Label-0')
    plt.scatter(seed2[:,0],seed2[:,1],label='Label-1')
    plt.scatter(seed3[:,0],seed3[:,1],label='Label-2')
    plt.legend()
    plt.savefig('data_visualize_2d.png')
    plt.show()
visualize_dist()
