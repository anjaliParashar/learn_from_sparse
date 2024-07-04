import numpy as np
import pickle
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# open a file, where you stored the pickled data

score_list = []
seed_list = []
for i in range(2):
    file_i = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/pusht_train_'+str(i)+'.pkl', 'rb')
    # dump information to that file
    data_i = pickle.load(file_i)
    file_i.close()
    score_list.append(data_i['score'])
    seed_list+=data_i['seed']

score_np = np.array(score_list)
score_np = score_np.reshape(-1, *score_np.shape[1:3]).swapaxes(0,2)
score_np = 1-score_np #convert score to risk
print('Score value',score_np.shape)

label_list = []
seed1 = []
seed2 = []
seed3 = []
for i in range(score_np.shape[2]):
    var_mu = np.cov(score_np[0,:,i]) #reference length value=4
    var_length = np.cov(score_np[:,-1,i]) #reference friction value = 1
    if var_length/var_mu >5:
        label = 0
        seed1.append(seed_list[i])
    elif var_length/var_mu<1:
        label=1
        seed2.append(seed_list[i])
    else:
        label=2
        seed3.append(seed_list[i])
    label_list.append(label)
print('label-0:',len(seed1),'label-1:',len(seed2),'label-2:',len(seed3))

#Unlabeled data
file_un = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/pusht_train_2.pkl', 'rb')
data_un = pickle.load(file_un)
file_un.close()
label_un = [-1]*10

#Pack data into a pickle file
label_list += label_un
seed_list = np.linspace(0,290,30)
label_list = np.array(label_list)
data = {'labels':label_list,'data':seed_list}
filename='data/pusht_fail_dist.pkl'
file = open(filename, 'wb')

# dump information to that file
pickle.dump(data, file)
file.close()

def visualize_dist():
    plt.plot(seed1,'o',label='Label-0')
    plt.plot(seed2,'o',label='Label-1')
    plt.plot(seed3,'o',label='Label-2')
    plt.legend()
    plt.show()