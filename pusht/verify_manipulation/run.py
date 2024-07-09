import numpy as np
import pickle
import torch
import pickle
import matplotlib.pyplot as plt
from network_score import NeuralNetwork, TrajectoryDataset
from inference import inference_pusht

device='cuda'
model = NeuralNetwork().to(device)
PATH = '/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/model.pt'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
loss = checkpoint['loss']

file = '/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/pusht_fail_dist_2d_extra.pkl'
file_var = open(file, 'rb')
# dump information to that file
data_var = pickle.load(file_var)
file_var.close()

data_ = data_var['data']
seed0 = data_[data_var['labels']==0]
seed1 = data_[data_var['labels']==1]
seed2 = data_[data_var['labels']==2]

score_list = []
score_true_list = []
i = 0
for seed in seed0:
    print(seed)
    X1 = torch.tensor((1)).to(device).float().reshape((-1,1))
    X2_train = np.array([(seed[0]-100)/200,(seed[1]-100)/400])
    X2_train = (2*X2_train) -1
    X2_train = torch.tensor(X2_train).to(device).float().reshape((1,-1))
    X_input = torch.hstack((X1,X2_train))
    score_pred = model(X_input).float().squeeze()
    score_list.append(score_pred.detach().cpu())
    pusht = inference_pusht(seed[0],seed[1],1,1,8)
    score_true = pusht.generate_dist().detach()
    score_true_list.append(score_true)
    print('score_true:',score_true,'score_pred:',score_pred)
    if i==10:
        break
    i+=1

plt.plot(score_list,'o',label='"Predict')
plt.plot(score_true_list,'o',label="True")
plt.legend()
plt.show()

