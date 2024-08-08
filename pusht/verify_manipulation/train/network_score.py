#@markdown ### **Network**
#@markdown
#@markdown Defines a 1D UNet architecture `ConditionalUnet1D`
#@markdown as the noies prediction network
#@markdown
#@markdown Components
#@markdown - `SinusoidalPosEmb` Positional encoding for the diffusion iteration k
#@markdown - `Downsample1d` Strided convolution to reduce temporal resolution
#@markdown - `Upsample1d` Transposed convolution to increase temporal resolution
#@markdown - `Conv1dBlock` Conv1d --> GroupNorm --> Mish
#@markdown - `ConditionalResidualBlock1D` Takes two inputs `x` and `cond`. \
#@markdown `x` is passed through 2 `Conv1dBlock` stacked together with residual connection.
#@markdown `cond` is applied to `x` with [FiLM](https://arxiv.org/abs/1709.07871) conditioning.
import numpy as np
import math
import torch
import torch.nn as nn
import pickle



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2,1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    

class TrajectoryDataset(torch.utils.data.Dataset):
    def __init__(self,filepaths):
        
        data = self.create_data(filepaths)
        # All demonstration episodes are concatinated in the first dimension N
        train_data = {
            # (N, action_dim)
            #'route': route_list,
            # (N, obs_dim)
            'theta': data['theta'],
            'seed': data['seed'],
            'score': data['score'],
        }
        self.train_data = train_data

    def create_data(self,filepaths):
        seed_list = []
        theta_list = []
        #score_np = np.zeros((3,3,20,40))
        score_np = np.zeros((100,10,10))
        j=0
        for file in filepaths:
            file_i = open(file, 'rb')
            # dump information to that file
            data_i = pickle.load(file_i)
            file_i.close()
            seed_list+=data_i['seed']
            score_np[:,:,j] = data_i['score']
            theta_list += data_i['length']
            print(len(data_i['length']))
            j+=1
        score_list = list(score_np.flatten())
        data = {'score':score_list,'theta':theta_list,'seed':seed_list}
        print('score',len(score_list))
        print('theta',len(theta_list))
        print('seed',len(seed_list))
        return data
            

    def __len__(self):
        # all possible segments of the dataset
        return len(self.train_data['seed'])

    def __getitem__(self, idx):

        # get nomralized data using these indices
        train_seed = np.array(self.train_data['seed'][idx])
        train_seed = np.array([(train_seed[0]-100)/400,(train_seed[1]-100)/400])
        train_seed = (2*train_seed) -1

        train_theta =np.array(self.train_data['theta'][idx])
        train_theta = (train_theta-3)/2
        train_theta = (2*train_theta) -1

        train_risk = np.array(self.train_data['score'][idx])
        train_risk = np.reshape(train_risk,(1,))

        train_data_idx = {
            #'route':train_route,
            'theta': train_theta,
            'seed':train_seed,
            'score': train_risk,
        }
        return train_data_idx

