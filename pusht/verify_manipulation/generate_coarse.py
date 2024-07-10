from inference import inference_pusht
import pickle
import numpy as np
from tqdm.auto import tqdm
import argparse
import torch
from network import ConditionalUnet1D

noise_pred_net = ConditionalUnet1D(
    input_dim=2,
    global_cond_dim=5*2
)
noise_pred_net.load_state_dict(torch.load('/home/anjali/push_T_diffusion_model'))
noise_pred_net = noise_pred_net.cuda()

noisy_action = torch.randn(
                        (1, 16,2), device='cuda')
def generate_dataset_length(args):
    I_min = args['I_min']
    I_max = args['I_max']
    I_n = args['I_n']
    J_n = args['J_n']
    M = args['M']
    m1 =1
    l_min,l_max,f_min,f_max = 3,5,0.1,1
    seed_list = []
    obs_list = []
    mass_list  =[]
    length_list = []
    friction_list  =[]
    score_ = np.zeros((100,I_n))
    l_idx = 0
    timer=0
    f1=1
    j=J_n
    for l1 in np.linspace(l_min,l_max,100):
        i_idx=0
        for i in np.linspace(I_min,I_max,I_n):
    
            pusht = inference_pusht(int(i),int(j),m1,f1,l1)
            obs0 = pusht.init_obs
            obs_list.append(obs0)
            score = pusht.generate_dist(noise_pred_net, noisy_action).detach()
            score_[l_idx,i_idx] = score
            seed0 = np.array([i,j])
            seed_list.append(seed0)
            mass_list.append(m1)
            length_list.append(l1)
            print(timer)
            timer+=1

            i_idx+=1
        l_idx+=1

                    
    print(score_)
    data = {'seed':seed_list,'length':length_list,'score':score_}
    filename='data/pusht_train_'+str(M)+'.pkl'
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)
    file.close()

def generate_dataset_all(args):

    I_min = args['I_min']
    I_max = args['I_max']
    I_n = args['I_n']
    J_n = args['J_n']
    M = args['M']
    m1 =1
    l_min,l_max,f_min,f_max = 3,5,0.1,1
    seed_list = []
    obs_list = []
    mass_list  =[]
    length_list = []
    friction_list  =[]
    score_ = np.zeros((2,2,I_n))
    l_idx = 0
    timer=0
    f1=1
    j=J_n
    for l1 in np.linspace(l_min,l_max,2):
        f_idx = 0
        for f1 in np.linspace(f_min,f_max,2):
            i_idx=0
            for i in np.linspace(I_min,I_max,I_n):
                pusht = inference_pusht(int(i),int(j),m1,f1,l1)
                obs0 = pusht.init_obs
                obs_list.append(obs0)
                score = pusht.generate_dist(noise_pred_net, noisy_action).detach()
                score_[l_idx,f_idx,i_idx] = score
                seed0 = np.array([i,j])
                seed_list.append(seed0)
                mass_list.append(m1)
                length_list.append(l1)
                friction_list.append(f1)
                print(timer)
                timer+=1
                i_idx+=1
            f_idx+=1
        l_idx+=1

                    
    print(score_)
    data = {'seed':seed_list,'length':length_list,'score':score_}
    filename='data/l_50_mu_50/pusht_train_'+str(M)+'.pkl'
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)
    file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--I_min", type=int)
    parser.add_argument("--I_max", type=int)
    parser.add_argument("--I_n", type=int)
    parser.add_argument("--J_n", type=int)
    parser.add_argument("--M", type=int)
    args = vars(parser.parse_args())
    generate_dataset_all(args)