from inference_final import generate_dist
import pickle
import numpy as np
from tqdm.auto import tqdm
import argparse
import ipdb

"""noise_pred_net = ConditionalUnet1D(
    input_dim=2,
    global_cond_dim=5*2
)
noise_pred_net.load_state_dict(torch.load('/home/anjalip/push_T/verify_manipulation/models/push_T_diffusion_model'))
noise_pred_net = noise_pred_net.cuda()"""



def generate_dataset(args):
    
    #I_min = args['I_min']
    #J_min = args['J_min']
    #I_max = args['I_max']
    #J_max = args['J_max']
    #I_n = args['I_n']
    I_min = 100
    I_max = 500
    I_n = 30

    #J_n = args['J_n']
    M = args['M']
    J_range = np.linspace(100,500,30)
    J_n = J_range[M]
    
    seed_list = []
    obs_list = []
    mass_list  =[]
    length_list = []
    score_ = np.zeros((10,I_n,3))
    #reward_ = np.zeros((100,I_n,6))
    #action_ = np.zeros((100,I_n,210,2))
    #obs_ = np.zeros((100,I_n,201,5))
    l_idx = 0
    seed=0
    j=J_n

    timer=0
    #J_min=100
    #J_max=500
    #J_n =30
    
    seeds=[1000,2000,3000]
    for length in np.linspace(l_min,l_max,10):
    #for j in np.linspace(J_min,J_max,J_n):
        i_idx=0
        for i in np.linspace(I_min,I_max,I_n):
            #Initialize 10 random seeds for policy evaluation  
            seed_i = 0   
            for seed in seeds:        
                score = generate_dist(i,j,length,seed)
                score_[l_idx,i_idx,seed_i] = score
                seed_i+=1
            #action_np = np.vstack(action_list)
            #obs_np = np.vstack(obs_batch_list)
            #reward_np = np.array(rewards)

            #action_[l_idx,i_idx,:,:] = action_np
            #obs_ [l_idx,i_idx,:,:]= obs_np
            #reward_[l_idx,i_idx,:,]= reward_np

            #obs_np = np.array(obs_batch_list)
            #breakpoint()
            print(timer)
            seed0 = np.array([i,j])
            seed_list.append(seed0)
            length_list.append(length)
            seed+=100
            i_idx+=1
            timer+=1
        l_idx+=1

                    
    #data = {'seed':seed_list,'length':length_list,'score':score_}#,'reward':reward_,'action':action_,'obs':obs_}
    #filename='data/l_10/pusht_train_T_pose_long'+str(M)+'.pkl'
    data = {'seed':seed_list,'score':score_}
    filename='data/l_10_seeds_3/pusht_4'+str(M)+'.pkl'
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)
    file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--I_min", type=int)
    #parser.add_argument("--J_min", type=int)
    #parser.add_argument("--I_max", type=int)
    #parser.add_argument("--J_max", type=int)
    #parser.add_argument("--I_n", type=int)
    #parser.add_argument("--J_n", type=int)
    parser.add_argument("--M", type=int)
    args = vars(parser.parse_args())
    with ipdb.launch_ipdb_on_exception():
        generate_dataset(args)
    