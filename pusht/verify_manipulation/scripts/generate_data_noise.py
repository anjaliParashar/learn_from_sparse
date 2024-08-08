import sys
sys.path.append('/home/anjali/learn_from_sparse')
from pusht.verify_manipulation.utils.inference_final import generate_dist
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
    I_min = 100
    I_max = 500
    I_n = 30

    M = args['M']
    J_range = np.linspace(I_min,I_max,I_n)
    I_range = np.linspace(I_min,I_max,I_n)
    J_n = J_range[M]
   
    j=J_n
    timer=0
  
    seed=10000 #Fix the seed
    sigma_1range = np.linspace(0,1,10)
    sigma_2range = np.linspace(0,1,10)
    score_ = np.zeros((I_n,len(sigma_1range),len(sigma_2range)))
    seed_list =[]
    
    i_idx=0
    for i in I_range:
        sm1=0
        for sigma_1 in sigma_1range:
            sm2=0
            for sigma_2 in sigma_2range:
                #Initialize 1 random seed for policy evaluation  
                score = generate_dist(i,j,seed, sigma_1,sigma_2)
                score_[i_idx,sm1,sm2] = score
                seed_i+=1
                print(timer)
                seed0 = np.array([i,j])
                seed_list.append(seed0)
                timer+=1
                sm2+=1
            sm1+=1
        i_idx+=1

                    
    data = {'seed':seed_list,'score':score_}
    filename='/home/anjali/learn_from_sparse/pusht/verify_manipulation/data/I_30_seeds_3/pusht_4'+str(M)+'.pkl'
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)
    file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int)
    args = vars(parser.parse_args())
    with ipdb.launch_ipdb_on_exception():
        generate_dataset(args)
    