import sys
sys.path.append('/home/anjali/learn_from_sparse')
from f1tenth.utils.simulation import generate_dist
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
    v_min = 0.5
    v_max = 4.5
    v_n = 30

    curv_min = 0.5
    curv_max = 4.5
    curv_n = 30

    M = args['M']
    curv_range = np.linspace(curv_min,curv_max,curv_n)
    v_range = np.linspace(v_min,v_max,v_n)
    curv = curv_range[M]
    print("Curv:",curv)
    timer=0
  
    seed=10000 #Fix the seed
    sigma_1range = np.logspace(-4,0,5)
    sigma_2range = np.logspace(-5,-3,5)
    risk_ = np.zeros((curv_n,len(sigma_1range),len(sigma_2range)))
    seed_list =[]
    
    i_idx=0
    for speed in v_range:
        sm1=0
        for sigma_1 in sigma_1range:
            sm2=0
            for sigma_2 in sigma_2range:
                #Initialize 1 random seed for policy evaluation  
                risk = generate_dist(sigma_1,sigma_2,speed,curv)
                risk_[i_idx,sm1,sm2] = risk
                print(timer)
                seed0 = np.array([curv,speed])
                seed_list.append(seed0)
                timer+=1
                sm2+=1
            sm1+=1
        i_idx+=1

                    
    data = {'seed':seed_list,'risk':risk_}
    filename='/home/anjali/learn_from_sparse/f1tenth/data/speed_30_sigma_5/f1tenth'+str(M)+'.pkl'
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
    