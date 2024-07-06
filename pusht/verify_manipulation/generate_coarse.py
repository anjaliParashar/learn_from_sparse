from inference import inference_pusht
import pickle
import numpy as np
from tqdm.auto import tqdm
import argparse

def generate_dataset(args):
   # l_min=args['l_min']
   # m_min=args['m_min']
   # f_min=args['f_min']
   # l_max=args['l_max']
   # m_max=args['m_max']
   # f_max=args['f_max']
    I_min = args['I_min']
    J_min = args['J_min']
    I_max = args['I_max']
    J_max = args['J_max']
    I_n = args['I_n']
    J_n = args['J_n']
    M = args['M']
    m1 =1
    l_min,l_max,f_min,f_max = 4,8,0.1,1
    seed_list = []
    obs_list = []
    mass_list  =[]
    length_list = []
    friction_list  =[]
    score_ = np.zeros((3,3,I_n,J_n))
    l_idx = 0
    timer=0
    for l1 in np.linspace(l_min,l_max,3):
        f_idx = 0
        for f1 in np.linspace(f_min,f_max,3):
            i_idx=0
            for i in np.linspace(I_min,I_max,I_n):
                j_idx=0
                for j in np.linspace(J_min,J_max,J_n):
                    #i in range(1000):
                    pusht = inference_pusht(int(i),int(j),m1,f1,l1)
                    obs0 = pusht.init_obs
                    obs_list.append(obs0)
                    #print(obs0,i,j)
                    score = pusht.generate_dist().detach()
                    score_[l_idx,f_idx,i_idx,j_idx] = score
                    seed0 = np.array([i,j])
                    seed_list.append(seed0)
                    #seed_list.append(i*100)
                    mass_list.append(m1)
                    length_list.append(l1)
                    friction_list.append(f1)
                    print(timer)
                    timer+=1
                    j_idx+=1
                i_idx+=1
            f_idx+=1
        l_idx+=1

                    
        
    data = {'seed':seed_list,'length':length_list,'friction':friction_list,'score':score_}
    filename='data/pusht_train_'+str(M)+'.pkl'
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)
    file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--I_min", type=int)
    parser.add_argument("--J_min", type=int)
    parser.add_argument("--I_max", type=int)
    parser.add_argument("--J_max", type=int)
    parser.add_argument("--I_n", type=int)
    parser.add_argument("--J_n", type=int)
    parser.add_argument("--M", type=int)
    args = vars(parser.parse_args())
    generate_dataset(args)