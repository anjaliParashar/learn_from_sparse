from inference import inference_pusht
import pickle
import numpy as np
from tqdm.auto import tqdm
import argparse

def generate_dataset(args):
    l_min=args['l_min']
    f_min=args['f_min']
    l_max=args['l_max']
    f_max=args['f_max']
    m1 = 1
    seed_list = []
    score_list = []
    length_list = []
    friction_list  =[]
    for l1 in np.linspace(l_min,l_max,5):
        for f1 in np.linspace(f_min,f_max,5):
            for i in range(5,10,1):
                #i in range(1000):
                pusht = inference_pusht(i*1000,m1,f1,l1)
                score = pusht.generate_dist()
                score_list.append(score)
                seed_list.append(i*1000)
                length_list.append(l1)
                friction_list.append(f1)
                print(i)

                    
        
    data = {'seed':seed_list,'length':length_list,'friction':friction_list,'score':score_list}
    filename='pusht_train_small_1.pkl'
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)
    file.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--f_min", type=float)
    parser.add_argument("--f_max", type=float)
    parser.add_argument("--m_min", type=float)
    parser.add_argument("--m_max", type=float)
    parser.add_argument("--l_min", type=float)
    parser.add_argument("--l_max", type=float)
    args = vars(parser.parse_args())
    generate_dataset(args)