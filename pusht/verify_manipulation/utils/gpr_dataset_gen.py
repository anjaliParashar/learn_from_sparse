import numpy as np
import pickle
import torch
import pickle
import matplotlib.pyplot as plt
# open a file, where you stored the pickled data

def dataset(plot=False):
    plot=True
    score_list = []
    seed_list = []
    for i in range(30):
        file_i = open('/home/anjalip/learn_from_sparse/pusht/verify_manipulation/data/l_10/pusht_train_T_pose_long'+str(i)+'.pkl', 'rb')
        # dump information to that file
        data_i = pickle.load(file_i)
        file_i.close()
        
        #Get scores and [X,Y] values from data dict
        score_list.append(data_i['score'])
        seed_list+=data_i['seed']

    #Score list to numpy
    score_np = np.array(score_list)
    score_np = 1-score_np #convert score to risk

    """
    score-shape: (Y_pts, L_pts, X_pts). X_pts and Y_pts refers to the no. of datapoints 
    on X and Y axis, L_pts is the number of divisions between L_min and L_max created. 
    """
    print('Score value',score_np.shape) 

    #Initialize np arrays to store variance and risk information
    var_np = np.zeros((score_np.shape[2],score_np.shape[0]))

    #Initialize X and Y arrays for visualization
    X = np.linspace(100,500,score_np.shape[2]) #100, 500 are hardcoded min and max limits 
    Y = np.linspace(100,500,score_np.shape[0])
    
    #Initialize lists for recording data from all four clusters
    seed1 = [] #High risk, high variance
    seed2 = [] #Low risk, high variance
    seed3 = [] #Low risk, Low variance
    seed4 = [] #High risk, Low variance
    score_1 = [] #High risk, high variance
    score_2 = [] #Low risk, high variance
    score_3 = [] #Low risk, Low variance
    score_4 =[] #High risk, Low variance

    fail = [] #record the actual no. of 'fail' datapoints
    for i in range(score_np.shape[2]): 
        for j in range(score_np.shape[0]):
            #Record range of risk from L_min=3.8 to L_max=4.2. Can use other metrics too
            var_length = np.max(score_np[j,3:8,i])-np.min(score_np[j,3:8,i])##np.average(score_np[j,:,i])###np.cov(score_np[j,:,i])#np.max(score_np[j,:,i])-np.min(score_np[j,:,i])#np.cov(score_np[:,i,j])#np.cov(score_np[j,:,i])
            var_np[i,j] = var_length
            risk_np = score_np[j,5,i] #"Simulated" risk, i.e, what we can observe from simulated model

            #Criteria for preliminary clustering:
            if var_length>0.3: #and risk_np>0.7:
                score_1.append(score_np[j,:,i])
                seed1.append([X[i],Y[j]])

            #These two clusters does not convey meaningful information from sensitivity pov
            elif var_length<0.3 and risk_np<0.7:
                score_2.append(score_np[j,:,i])
                seed2.append([X[i],Y[j]])
            
            else:
                score_3.append(score_np[j,:,i])
                seed3.append([X[i],Y[j]])
            """
            elif var_length>0.3 and risk_np<0.7:
                score_2.append(score_np[j,:,i])
                seed2.append([X[i],Y[j]])
            """
            if risk_np>=0.7:
                fail.append([X[i],Y[j]])

    #Visualize:
    seed1np = np.array(seed1)
    seed2np = np.array(seed2)
    seed3np = np.array(seed3)
    #seed4np = np.array(seed4)
    plt.scatter(seed1np[:,0],seed1np[:,1],label='High risk, High var')
    plt.scatter(seed2np[:,0],seed2np[:,1],label='Low risk, High var')
    plt.scatter(seed3np[:,0],seed3np[:,1],label='Low risk, Low var')
    #plt.scatter(seed4np[:,0],seed4np[:,1],label='High risk, Low var')
    plt.legend()
    plt.title("Clustering for GPR")
    plt.show()
    plt.savefig('CLuster_GPR.png')
        
    return seed1,seed2,seed3,seed4,score_1,score_2,score_3,score_4, fail

def gpr_score():
    seed1,seed2,seed3,seed4, score_1,score_2,score_3,score_4,fail = dataset()

    # High variance clusters, 'true': risk values for l=4.2, 'estimate' risk for l=4.0
    score_true1 = np.array(score_1)[:,7] 
    score_estimate1 = np.array(score_1)[:,5]

    score_true2 = np.array(score_2)[:,7] 
    score_estimate2 = np.array(score_2)[:,5] 

    # Low variance clusters, 'true': risk values for l=4.2, 'estimate' risk for l=4.0
    score_true3 = np.array(score_3)[:,7] 
    score_estimate3 = np.array(score_3)[:,5]

    #score_true4 = np.array(score_4)[:,7] 
    #score_estimate4 = np.array(score_4)[:,5] 

    #data = {'fail':fail,'seed1':seed1, 'seed2':seed2,'seed3':seed3,'seed4':seed4,'score_true1':score_true1,'score_true2':score_true2,'score_true3':score_true3,'score_true4':score_true4,'score_estimate1':score_estimate1,'score_estimate2':score_estimate2,'score_estimate3':score_estimate3,'score_estimate4':score_estimate4}
    data = {'fail':fail,'seed1':seed1, 'seed2':seed2,'seed3':seed3,'score_true1':score_true1,'score_true2':score_true2,'score_true3':score_true3,'score_estimate1':score_estimate1,'score_estimate2':score_estimate2,'score_estimate3':score_estimate3}
    
    filename='data/gpr/score_clusters.pkl'
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)
    file.close()
