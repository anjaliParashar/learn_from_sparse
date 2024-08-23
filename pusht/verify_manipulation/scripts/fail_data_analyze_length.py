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
        #file_i = open('/home/anjalip/learn_from_sparse/pusht/verify_manipulation/data/l_100/pusht_train_batched_length_extra'+str(i+1)+'.pkl', 'rb')
        #file_i = open('/home/anjalip/learn_from_sparse/pusht/verify_manipulation/data/l_10/pusht_train_T_pose'+str(i+1)+'.pkl', 'rb')
        file_i = open('/home/anjalip/learn_from_sparse/pusht/verify_manipulation/data/l_10/pusht_train_T_pose_long'+str(i)+'.pkl', 'rb')
        # dump information to that file
        data_i = pickle.load(file_i)
        file_i.close()
        score_list.append(data_i['score'])
        seed_list+=data_i['seed']

    score_np = np.array(score_list)
    #score_np = score_np.reshape(-1, *score_np.shape[1:3]).swapaxes(0,2)
    score_np = 1-score_np #convert score to risk
    #score_np = np.array(torch.sigmoid(20*torch.tensor(score_np)-10))
    print('Score value',score_np.shape)
    label_list = []
    seed1 = []
    seed2 = []
    seed3 = []
    var_np = np.zeros((30,30))
    risk_np = np.zeros((30,30))
    score_np = score_np[:,0:10,:]
    seed1 = []
    seed2 = []
    score_1 = []
    score_2 = []
    X = np.linspace(100,500,30)
    Y = np.linspace(100,500,30)
    #var_np = np.zeros((50,3))
    fail = []
    for i in range(score_np.shape[2]):
        for j in range(score_np.shape[0]):
            var_length = np.max(score_np[j,3:8,i])-np.min(score_np[j,3:8,i])#np.cov(score_np[j,3:7,i])#np.average(score_np[j,:,i])#np.max(score_np[j,3:7,i])-np.min(score_np[j,3:7,i])##np.cov(score_np[j,:,i])#np.max(score_np[j,:,i])-np.min(score_np[j,:,i])#np.cov(score_np[:,i,j])#np.cov(score_np[j,:,i])
            var_np[i,j] = var_length
            risk_np[i,j] = score_np[j,0,i]
            if var_length>0.05:
                score_1.append(score_np[j,:,i])
                seed1.append([X[i],Y[j]])
            else:
                score_2.append(score_np[j,:,i])
                seed2.append([X[i],Y[j]])
            if risk_np[i,j]>=0.7:
                fail.append([X[i],Y[j]])
            #if risk_np[i,j]>0.9:
            #    print(i,j)

    #1st stage filtering
    X_list = []
    Y_list = []
    score_sensitive = []
    var_sensitive = []
    for i in range(score_np.shape[2]):
        for j in range(score_np.shape[0]):
            var_length = np.max(score_np[j,4:8,i])-np.min(score_np[j,4:8,i])#np.cov(score_np[j,3:7,i])#np.average(score_np[j,:,i])#np.max(score_np[j,3:7,i])-np.min(score_np[j,3:7,i])##np.cov(score_np[j,:,i])#np.max(score_np[j,:,i])-np.min(score_np[j,:,i])#np.cov(score_np[:,i,j])#np.cov(score_np[j,:,i])
            var_np[i,j] = var_length
            if var_length>0.12:
                score_sensitive.append(score_np[j,:,i])
                X_list.append(X[i])
                Y_list.append(Y[i])
                var_sensitive.append(var_length)

    #plt.figure()
    #X = np.linspace(200,450,50)
    #Y = np.linspace(299,307,3)
    file_i = open('/home/anjalip/learn_from_sparse/pusht/verify_manipulation/data/l_10/pusht_train_T_pose_length_41.pkl', 'rb')
    data_i = pickle.load(file_i)
    file_i.close()

    score_i = data_i['score']
    score_i = 1-score_i
    #score_i = np.swapaxes(score_i,0,1)
    if plot==True:
        plt.contourf(X[7:25],Y[7:25],var_np.T[7:25,7:25],levels=10)
        plt.colorbar()
        plt.title('Risk average across 10 values of length')
        plt.scatter(256,256,s=200,color='red')
        plt.xlabel('X',fontsize=15)
        plt.ylabel('Y',fontsize=15)
        #plt.show()
        #plt.savefig('contour_risk_2_seed.png')

        
        plt.figure()
        X_ = np.linspace(100,500,10)
        Y_ = np.linspace(100,500,10)
        plt.contourf(X_,Y_,score_i,levels=10)
        plt.colorbar()
        plt.title('Risk average across 10 seeds for l=4')
        plt.xlabel('X',fontsize=15)
        plt.ylabel('Y',fontsize=15)
        
        plt.figure()
        d = np.ma.array(score_np[:,5,:], mask=var_np <0.3)
        plt.contourf(X[7:25],Y[7:25],risk_np[7:25,7:25].T,levels=10)
        
        plt.colorbar()
        plt.scatter(256,256,s=200,color='red')
        plt.title('Risk for l=3.5',fontsize=20)

        plt.figure()
        #plt.contourf(X,Y,d,levels=10)
        plt.contourf(X,Y,score_np[:,5,:],levels=10)
        plt.colorbar()
        plt.title('Risk for l=4.0',fontsize=20)

        plt.figure()
        plt.contourf(X[7:25],Y[7:25],score_np[7:25,7,7:25],levels=10)
        plt.colorbar()
        plt.title('Risk for l=4.2',fontsize=20)
        plt.show()
        #plt.show()
            #plt.savefig('contour_risk_35_2.png')
            #breakpoint()
        
    return seed1,seed2,score_1,score_2, fail

def gpr_score():
    seed1,seed2, score_1,score_2,fail = dataset()
    #score_true = np.array([predict_score(seed,5) for seed in seed1[0:20]]) # Actual risk of 'Z' values for 'X' that is sensitive to change in model parameter
    #score_estimate = np.array([predict_score(seed,4) for seed in seed1[0:20]]) # predict risk of 'Z' values for 'X' that is sensitive to change in model parameter
    #breakpoint()
    
    score_true1 = np.array(score_1)[:,-1]
    score_estimate1 = np.array(score_1)[:,5]
    score_true2 = np.array(score_2)[:,-1]
    score_estimate2 = np.array(score_2)[:,5]
    data = {'fail':fail,'seed1':seed1,'score_estimate1':score_estimate1,'score_true1':score_true1,'seed2':seed2,'score_estimate2':score_estimate2,'score_true2':score_true2}
    filename='data/gpr/score_estimate.pkl'
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)
    file.close()

if __name__ == "__main__":
    gpr_score()
    #dataset()