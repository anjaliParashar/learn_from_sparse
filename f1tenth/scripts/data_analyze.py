import pickle
import numpy as np
import matplotlib.pyplot as plt

seed_list = []
risk_list = []
for i in range(30):
    file_i=open('/home/anjali/learn_from_sparse/f1tenth/data/speed_30_sigma_5/f1tenth' +str(i)+'.pkl','rb')
    # load information from file
    data_i = pickle.load(file_i)
    file_i.close()
    risk_list.append(data_i['risk'])
    seed_list+=data_i['seed']

seed_np = np.array(seed_list)
risk_np = np.array(risk_list)

var_np1 = np.zeros((30,30))
var_np2 = np.zeros((30,30))
risk_1 = []
risk_2 = []
seed1 = []
seed2 = []
label_1 = []
label_2 = []
speed = np.linspace(0.5,4.5,30)
curv = np.linspace(0.5,4.5,30)

for j in range(risk_np.shape[1]): #Speed
    for i in range(risk_np.shape[0]): #Curvature
        var_sigma1 = np.max(risk_np[j,i,0:4,0])-np.min(risk_np[j,i,0:4,0])
        var_sigma2 = np.max(risk_np[j,i,0,0:4])-np.min(risk_np[j,i,0,0:4])
        var_np1[i,j] = var_sigma1
        var_np2[i,j] = var_sigma2
        if (risk_np[j,i,2:5,2:5]>20).all():#risk_np[j,i,0,0]>20: #(risk_np[j,i,:,:]>20).all():#(risk_np[j,i,0,0]>=18 and var_sigma1<10 and var_sigma2<10): #or (risk_np[j,i,0,0]>30): #Fail with certainity
            risk_2.append(risk_np[j,i,0,0].T)
            seed2.append([curv[i],speed[j]])
            label_2.append(1)
        else:
            risk_1.append(risk_np[j,i,0,0].T)
            seed1.append([curv[i],speed[j]])
            label_1.append(0)
#Visualize data classification for Flow-GMM            
seed1_np = np.array(seed1)
seed2_np = np.array(seed2)
plt.scatter(seed1_np[:,0],seed1_np[:,1],color='blue',label='seed1')
plt.scatter(seed2_np[:,0],seed2_np[:,1],color='orange',label='seed2')
plt.legend()
plt.show()
plot =False
if plot==True:
    plt.contourf(speed,curv,var_np1.T,levels=30)
    plt.colorbar()
    plt.title('Risk variance across 30 values of length for sigma_1')
    plt.xlabel('Speed',fontsize=15)
    plt.ylabel('Curvature',fontsize=15)
    plt.show()

    plt.contourf(speed,curv,var_np2.T,levels=30)
    plt.colorbar()
    plt.title('Risk variance across 30 values of length for sigma_2')
    plt.xlabel('Speed',fontsize=15)
    plt.ylabel('Curvature',fontsize=15)
    plt.show()
data_ =seed1+seed2
labels=label_1+label_2
print(len(seed2))
data={'data':np.array(data_),'labels':np.array(labels)}
file = open('/home/anjali/learn_from_sparse/f1tenth/data/gpr/sim/f1tenth_cluster_train.pkl','wb')
pickle.dump(data,file)
file.close()