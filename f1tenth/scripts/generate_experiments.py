import sys
sys.path.append("/home/anjali/learn_from_sparse")
import numpy as np
import pickle
from f1tenth.utils.ref_curve import lane_change_trajectory
from f1tenth.utils.simulation import do_simulation
import matplotlib.pyplot as plt
file = open("/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/experiment_initial_1.pkl",'rb')
data = pickle.load(file)
file.close()

for i in range(len(data['means'])):
    print(data["Z_exp"][i],data['means'][i], i)
    x_ref,y_ref,_ = lane_change_trajectory(v=1.5,lane_width=data['Z_exp'][i][0])
    #yaw, risk_,x,y,cx,cy = do_simulation(v=data['Z_exp'][i][1],lane_width=data['Z_exp'][i][0])
    yaw, risk_, mean_dist, max_dist, final, x,y,cx,cy = do_simulation(v=data['Z_exp'][i][1],lane_width=data['Z_exp'][i][0])
    file_i = open("/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/initial_"+str(i)+".pkl",'wb')
    data_i = {'X':y_ref,"Y":x_ref,"V_ref":data['Z_exp'][i][1],'lw':data['Z_exp'][i][0],'X_sim':x,'Y_sim':y}
    print(data['Z_exp'][i][1])
    pickle.dump(data_i,file_i)
    file_i.close()
    plt.figure()
    plt.scatter(x_ref,y_ref,label='Reference')
    plt.plot(x,y,label='Actual')
    plt.legend()
    plt.show()

