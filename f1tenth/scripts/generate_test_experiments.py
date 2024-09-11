import sys
sys.path.append("/home/anjali/learn_from_sparse")
import numpy as np
import pickle
from f1tenth.utils.ref_curve import lane_change_trajectory
from f1tenth.utils.simulation import do_simulation
import matplotlib.pyplot as plt


i=2
x_ref,y_ref,_ = lane_change_trajectory(v=1.5,lane_width=1.2)
#yaw, risk_,x,y,cx,cy = do_simulation(v=data['Z_exp'][i][1],lane_width=data['Z_exp'][i][0])
yaw, risk_, mean_dist, max_dist, final, x,y,cx,cy = do_simulation(v=1.2,lane_width=1.2)
file_i = open("/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/test_"+str(i)+".pkl",'wb')
data_i = {'X':y_ref,"Y":x_ref,"V_ref":1.2,'lw':1.2,'X_sim':x,'Y_sim':y}

pickle.dump(data_i,file_i)
file_i.close()
plt.figure()
plt.scatter(x_ref,y_ref,label='Reference')
plt.plot(x,y,label='Actual')
plt.legend()
plt.show()

