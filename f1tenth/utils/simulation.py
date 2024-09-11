import sys
sys.path.append('/home/anjali/learn_from_sparse')
import scipy.spatial
import torch
import numpy as np
from f1tenth.utils.lqr_bicycle import LQR
from f1tenth.utils.dynamics import Toy
from f1tenth.utils import cubic_spline
from f1tenth.utils.ref_curve import calc_speed_profile, lane_change_trajectory
import matplotlib.pyplot as plt
import numpy
import ipdb
import scipy
import pickle
from torch.distributions.multivariate_normal import MultivariateNormal

def do_simulation_curv(v,lane_width,sigma1,sigma2,seed=10000):
    lqr_control = LQR()
    x_ref,y_ref,_= lane_change_trajectory(v=1.5,lane_width=lane_width)
    cx, cy, cyaw, ck, s = cubic_spline.calc_spline_course(x_ref, y_ref, ds=0.1)
    goal = np.array([x_ref[-1], y_ref[-1]])

    target_speed = v #10/3.6 originally
    speed_profile = calc_speed_profile(cyaw,target_speed)

    #fixing the seed should also fix the seed for these sampling
    seed=seed
    torch.manual_seed(seed=seed)
    noise1 = MultivariateNormal(torch.zeros(4), sigma1*torch.eye(4))
    noise2 = MultivariateNormal(torch.zeros(4), sigma2*torch.eye(4))

    #Initialize state with noise in dynamics and state estimation, I.C., and reference trajectory data
    state = Toy(noise1=noise1,noise2=noise2,x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0) 
    state_noise = state #+ np.array(noise2.sample())
    T = 40 #max simulation time
    goal_dis = 0.1
    stop_speed = 0.05
    time = 0.0
    x = []
    y = []
    yaw = []
    v = []
    t = []
    dt = 0.1
    e, e_th = 0.0, 0.0
    target_indx = []
    
    acceleration = []
    steering = []
    j=0
    while T >= time:
        dl, target_ind, e, e_th, ai,K = lqr_control.lqr_speed_steering_control_curv(
            state_noise,cx,cy,cyaw,ck , e, e_th, speed_profile)
        target_indx.append(target_ind)
        state = state.update(state,dt,ai, dl)

        noise = noise2.sample()
        state_noise.x = state.x + np.array(noise[0])
        state_noise.y = state.y + np.array(noise[1])
        state_noise.yaw = state.yaw + np.array(noise[2])
        state_noise.v = state.v + np.array(noise[3])

        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if np.hypot(dx, dy) <= goal_dis:
            print("Goal")
            ai=0
            dl=0
            state = state.update(state,dt,ai, dl)
            break
        
        if state.x-x_ref[-1] > 0.1:
            break
        
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)   
        steering.append(dl)
        acceleration.append(ai)

    #x = x[-100:]
    #y = y[-100:]
    
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    yaw = numpy.asarray(yaw)
    acceleration = numpy.asarray(acceleration)
    steering = numpy.asarray(steering)
    return x,y,x_ref,y_ref, yaw,v, speed_profile,acceleration,steering ##Uncomment this line for analysis and plotting for specific value of theta
    #return x,y,cx,cy

def do_simulation(v,lane_width):
    lqr_control = LQR()
    x_ref,y_ref,_= lane_change_trajectory(1.5,lane_width)
    cx, cy, cyaw, ck, s = cubic_spline.calc_spline_course(x_ref, y_ref, ds=0.1)
    goal = np.array([x_ref[-1], y_ref[-1]])

    target_speed = v #10/3.6 originally
    speed_profile = calc_speed_profile(cyaw,target_speed)

    #Initialize state with noise in dynamics and state estimation, I.C., and reference trajectory data
    state = Toy(noise1=None,noise2=None,x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0) 
    T = 40 #max simulation time
    goal_dis = 0.1
    stop_speed = 0.05
    time = 0.0
    x = []
    y = []
    yaw = []
    v = []
    t = []
    dt = 0.1
    e, e_th = 0.0, 0.0
    target_indx = []
    
    acceleration = []
    steering = []
    j=0
    while T >= time:
        
        dl, target_ind, e, e_th, ai,K = lqr_control.lqr_speed_steering_control_curv(
            state,cx,cy,cyaw,ck , e, e_th, speed_profile)
        target_indx.append(target_ind)
        state = state.update(state,dt,ai, dl)

        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if np.hypot(dx, dy) <= goal_dis:
            print("Goal")
            ai=0
            dl=0
            state = state.update(state,dt,ai, dl)
            break
        
        if state.x-x_ref[-1] > 0.1:
            break
        
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)   
        steering.append(dl)
        acceleration.append(ai)

    #x = x[-100:]
    #y = y[-100:]
    
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    yaw = numpy.asarray(yaw)
    acceleration = numpy.asarray(acceleration)
    steering = numpy.asarray(steering)
    risk_, mean_dist, max_dist, final = risk(x,y,cx,cy,lane_width)
    return steering, risk_, mean_dist, max_dist, final, x,y,cx,cy 

def do_simulation_filepath(filename):
    lqr_control = LQR()
    file=open(filename,'rb')
    data = pickle.load(file)
    file.close()
    x_ref = np.array(data['X'])*0.6-4.0
    y_ref = np.array(data['Y'])*0.6 +0.5
    x_ref = list(x_ref)
    y_ref = list(y_ref)
    cx, cy, cyaw, ck, s = cubic_spline.calc_spline_course(x_ref, y_ref, ds=0.1)
    goal = np.array([x_ref[-1], y_ref[-1]])

    target_speed = 10/3.6# originally
    speed_profile = calc_speed_profile(cyaw,target_speed)

    #Initialize state with noise in dynamics and state estimation, I.C., and reference trajectory data
    state = Toy(noise1=None,noise2=None,x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0) 
    T = 40 #max simulation time
    goal_dis = 0.1
    stop_speed = 0.05
    time = 0.0
    x = []
    y = []
    yaw = []
    v = []
    t = []
    dt = 0.1
    e, e_th = 0.0, 0.0
    target_indx = []
    
    acceleration = []
    steering = []
    j=0
    while T >= time:
        dl, target_ind, e, e_th, ai,K = lqr_control.lqr_speed_steering_control_curv(
            state,cx,cy,cyaw,ck , e, e_th, speed_profile)
        target_indx.append(target_ind)
        state = state.update(state,dt,ai, dl)

        if abs(state.v) <= stop_speed:
            target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if np.hypot(dx, dy) <= goal_dis:
            print("Goal")
            ai=0
            dl=0
            state = state.update(state,dt,ai, dl)
            break
        
        if state.x-x_ref[-1] > 0.1:
            break
        
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)   
        steering.append(dl)
        acceleration.append(ai)

    #x = x[-100:]
    #y = y[-100:]
    
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    yaw = numpy.asarray(yaw)
    acceleration = numpy.asarray(acceleration)
    steering = numpy.asarray(steering)
    risk_ = risk(x,y,cx,cy)
    return x,y
 
def find_closest(x,y,x_ckpt):
    dist_actual = np.abs(x-x_ckpt)
    x_actual = np.argmin(dist_actual)
    
    if np.linalg.norm(x[x_actual]-x_ckpt)>0.01  and x_actual+1<len(x):
        y_ref = (y[x_actual]+y[x_actual+1])/2
        x_ref = (x[x_actual]+x[x_actual+1])/2
    else:
        y_ref= y[x_actual]
        x_ref = x[x_actual]    
    return x_ref,y_ref

def mean_dist_(x,y,cx,cy,lane_width):
    x_ckpt = np.linspace(0,5,100)
    mean_dist = np.zeros((100,))
    for i,y_ in zip(range(100),x_ckpt):
        x_a,y_a = find_closest(x,y,y_)
        x_r,y_r = find_closest(cx,cy,y_)
        mean_dist[i]= np.linalg.norm(np.array([x_a-x_r,y_a-y_r]))
    return np.mean(mean_dist)

def risk(x,y,cx,cy,lane_width):
    X = np.array([x,y]).squeeze()
    CX = np.array([cx,cy])#[:,0:X.shape[1]]
    dist_mat = scipy.spatial.distance.cdist(X.T,CX.T)
    min_dist = np.min(dist_mat,axis=1)
    mean_dist = mean_dist_(x,y,cx,cy,lane_width)
    #mean_dist = min_dist.mean()
    max_dist = min_dist.max()
    final = np.min(np.linalg.norm(X-CX[:,-1][:,None],axis=0))
    risk_ = 20*mean_dist + max_dist+10*final
    print("Mean:",mean_dist,"Max:",max_dist,"Final",final)
    return risk_, mean_dist, max_dist, final#np.linalg.norm(X-CX)

def generate_dist(sigma1,sigma2,speed,curvature):
    x,y,cx,cy,_,_,_,_,_ = do_simulation_curv(speed,curvature,sigma1,sigma2)
    risk_, mean_risk,max_risk, final_risk = risk(x,y,cx,cy,curvature)
    print("Risk",risk_,"Speed",speed)
    return risk_, mean_risk,max_risk, final_risk, x,y,cx,cy

def visualize(cx,cy,x,y,speed_profile):
    #Visualize
    plt.plot(x,y,label='Actual',color='red',linewidth=2)
    plt.plot(cx,cy,'--',label='Reference')
    plt.ylim([-3,6])
    plt.xlim([0,5])
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(v,label='Actual')
    plt.plot(speed_profile,label='Reference')
    plt.legend()
    plt.title('Speed Profile')
    plt.show()

    fig,axs=plt.subplots(1,2)
    axs[0].plot(acceleration)
    axs[0].set_title('Acceleration')
    axs[0].axhline(5,0,10,color='black',ls='--')
    axs[0].axhline(-5,0,10,color='black',ls='--')

    axs[1].plot(steering*180/np.pi)
    axs[1].axhline(30,0,10,color='black',ls='--')
    axs[1].set_title('Steeering angle')
    plt.show()

if __name__=="__main__":
    #Very sensitive to sigma_2. Accepetable range for sigma_1=[0.0001,1.0], sigma_2=[0.0001,0.001]
    #Speed =[1,5], lane_width = [0.5,5.5], Risk threshold 15 is acceptable
    v_ref= 1.0
    lane_width = 1.48
    x,y,cx,cy,yaw,v,speed_profile,acceleration,steering = do_simulation_curv(v_ref,lane_width,sigma1=0.00001,sigma2=0.00001) 
    steering, risk_,x,y,cx,cy = do_simulation(v_ref,lane_width) 
    
    #print("Curv:",lane_width,"Risk",risk(x,y,cx,cy))
    print("Yaw variance:",np.var(yaw))
    visualize(cx,cy,x,y,speed_profile)
    plt.figure()
    plt.plot(yaw)
    plt.show()
    #breakpoint()
    #data={'X':cy,'Y':cx}
    #filename=open('/home/anjali/learn_from_sparse/f1tenth/data/gpr/experiments/experiment_trial_2.pkl','wb')
    #pickle.dump(data,filename)
    #filename.close()

