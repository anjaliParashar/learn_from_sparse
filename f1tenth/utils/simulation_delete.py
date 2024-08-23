import sys
sys.path.append('/home/anjali/work/SQP/nla_falsifier/LQR')
import torch
import numpy as np
from lqr_bicycle import LQR
from dynamics import Toy
import cubic_spline
from ref_curve import Ref_Curve_2,ref_update,ref_to_spline,calc_speed_profile
import matplotlib.pyplot as plt
from derivatives_nn import get_f,get_open_loop_control,get_control,gradient_update,gradient_reference,gradient_state
from torch.func import jacrev
import numpy
import time
from f1tenth.utils.inference_delete import get_reference_vae
import ipdb
nominal_params = {
        #"yaw_ref": torch.tensor(1.0).float(),
        "v": torch.tensor(10.0/3.6).float(),
        #"omega_ref": torch.tensor(0.0).float(),
        "x_ref": torch.tensor(0.0).float(),
        "y_ref": torch.tensor(0.0).float()
    }
device='cuda'


def do_simulation_curv(theta):
    lqr_control = LQR()
    x_,y_,x_torch,y_torch = get_reference_vae(theta)
    cx, cy, cyaw, ck, s = cubic_spline.calc_spline_course(x_, y_, ds=0.1)
    """
    cx = cx_[5:-20]
    cy = cy_[5:-20]
    cyaw = cyaw_[5:-20]
    ck = ck_[5:-20]
    """
    goal = torch.tensor([x_[-1], y_[-1]])
    #print('Goal:',goal)
    #x_,y_,cx, cy, cyaw, ck, s,goal = ref_to_spline([a[0,0].detach().cpu(),a[1,0].detach().cpu()],nominal_params,0.1,10)   
    cx_tensor = torch.tensor(cx).to(device)
    cy_tensor = torch.tensor(cy).to(device)
    cyaw_tensor = torch.tensor(cyaw).to(device)
    target_speed = 10.0/3.6
    speed_profile = calc_speed_profile(cyaw,target_speed= 2)#10.0 / 3.6)
    state = Toy(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)
    print(cx[0],cy[0],cyaw[0])
    T = 10 #max simulation time
    goal_dis = 0.3
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
    
    g_ref = torch.zeros((4,4)).to(device)
    g_state = torch.zeros((4,4)).to(device)
    g_ref_list = []
    g_state_list  = []
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
        if torch.hypot(dx, dy) <= goal_dis:
            print("Goal")
            break
        
        x.append(state.x.detach())
        y.append(state.y.detach())
        yaw.append(state.yaw.detach())
        v.append(state.v.detach())
        t.append(time)   
        steering.append(dl.detach())
        acceleration.append(ai.detach())
        ### Construct gradient ODE
        if True:
            v_ref = torch.tensor(4).to(device)
            K =K.to(device)
            delta_t = torch.tensor(dt).to(device)
            tstep= torch.tensor(time).to(device)

            control_past = torch.tensor([dl,ai]).reshape((2,1)).to(device)
            state_vector = torch.tensor([state.x,state.y,state.yaw,state.v]).reshape((4,1)).to(device)
            state_ref_p = torch.vstack((cx_tensor[target_indx[-1]],cy_tensor[target_indx[-1]],cyaw_tensor[target_indx[-1]],v_ref)).to(device)
            control_o = get_open_loop_control(theta,tstep,delta_t).reshape((2,1))
            
            #state_ref = get_state_ref(state_ref_p,a,v_ref,tstep,delta_t).to(device)

            control_past.requires_grad=True
            state_vector.requires_grad = True
            state_ref_p.requires_grad = True
            df_dx = jacrev(get_f,argnums=0)(state_vector,control_past).reshape((4,4))
            df_du = jacrev(get_f,argnums=1)(state_vector,control_past).reshape((4,2))

            du_dx = jacrev(get_control,argnums=0)(state_vector,state_ref_p,K,control_past).reshape((2,4))
            du_dxref = jacrev(get_control,argnums=1)(state_vector,state_ref_p,K,control_past).reshape((2,4))

            df_dxref = jacrev(get_f,argnums=0)(state_ref_p,control_o).reshape((4,4))
            df_du_o = jacrev(get_f,argnums=1)(state_ref_p,control_o).reshape((4,2))
            
            du_dtheta = jacrev(get_open_loop_control,argnums=0)(theta,tstep,delta_t).reshape((2,4))

            if j==0:
                du_dx = torch.zeros((2,4)).to(device)
                du_dxref = torch.zeros((2,4)).to(device)

            g_ref = gradient_reference(g_ref, df_dxref,df_du_o,du_dtheta,dt)
            g_state = gradient_state(g_state, df_dx,df_du,du_dx,du_dxref,dt)
            
            
            g_state_list.append(g_state.detach().cpu())
            g_ref_list.append(g_ref.detach().cpu())
        j+=1
    x = x[-100:]
    y = y[-100:]
    
    #grad_x_a = torch.tensor(numpy.asarray(grad_a_list)).reshape((len(x),4,2)).to(device)
    #grad_xref_a = torch.tensor(numpy.asarray(grad_xref_a_list)).reshape((len(x),4,2)).to(device) # DO similar operations for g_state, g_ref
    x = numpy.asarray(x)
    y = numpy.asarray(y)
    print(x.shape)
    cx = numpy.asarray(x_torch.detach().cpu())
    cy = numpy.asarray(y_torch.detach().cpu())
    acceleration = numpy.asarray(acceleration)
    steering = numpy.asarray(steering)
    if True:
        g_state_list = torch.tensor(numpy.asarray(g_state_list)).to(device)
        g_ref_list = torch.tensor(numpy.asarray(g_ref_list)).to(device)
        if (x.shape[0]<100 and x.shape[0]!=0):
            d=x.shape[0]
            x = numpy.concatenate((x,numpy.repeat(x[-1],100-d).reshape((100-d,1))),0)
            y = numpy.concatenate((y,numpy.repeat(y[-1],100-d).reshape((100-d,1))),0)
            g_state_list = torch.cat((g_state_list,torch.zeros((101-d,4,4)).to(device)),0)
            g_ref_list = torch.cat((g_ref_list,torch.zeros((101-d,4,4)).to(device)),0)
            #g_state_list[-1].repeat(100-d)
            print(x.shape)
        elif x.shape[0]==0:
            print('flag',x)
            d=x.shape[0]
            x = numpy.repeat(0.0,100-d).reshape((100-d,1))
            y = numpy.repeat(0.0,100-d).reshape((100-d,1))
            g_state_list = torch.zeros((101-d,4,4)).to(device)
            g_ref_list = torch.zeros((101-d,4,4)).to(device)
            #g_state_list[-1].repeat(100-d)
            print(x.shape)
        
        assert x.shape[0] == 100, "x should be size 100'"
        assert g_state_list.shape[0] == 101
        #print(g_state_list.shape)
        assert g_ref_list.shape[0] == 101
    #print(g_ref_list.shape)
    #return x,y,cx,cy,g_state_list,g_ref_list
    return x,y,cx,cy, v, speed_profile,acceleration,steering ##Uncomment this line for analysis and plotting for specific value of theta
#theta = torch.ones((1,4)).to(device)
#x, y,cx, cy, v, speed_profile,acc_np,steering = do_simulation_curv(theta)
#x,y,cx,cy,g_state_list,g_ref_list=do_simulation_curv(theta)
#theta = torch.tensor([-0.8724,  0.8997,  2.0188,  2.3984]).reshape((1,4)).to(device)
def generate_track_bounds(track, theta,width=0.5):
    bounds_low = torch.zeros((2, track.shape[1]))
    bounds_upp = torch.zeros((2, track.shape[1]))

    for idx in range(track.shape[1]):
        x = track[0, idx]
        y = track[1, idx]
        th = torch.tensor(theta[idx])
        bounds_upp[0, idx] = 0 * torch.cos(th) - width * torch.sin(th) + x  # X
        bounds_upp[1, idx] = 0 * torch.sin(th) + width * torch.cos(th) + y  # Y

        bounds_low[0, idx] = 0 * torch.cos(th) - (-width) * torch.sin(th) + x  # X
        bounds_low[1, idx] = 0 * torch.sin(th) + (-width) * torch.cos(th) + y  # Y

    return bounds_low, bounds_upp
#theta = torch.tensor([-49.9702,  50.0058, -49.9713, -50.0485]).reshape((1,4)).to(device)
theta = torch.tensor([0.141277,0.60114745,0.0383763,0.648825]).reshape((1,4)).to(device) #Failure case
breakpoint()
x,y,cx,cy, v, speed_profile,acc_np,steering = do_simulation_curv(theta)
x_,y_,x_torch,y_torch = get_reference_vae(theta)
cx, cy, cyaw, ck, s = cubic_spline.calc_spline_course(x_, y_, ds=0.01)
track = numpy.array([cx,cy])
bounds_low, bounds_upp = generate_track_bounds(track,cyaw)
print(bounds_low.shape)
#plt.plot(x_torch.detach().cpu(),y_torch.detach().cpu(),linewidth=2)
plt.plot(cx,cy,linewidth=2)
plt.plot(bounds_low[0,:],bounds_low[1,:])
plt.plot(bounds_upp[0,:],bounds_upp[1,:])
plt.show()
plt.savefig('racelines.png')

#x_ = list(torch.tensor([0.0, 6.0, 12.5, 10.0]))#, 17.5, 20.0, 25.0]))
#y_ = list(torch.tensor([0.0, -3.0, -5.0, 6.5]))#, 3.0, 0.0, 0.0]))
#theta = torch.tensor([1.00141277,1.00114745,0.0383763,0.99648825]).reshape((1,4)).to(device) #Failure case

#theta = torch.tensor([0.141277,0.60114745,0.0383763,0.648825]).reshape((1,4)).to(device) #Failure case

#theta = torch.tensor([0.15,0.15,0.1,0.1]).reshape((1,4)).to(device) #0.9,0.0,0.0,0.1
#x_ = list(torch.linspace(-5,5,100))
#y_ = list(torch.zeros((100,)))
#x_ = list(torch.tensor([0.0,0.0,0.0,0.0,0.0,0.0,-0.5,-1.0,-1.0,-1.0]))#,0.0,0.0,0.0,0.0,0.0,0.0]))
#y_ = list(torch.tensor([-5.0,-4.0,-3.5,-3.0,-2.5,-2.0, 0.0, 2.0,2.5,3.0]))#,2.5,3.0,3.5,4.0,4.5,5.0]))
#print(len(x_),len(y_))
#x_,y_,x_torch,y_torch = get_reference_vae(theta)
#cx, cy, cyaw, ck, s = cubic_spline.calc_spline_course(y_, x_, ds=0.1)



if False:
    x,y,cx,cy, v, speed_profile,acc_np,steering = do_simulation_curv(theta)

    import pickle
    file_ = open('ego_trajectory_failure_1.pkl','wb')
    data = {'X':cx,'Y':cy}
    pickle.dump(data, file_)
    file_.close()

    plt.plot(cx[5:],cy[5:],label='c')
    plt.plot(x,y)
    plt.legend()
    plt.show()
if False:
    fig, axs = plt.subplots(2,1)
    axs[0].plot(x,y,label='actual',linewidth=2)
    axs[0].plot(cx,cy,'-.',label='reference',linewidth=2)
    #axs[0, 0].set_title('Trajectory',fontsize=15)
    axs[0].set_xlabel('x (m)',fontsize=15)
    axs[0].set_ylabel('y (m)',fontsize=15)
    axs[0].legend(fontsize=15)

    """
    axs[0, 1].plot(v,label='actual',linewidth=2)
    axs[0, 1].plot(speed_profile[0:len(v)],'-.',label='reference',linewidth=2)
    #axs[0, 1].set_title('Speed',fontsize=15)
    axs[0,1].set_ylabel('Speed (m/s)',fontsize=15)
    #axs[0,1].set_xlabel('Time',fontsize=15)
    axs[0,1].legend(fontsize=10)

    axs[1, 0].plot(acc_np,linewidth=2)
    #axs[1, 0].set_title('Acceleration')
    axs[1,0].axhline(2,0,10,color='black',ls='--')
    #axs[1,0].axhline(-2,0,10,color='black',ls='--')
    #axs[1,0].set_xlabel('Time',fontsize=15)
    axs[1,0].set_ylabel('Acceleration $(m/s^2)$',fontsize=15)
    """

    axs[1].plot(steering*57.3,linewidth=2)
    #axs[1, 1].set_title('Acceleration')
    axs[1].axhline(25,0,10,color='black',ls='--')
    axs[1].axhline(-25,0,10,color='black',ls='--')
    #axs[1,1].set_xlabel('Time',fontsize=15)
    axs[1].set_ylabel('Steering angle (deg)',fontsize=15)
    plt.show()
    plt.plot(x,y)
    plt.show()
    #plt.savefig('delete.png')
#plt.plot(x,y,label='actual',linewidth=3)
#plt.scatter(0,0,color='red')
#plt.plot(cx,cy,'-.',label='reference',linewidth=2)
#plt.title('a = 24.5',fontsize=20)
#plt.xticks([-3,-2,-1,0,1,2],fontsize=15)
#plt.yticks([0,2,4,6,8],fontsize=15)
#plt.xlabel('x (m)',fontsize=20)
#plt.ylabel('y (m)',fontsize=20)
#plt.legend(fontsize=20)
#plt.savefig('LQR_test.png')
"""
#a_range = [8.4072,24.5,30.5,40.5]
a_range = [8.4072,24.5,30.5,40.5,50,60,70,80]
lat_acc_list = numpy.zeros((501,8))
a_list = numpy.zeros((501,8))
delta_list =numpy.zeros((501,8))
curve = numpy.zeros((501,2,8))
curve_ref = numpy.zeros((501,2,8))
i =0
for a in a_range:
    lat_acc,x, y, cx, cy, v, speed_profile,grad_a_list,grad_xref_a_list,ctrl_vec = do_simulation_curv(a)
    curve[:,0,i] = x[:,0]
    curve[:,1,i] = y[:,0]
    curve_ref[:,0,i] = cx
    curve_ref[:,1,i] = cy
    delta_list[:,i] = ctrl_vec[:,0]
    a_list[:,i] = ctrl_vec[:,1]
    lat_acc_list[:,i] = lat_acc
    i+=1

plt.plot(x,y,label='actual',linewidth=3)
plt.scatter(0,0,color='red')
plt.plot(cx,cy,'-.',label='reference',linewidth=2)
plt.title('a = 24.5',fontsize=20)
plt.xticks([-3,-2,-1,0,1,2],fontsize=15)
plt.yticks([0,2,4,6,8],fontsize=15)
plt.xlabel('x (m)',fontsize=20)
plt.ylabel('y (m)',fontsize=20)
plt.legend(fontsize=20)
"""
"""
#x, y, yaw, v, x_ref,y_ref,yaw_ref,v_ref = do_simulation_ref()
a_range=[torch.tensor(4).reshape((1,)),torch.tensor(40).reshape((1,)),torch.tensor(400).reshape((1,))]
dx_da = []
u = []
x_list = []
y_list = []
for a in a_range:
    t, x, y, cx, cy, v, speed_profile,grad_a_list,grad_xref_a_list,ctrl_vec = do_simulation_curv(a[0])
    dx_da.append(grad_a_list.detach().cpu())
    x_list.append(x)
    y_list.append(y)
    u.append(ctrl_vec)
plt.figure()
plt.plot(x,y,label='actual')
plt.plot(cx,cy,'-.',label='reference')
plt.legend()

plt.figure()
plt.plot(v,label='speed')
plt.plot(speed_profile,label='reference velocity')
plt.legend()
"""