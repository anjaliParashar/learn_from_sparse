import sys
sys.path.append('/home/anjali/learn_from_sparse')
import torch
import numpy as np
import f1tenth.utils.cubic_spline as cubic_spline
import matplotlib.pyplot as plt
class Ref_Curve_Spline():
    def __init__(self,ax=[0.0, 6.0, 12.5, 10.0, 17.5, 20.0, 25.0],ay=[0.0, -3.0, -5.0, 6.5, 3.0, 0.0, 0.0]):
        self.ax = ax
        self.ay = ay
        self.goal = torch.tensor([ax[-1], ay[-1]])
        cx, cy, cyaw, ck, s = cubic_spline.calc_spline_course(
        ax, ay, ds=0.1)
        self.x = torch.tensor(cx)
        self.y = torch.tensor(cy)
        self.yaw = torch.tensor(cyaw)
        self.k = torch.tensor(ck)
    
class Ref_Curve():
    def __init__(self,pt):
        self.omega = pt['omega_ref']
        self.yaw= pt['yaw_ref']
        self.v= pt['v']
        self.x = pt['x_ref']
        self.y = pt['y_ref']

class Ref_Curve_2():
    def __init__(self,pt):
        self.x = pt['x_ref']
        self.y = pt['y_ref']
        self.v= pt['v']
    
def ref_update(a,state,delta_t,tstep):
        omega = a * torch.sin(torch.tensor(tstep * delta_t))
        yaw = state.yaw + delta_t * state.omega
        x = state.x + delta_t  * state.v * torch.cos(state.yaw)
        y = state.y + delta_t  * state.v * torch.sin(state.yaw)
        v = state.v
        new_params = {
        "yaw_ref":yaw,
        "omega_ref": omega,
        "x_ref": x,
        "y_ref": y,
        "v": v
        }
        next_state = Ref_Curve(new_params)
        return next_state

def ref_update_2(a,state,delta_t,tstep):
        a1 = a[0]
        a2 = a[1]
        #a3 = a[2]
        t = torch.tensor(tstep * delta_t)
        #x = state.x + delta_t  * 1*state.v*a1*(t+torch.sin(1*a1*t))
        #y = state.y + delta_t  * 1*state.v*a2*(t**2 + torch.cos(1*a2*t))/5
        x = state.x + delta_t  * 1*state.v*(torch.sin(1*a1*t)+3*torch.sin(1*a2*t))
        y = state.v + delta_t  * 1*state.v*(torch.cos(1*a1*t)+3*torch.cos(1*a2*t))
        v = state.v
        new_params = {
        "x_ref": x,
        "y_ref": y,
        "v": v
        }
        next_state = Ref_Curve_2(new_params)
        return next_state

def lane_change_trajectory(v, lane_width,t_total=3):
    """
    Generate the trajectory for a lane change maneuver.

    Parameters:
    v (float): Velocity of the vehicle (m/s).
    t_total (float): Total time for lane change (seconds).
    lane_width (float): Width of the lane (meters).
    curvature_factor (float): Factor controlling the curvature. 
                              Higher values result in a sharper turn.

    Returns:
    x (numpy array): x-coordinates of the trajectory.
    y (numpy array): y-coordinates of the trajectory.
    t (numpy array): Time array for the trajectory.
    """
    
    # Time array
    t = np.linspace(0, t_total, num=400)
    curvature_factor=2.5  
    # x-coordinates assuming constant velocity
    x = v * t
    # y-coordinates with curvature control
    y = (lane_width / 2) * (1 - np.cos(curvature_factor * np.pi * t / t_total))

    return x, y, t


def ref_to_spline(a,pt,delta_t,T):
    curve = Ref_Curve_2(pt)
    x_ = []
    y_ = []
    time=0
    while T>=time:
        curve_next = ref_update_2(a,curve,delta_t,time)
        time=time+delta_t
        x_.append(curve_next.x.detach().cpu())
        y_.append(curve_next.y.detach().cpu())
        curve = curve_next
    cx, cy, cyaw, ck, s = cubic_spline.calc_spline_course(x_, y_, ds=0.01)
    goal = torch.tensor([x_[-1], y_[-1]])
    return x_,y_,cx,cy,cyaw,ck,s,goal


def calc_speed_profile(cyaw, target_speed):
    speed_profile = [target_speed] * len(cyaw)
    direction = 1.0

    # Set stop point
    for i in range(len(cyaw) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        switch = torch.pi / 4.0 <= dyaw < torch.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0
        #return speed_profile
    # speed down
    # if i>=len(cyaw)-20:
    #     for i in range(20):
    #         speed_profile[-i] = target_speed / (50 - i)
    #         #if speed_profile[-i] <= 1.0 / 3.6:
    #         #    speed_profile[-i] = 1.0 / 3.6
        return speed_profile
