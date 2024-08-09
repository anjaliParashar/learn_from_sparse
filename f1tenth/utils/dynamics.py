import matplotlib.pyplot as plt
import torch

#Use this to compute df/dx and df/du
class STCar():
    def __init__(self,x=torch.tensor(0.0), y=torch.tensor(0.0), delta_a = torch.tensor(0.0),yaw=torch.tensor(0.0), v=torch.tensor(0.0),yaw_dot = torch.tensor(0.0),beta = torch.tensor(0.0)):
        #States
        self.x = x
        self.y = y
        self.delta_a = delta_a
        self.v = v
        self.yaw = yaw
        self.yaw_dot = yaw_dot
        self.beta = beta

        # test params
        self.mu = torch.tensor(1.0489)
        self.C_Sf = torch.tensor(21.92/1.0489)
        self.C_Sr = torch.tensor(21.92/1.0489)
        self.lf = torch.tensor(0.3048*3.793293)
        self.lr = torch.tensor(0.3048*4.667707)
        self.h = torch.tensor(0.3048*2.01355)
        self.m = torch.tensor(4.4482216152605/0.3048*74.91452)
        self.I = torch.tensor(4.4482216152605*0.3048*1321.416)
        self.g = torch.tensor(9.81)
        #steering constraints
        self.s_min = -1.066  #minimum steering angle [rad]
        self.s_max = 1.066  #maximum steering angle [rad]
        self.sv_min = -0.4  #minimum steering velocity [rad/s]
        self.sv_max = 0.4  #maximum steering velocity [rad/s]

        #longitudinal constraints
        self.v_min = -13.6  #minimum velocity [m/s]
        self.v_max = 50.8  #minimum velocity [m/s]
        self.v_switch = 7.319  #switching velocity [m/s]
        self.max_a = torch.tensor(6)  #maximum absolute acceleration [m/s^2]
        self.max_steer = torch.deg2rad(torch.tensor(30.0))
        
    def f_kin(self,state, control):
        """
        x_dot = f(x,u(x,x_ref))
        """
        mu = self.mu
        C_Sf = self.C_Sf 
        C_Sr = self.C_Sr 
        lf = self.lf 
        lr = self.lr 
        h = self.h 
        m = self.m 
        I = self.I 
        g = self.g

        #unpack state variables
        """
        delta_a = state[2,:]
        V = state[3,:]
        psi = state[4,:]
        X = state[0,:]
        Y = state[1,:]
        """
        #unpack state variables
        delta_a = state.delta_a#state[2,:]
        V = state.v#state[3,:]
        psi = state.yaw#state[4,:]
        X = state.x#state[0,:]
        Y = state.y#state[1,:]

        delta = control[0,:]
        a = control[1,:]
        lwb = 1#lf + lr
        #print(lwb)
        f = torch.vstack((torch.multiply(V,torch.cos(psi)),torch.multiply(V,torch.sin(psi)),delta,a,torch.multiply((V/lwb),torch.tan(delta_a)),torch.tensor(0.0),torch.tensor(0.0)))#delta/lwb*torch.tan(delta_a) +V/(lwb*torch.cos(delta_a)**2)*speed,torch.tensor(0.0)))
        #f = torch.vstack((torch.multiply(v,torch.cos(torch.tensor(yaw))), torch.multiply(v,torch.sin(torch.tensor(yaw))), torch.multiply((v / L),torch.tan(delta)),a))
        """
        f = torch.zeros_like(state)
        f[0,:] = torch.multiply(V,torch.cos(psi))
        f[1,:] = torch.multiply(V,torch.sin(psi))
        f[2,:] = delta
        f[3,:] = speed
        f[4,:] = V/lwb*torch.tan(delta_a)
        f[5,:] = delta/lwb*torch.tan(delta_a) +V/(lwb*torch.cos(delta_a)**2)*speed
        f[6,:] = 0
        """
        return f
    
    def f_dyn(self,state, control):
        """
        x_dot = f(x,u(x,x_ref))
        """
        mu = self.mu
        C_Sf = self.C_Sf 
        C_Sr = self.C_Sr 
        lf = self.lf 
        lr = self.lr 
        h = self.h 
        m = self.m 
        I = self.I 
        g = self.g

        #unpack state variables
        delta_a = state.delta_a#state[2,:]
        beta = state.beta#state[6,:]
        V = state.v#state[3,:]
        yaw_rate = state.yaw_dot#state[5,:]
        psi = state.yaw#state[4,:]
        X = state.x#state[0,:]
        Y = state.y#state[1,:]

        delta = control[0,:]
        speed = control[1,:]

        f0 = torch.multiply(V,torch.cos(beta+psi))
        f1 = torch.multiply(V,torch.sin(beta+psi))
        f2 = delta
        f3 = speed
        f4 = yaw_rate
        #f[5,:] = -mu*m/(state[3,:]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-speed*h) + lr**2*C_Sr*(g*lf + speed*h))*state[5,:] +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + speed*h) - lf*C_Sf*(g*lr - speed*h))*state[6,:] +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - speed*h)*state[2,:]
        #f[6,:] = (mu/(state[3,:]**2*(lr+lf))*(C_Sr*(g*lf + speed*h)*lr - C_Sf*(g*lr - speed*h)*lf)-1)*state[5,:] -mu/(state[3,:]*(lr+lf))*(C_Sr*(g*lf + speed*h) + C_Sf*(g*lr-speed*h))*state[6,:] +mu/(state[3,:]*(lr+lf))*(C_Sf*(g*lr-speed*h))*state[2,:]
        f5 = -mu*m/(V*I*(lr+lf))*(lf**2*C_Sf*(g*lr-speed*h) + lr**2*C_Sr*(g*lf + speed*h))*yaw_rate +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + speed*h) - lf*C_Sf*(g*lr - speed*h))*beta +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - speed*h)*delta_a
        f6 = (mu/(V**2*(lr+lf))*(C_Sr*(g*lf + speed*h)*lr - C_Sf*(g*lr - speed*h)*lf)-1)*yaw_rate -mu/(V*(lr+lf))*(C_Sr*(g*lf + speed*h) + C_Sf*(g*lr-speed*h))*beta +mu/(V*(lr+lf))*(C_Sf*(g*lr-speed*h))*delta_a
        f = torch.vstack((f0,f1,f2,f3,f4,f5,f6))
        """
        f = torch.zeros_like(state)
        f[0,:] = torch.multiply(V,torch.cos(beta+psi))
        f[1,:] = torch.multiply(V,torch.sin(beta+psi))
        f[2,:] = delta
        f[3,:] = speed
        f[4,:] = yaw_rate
        #f[5,:] = -mu*m/(state[3,:]*I*(lr+lf))*(lf**2*C_Sf*(g*lr-speed*h) + lr**2*C_Sr*(g*lf + speed*h))*state[5,:] +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + speed*h) - lf*C_Sf*(g*lr - speed*h))*state[6,:] +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - speed*h)*state[2,:]
        #f[6,:] = (mu/(state[3,:]**2*(lr+lf))*(C_Sr*(g*lf + speed*h)*lr - C_Sf*(g*lr - speed*h)*lf)-1)*state[5,:] -mu/(state[3,:]*(lr+lf))*(C_Sr*(g*lf + speed*h) + C_Sf*(g*lr-speed*h))*state[6,:] +mu/(state[3,:]*(lr+lf))*(C_Sf*(g*lr-speed*h))*state[2,:]
        f[5,:] = -mu*m/(V*I*(lr+lf))*(lf**2*C_Sf*(g*lr-speed*h) + lr**2*C_Sr*(g*lf + speed*h))*yaw_rate +mu*m/(I*(lr+lf))*(lr*C_Sr*(g*lf + speed*h) - lf*C_Sf*(g*lr - speed*h))*beta +mu*m/(I*(lr+lf))*lf*C_Sf*(g*lr - speed*h)*delta_a
        f[6,:] = (mu/(V**2*(lr+lf))*(C_Sr*(g*lf + speed*h)*lr - C_Sf*(g*lr - speed*h)*lf)-1)*yaw_rate -mu/(V*(lr+lf))*(C_Sr*(g*lf + speed*h) + C_Sf*(g*lr-speed*h))*beta +mu/(V*(lr+lf))*(C_Sf*(g*lr-speed*h))*delta_a
        """
        return f
    
    def update(self, state, dt, a, delta):
        max_steer = self.max_steer
        max_a = self.max_a
        if delta >= max_steer:
            delta = max_steer
        if delta <= - max_steer:
            delta = - max_steer
        
        if a >= max_a:
            a = max_a
        if a <= - max_a:
            a = - max_a
        
        control = torch.vstack((delta,a))
        f_dyn = self.f_dyn(state,control)
        f_kin = self.f_kin(state,control)
        #if state[3,:]>0.5:
        if state.v>6:
            f_ = f_dyn
            #print('Kinematic')
            #state = state + dt*f_dyn
        else:
            #state = state + dt*f_kin
            f_ = f_kin
            #print('Dynamic')
        state.x = state.x + dt*f_[0,:]
        state.y = state.y + dt*f_[1,:]
        state.delta_a = state.delta_a + dt*f_[2,:]
        state.v = state.v + dt*f_[3,:]
        state.yaw = state.yaw + dt*f_[4,:]
        state.yaw_dot = state.yaw_dot + dt*f_[5,:]
        state.beta = state.beta + dt*f_[6,:]
        return state

    def get_state_dim(self):
        return (7,)

    def get_action_dim(self):
        return (2,)

class Toy():
    def __init__(self,x=torch.tensor(0.0), y=torch.tensor(0.0), yaw=torch.tensor(0.0), v=torch.tensor(0.0),L = torch.tensor(0.29)):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L
        self.max_steer = torch.deg2rad(torch.tensor(25.0))
        self.dt = torch.tensor(0.1)
        self.max_a = torch.tensor(1)

    def get_state_dim(self):
        return (4,)

    def get_action_dim(self):
        return (2,)
    
    def f(self,state, control):
        """
        x_dot = f(x,u(x,x_ref))

        Error matrices
        e = state[0,:]
        dot_e = state[1,:]
        th_e = state[2,:]
        dot_th_e = state[3,:]
        delta_v = state[4,:]
        dt = 0.1 
        f[0,:] = e + dot_e*dt
        f[1,:] = torch.multiply(v,th_e)
        f[2,:] = th_e + dt*dot_th_e
        f[3,:] = torch.multiply((v/L),control[0,:])
        f[4,:] = delta_v + dt*control[1,:]
        """
        x = state.x #[0,:]
        y = state.y #[1,:]
        yaw = state.yaw #[2,:]
        v = state.v #[3,:]

        delta = control[0,:]
        a = control[1,:]
        L = self.L

        #f = torch.zeros_like(self.get_state_dim())
        f = torch.vstack((torch.multiply(v,torch.cos(torch.tensor(yaw))), torch.multiply(v,torch.sin(torch.tensor(yaw))), torch.multiply((v / L),torch.tan(delta)),a))

        return f


    def update(self, state, dt, a, delta):
        #car = Toy()
        max_steer = self.max_steer
        max_a = self.max_a
        if delta >= max_steer:
            delta = max_steer
        if delta <= - max_steer:
            delta = - max_steer
        
        if a >= max_a:
            a = max_a
        if a <= - max_a:
            a = - max_a
        control = torch.vstack((delta,a))
        f_ = self.f(state,control)
        state.x = state.x + dt*f_[0,:]
        state.y = state.y + dt*f_[1,:]
        state.yaw = state.yaw + dt*f_[2,:]
        state.v = state.v + dt*f_[3,:]

        return state
    
    
"""
state =  STCar() #torch.tensor([0,0,0,0.6,0.1,0,0]).reshape((7,1))
T = 1
dt = 0.01
ai = torch.tensor(-0.7*9.81)
dl = torch.deg2rad(torch.tensor(10.0))
x = []
y = []
for i in range(1000):
    state = state.update(state,dt,ai, dl)
    x.append(state.x)
    y.append(state.y)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.savefig('STCar_test.png')
"""