import matplotlib.pyplot as plt
import torch
import numpy as np

class Toy():
    def __init__(self,noise1=None,noise2=None,x=np.array(0.0), y=np.array(0.0), yaw=np.array(0.0), v=np.array(0.0),L = np.array(0.29)):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.L = L
        self.max_steer = np.deg2rad(np.array(50.0)) #Max steering angle
        self.max_a = np.array(2) #Max acceleration

        self.dt = np.array(0.1)
        self.noise1 = noise1
        self.noise2 = noise2

    def get_state_dim(self):
        return (4,)

    def get_action_dim(self):
        return (2,)
    
    def f(self,state, control):
        x = state.x #[0,:]
        y = state.y #[1,:]
        yaw = state.yaw #[2,:]
        v = state.v #[3,:]

        delta = control[0,:]
        a = control[1,:]
        L = self.L

        #f = torch.zeros_like(self.get_state_dim())
        f = np.vstack((np.multiply(v,np.cos(np.array(yaw))), np.multiply(v,np.sin(np.array(yaw))), np.multiply((v / L),np.tan(delta)),a))

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
        control = np.vstack((delta,a))
        f_ = self.f(state,control)
        if self.noise1 is not None:
            f_+= np.array(self.noise1.sample()).reshape(-1,1)
        state.x = state.x + dt*f_[0,:] #+ np.array(self.noise2.sample()[0])
        state.y = state.y + dt*f_[1,:] #+ np.array(self.noise2.sample()[1])
        state.yaw = state.yaw + dt*f_[2,:] #+ np.array(self.noise2.sample()[2])
        state.v = state.v + dt*f_[3,:] #+ np.array(self.noise2.sample()[3])
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