import torch
from torch.func import jacrev
import sys
sys.path.append('/home/anjali/work/raceline_generation')
from network import AutoEncoder
device = 'cuda'
model = AutoEncoder().to(device)
checkpoint = torch.load('/home/anjali/work/results/1024_500k_n1_500/vae_1024_500k_500_n1.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def get_f(state, control):
        x = state[0,:]
        y = state[1,:]
        yaw = state[2,:]
        v = state[3,:]

        delta = control[0,:]
        a = control[1,:]
        L = 0.5
        #f = torch.zeros_like(self.get_state_dim())
        f = torch.vstack((torch.multiply(v,torch.cos(yaw)), torch.multiply(v,torch.sin(yaw)), torch.multiply((v / L),torch.tan(delta)),a))
        return f

def get_open_loop_control(theta,tstep,delta_t):
    decoded = model.decoder(theta).to(device)
    batch_size =1
    phi_max = torch.pi/20
    a_max = 30
    phi = torch.cat((decoded[:,0:5].reshape((5,1)),decoded[:,5:10].reshape((5,1))),-1)*torch.pi/2
    a = torch.cat(((2*decoded[:,10:15]-1).reshape((5,1))*phi_max,(2*decoded[:,15:20]-1).reshape((5,1))*a_max),-1)
    p = decoded[:,-10:].reshape((5,2))*200

    t = tstep * delta_t
    control = torch.sum(a*torch.cos(((2*torch.pi*t/p)-phi)),0)
    return control

def get_control(state,state_ref,K,control_past):
    x = state[0,:]
    y = state[1,:]
    yaw = state[2,:]
    v = state[3,:]
    L = 0.5

    x_ref = state_ref[0,:]
    y_ref = state_ref[1,:]
    yaw_ref = state_ref[2,:]
    v_ref = state_ref[3,:]

    delta = control_past[0,:]
    a = control_past[1,:]

    e = torch.sqrt(torch.linalg.norm(state[0:2,:]-state_ref[0:2,:]))
    x_ = torch.zeros((5, 1)).to(device)
    x_[0, 0] = e
    x_[1, 0] = (v*(torch.cos(yaw)+torch.sin(yaw)) - v_ref*(torch.cos(yaw_ref)+torch.sin(yaw_ref)))/e
    x_[2, 0] =  yaw-yaw_ref
    x_[3, 0] = torch.multiply((v / L),torch.tan(delta))-yaw_ref
    x_[4, 0] = v - v_ref
    return -K@x_

def gradient_update(dl_dx,dl_dxref,g_ref,g_state):
    gradient_ = (dl_dx.float()@g_state.float() + dl_dxref.float())@g_ref.float()
    return gradient_
def gradient_reference(g_ref, df_dxref,df_du_o,du_dtheta,dt):
    g_next_ref = g_ref.float() + dt*(df_dxref.float()@g_ref.float() + df_du_o.float()@du_dtheta.float())
    return g_next_ref

def gradient_state(g_state, df_dx,df_du,du_dx,du_dxref,dt):
    g_next_state = g_state.float() + dt*(((df_dx.float()+ (df_du.float()@du_dx.float()))@g_state.float()) + (df_du.float()@du_dxref.float()))
    return g_next_state

"""
device='cuda'
a = torch.tensor(4.0,requires_grad=True).to(device)
state = torch.ones((4,1),requires_grad=True).to(device)
control_past = torch.ones((2,1),requires_grad=True).to(device)
state_ref_p = torch.ones((4,1),requires_grad=True).to(device)
state_ref = get_state_ref(state_ref_p,a,v=10/3.6,tstep=1,delta_t=0.01).to(device)
v = torch.tensor(10/3.6)
K = torch.ones((2,5))
delta_t = torch.tensor(0.01)
tstep= torch.tensor(1)
df_dx = jacrev(f,argnums=0)(state,control_past).reshape((4,4))
df_du = jacrev(f,argnums=1)(state,control_past).reshape((4,2))

du_dx = jacrev(lqr,argnums=0)(state,state_ref,K,control_past).reshape((2,4))
du_dxref = jacrev(lqr,argnums=1)(state,state_ref,K,control_past).reshape((2,4))
dxref_da = jacrev(get_state_ref,argnums=1)(state_ref_p,a,v,tstep,delta_t)


control_past = torch.vstack((dl,ai),requires_grad=True).to(device).reshape((2,1))
state_vector = torch.vstack((state.x,state.y,state.yaw,state.v),requires_grad=True).to(device).reshape((4,1))
a = torch.tensor(0.4,requires_grad=True).to(device)

state_ref_p = torch.vstack(cx(target_indx[-2],cy(target_indx[-2],cyaw(target_indx[-2],torch.tensor(10/3.6),requires_grad=True).to(device)
state_ref = get_state_ref(state_ref_p,a,v=10/3.6,tstep=1,delta_t=0.01).to(device)
v = torch.tensor(10/3.6)
K = torch.tensor(K)
delta_t = torch.tensor(dt)
tstep= torch.tensor(time)
df_dx = jacrev(f,argnums=0)(state,control_past).reshape((4,4))
df_du = jacrev(f,argnums=1)(state,control_past).reshape((4,2))

du_dx = jacrev(lqr,argnums=0)(state,state_ref,K,control_past).reshape((2,4))
du_dxref = jacrev(lqr,argnums=1)(state,state_ref,K,control_past).reshape((2,4))
dxref_da = jacrev(get_state_ref,argnums=1)(state_ref_p,a,v,tstep,delta_t)
"""