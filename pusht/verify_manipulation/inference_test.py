#@markdown ### **Inference**
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os
from env_T import PushTEnv
from data_gen import PushTStateDataset
from network import ConditionalUnet1D

device = torch.device('cuda')

# parameters
pred_horizon = 16
obs_horizon = 2
action_horizon = 16
action_dim = 2
obs_dim = 5
#|o|o|                             observations: 2
#| |a|a|a|a|a|a|a|a|               actions executed: 8
#|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

#Dataset
dataset_path = "pusht_cchi_v7_replay.zarr.zip"
if not os.path.isfile(dataset_path):
    id = "1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t"
    gdown.download(id=id, output=dataset_path, quiet=False)

# create dataset from file
dataset = PushTStateDataset(
    dataset_path=dataset_path,
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)
# save training data statistics (min, max) for each dim
stats = dataset.stats

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=256,
    num_workers=1,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)
noise_pred_net.load_state_dict(torch.load('/home/anjali/push_T_diffusion_model'))

noise_pred_net = noise_pred_net.cuda()

optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)
def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

# limit enviornment interaction to 200 steps before termination
max_steps = 200
env = PushTEnv(x0=500,y0=100,mass=0.1,friction=1,length=4)
# use a seed >200 to avoid initial states seen in the training dataset
env.seed(x0=100,y0=100)

# get first observation
obs = env.reset()
print(obs)
# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards
imgs = [env.render(mode='rgb_array')]
rewards = list()
done = False
step_idx = 0
alpha = 0.1

#def cost_grad(nmean):
seed=42
torch.manual_seed(seed=seed)
with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)
        # normalize observation
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        # device transfer
        nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

        # infer action
        with torch.no_grad():
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = noise_pred_net(
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
                var = noise_scheduler._get_variance(k)
                nmean = naction - var*torch.ones(naction.shape).to(device)
                naction = nmean + (var)*torch.ones(nmean.shape).to(device)
                
                if rewards==[]:
                    reward = torch.tensor(0)
                else:
                    reward = rewards.pop()
                
                naction = naction.detach().to('cpu').numpy()
                action = torch.tensor(unnormalize_data(naction, stats=stats['action'])).to(device)
                action = action + 0.1*var*(torch.exp(reward-0.1))*torch.ones(nmean.shape).to(device)
                action = action.detach().to('cpu').numpy()
                naction = torch.tensor(normalize_data(action,stats = stats['action'])).to(device)
                # (B, pred_horizon, action_dim)
                
                 #- 0.000000001*var*torch.abs(reward-1)*torch.ones(nmean.shape).to(device)
                #naction = naction.to(device) + 10*var*(torch.abs(obs[2]-256) + torch.abs(obs[3]-256))*torch.ones(nmean.shape).to(device)
                #naction = naction + 0.001*var*(naction-torch.tensor([256,256,np.pi/4]))
                
        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        action_pred = unnormalize_data(naction, stats=stats['action'])

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        # (action_horizon, action_dim)

        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, coverage, reward, done, info = env.step(action[i])
            # save observations
            info = env._get_info()
            shape = info['block_pose']
            
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            #reward.backward()
            #print('grad',coverage/0.95)
            print('reward',reward)
            imgs.append(env.render(mode='rgb_array'))

            # update progress bar
            step_idx += 1
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break

# print out the maximum target coverage
print('Score: ', max(rewards))

# visualize
from IPython.display import Video
vwrite('vis_1_01_5.gif', imgs)
#Video('vis__1_01_5.mp4', embed=True, width=1024*4, height=1024*4)