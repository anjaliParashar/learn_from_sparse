# A script for testing inference based on diffusion policy. Adapted from https://github.com/real-stanford/diffusion_policy
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
import time
from torch.distributions.multivariate_normal import MultivariateNormal

# env import
import sys
sys.path.append('/home/anjali/learn_from_sparse')
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
from pusht.verify_manipulation.utils.env_T import PushTEnv
from pusht.verify_manipulation.train.data_gen import PushTStateDataset
from pusht.verify_manipulation.train.network import ConditionalUnet1D
import pickle

device = torch.device('cuda')
file = open('/home/anjali/learn_from_sparse/pusht/verify_manipulation/models/noise.pkl','rb')
data = pickle.load(file)
file.close()
noisy_action_list = data['noise']
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

#noise_pred_net.load_state_dict(torch.load('/home/anjalip/push_T/verify_manipulation/models/push_T_diffusion_model'))
noise_pred_net = noise_pred_net.cuda()

ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())

#ckpt_path = "/home/anjali/learn_from_sparse/pusht/verify_manipulation/models/pusht_state_100ep.ckpt"
ckpt_path = "/home/anjali/learn_from_sparse/pusht/verify_manipulation/models/test.ckpt"

state_dict = torch.load(ckpt_path, map_location='cuda')
ema_noise_pred_net = noise_pred_net
ema_noise_pred_net.load_state_dict(state_dict)

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
x0=100
y0=100
env = PushTEnv(x0=x0,y0=y0,mass=1,friction=1,length=4)
# use a seed >200 to avoid initial states seen in the training dataset

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
i_idx=0
#def cost_grad(nmean):
seed=1000
torch.manual_seed(seed=seed)
noisy_action = torch.randn(
                (1, pred_horizon, action_dim), device=device)

noisy_action = noisy_action_list[0]
start_time = time.time()
with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
    while not done:
        B = 1
        # stack the last obs_horizon (2) number of observations
        obs_seq = np.stack(obs_deque)
        #print(obs_deque,obs_seq)
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
                (noisy_action.shape[0], pred_horizon, action_dim), device=device)
            #noisy_action = noisy_action_list[i_idx]
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise

                noise_pred = ema_noise_pred_net(
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
                
                if rewards==[]:
                    reward = torch.tensor(0)
                else:
                    reward = rewards.pop()
            
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
        noise_1 = MultivariateNormal(torch.zeros(2), 1*torch.eye(2)).sample()
        noise_2 = MultivariateNormal(torch.zeros(2), 1*torch.eye(2)).sample()
        for i in range(len(action)):
            # stepping env
            obs, reward, done, info = env.step(action[i],noise_1,noise_2)
            # save observations
            info = env._get_info()
            shape = info['block_pose']
            
            obs_deque.append(obs)
            # and reward/vis
            rewards.append(reward)
            #print('reward',reward)

            imgs.append(env.render(mode='rgb_array'))

            # update progress bar
            step_idx += 1
            #print(step_idx)
            pbar.update(1)
            pbar.set_postfix(reward=reward)
            if step_idx > max_steps:
                done = True
            if done:
                break
        i_idx+=1
end_time = time.time()
print(end_time-start_time)
#### VISUALIZATION ####
import matplotlib.pyplot as plt
import matplotlib.animation as animation
output_path = "/home/anjali/learn_from_sparse/pusht/verify_manipulation/media/push_T_animation_noisy.mp4"
frames = [] # for storing the generated images
fig = plt.figure()
for img in imgs:
    frames.append([plt.imshow(img, animated=True)])
ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)

ani.save(output_path)
# plt.show()