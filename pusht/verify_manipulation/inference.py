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
import copy
import sys

class inference_pusht():
    def __init__(self,seed_num, mass_in, friction_in, length_in):
        self.device = torch.device('cuda')

        # parameters
        self.pred_horizon = 16
        self.obs_horizon = 2
        self.action_horizon = 16
        self.action_dim = 2
        self.obs_dim = 5
        #|o|o|                             observations: 2
        #| |a|a|a|a|a|a|a|a|               actions executed: 8
        #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

        #create dataset
        self.dataset_path=  "pusht_cchi_v7_replay.zarr.zip"
        self.dataset = self.create_dataset(dataset_path=self.dataset_path)
        # save training data statistics (min, max) for each dim
        self.stats = self.dataset.stats

        # create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=256,
            num_workers=1,
            shuffle=True,
            # accelerate cpu-gpu transfer
            pin_memory=True,
            # don't kill worker process afte each epoch
            persistent_workers=True
        )

        #create noise prediction network
        self.noise_pred_net = self.noise_pred_net()

        #optimizer
        self.optimizer = torch.optim.AdamW(params=self.noise_pred_net.parameters(),lr=1e-4, weight_decay=1e-6)

        self.num_diffusion_iters = 100  

        #noise_scheduler 
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule='squaredcos_cap_v2',
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type='epsilon'
        )

        # limit enviornment interaction to 200 steps before termination
        self.max_steps = 200
        self.friction=friction_in
        self.mass = mass_in
        self.length=length_in
        self.env,self.obs = self.create_environment(seed_num)
        self.init_obs = self.obs.copy()
    
    def create_dataset(self,dataset_path):
        # create dataset from file
        dataset = PushTStateDataset(
            dataset_path=dataset_path,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon
        )
        return dataset
    
    def noise_pred_net(self):       
        # Standard ADAM optimizer
        # Note that EMA parametesr are not optimized
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim*self.obs_horizon
        )
        noise_pred_net.load_state_dict(torch.load('models/push_T_diffusion_model'))
        noise_pred_net = noise_pred_net.cuda()
        return noise_pred_net

    def normalize_data(self,data, stats):
        # nomalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self,ndata, stats):
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data
    
    def create_environment(self,seed_num):
        env = PushTEnv(mass=self.mass,friction=self.friction,length=self.length)
        # use a seed >200 to avoid initial states seen in the training dataset
        env.seed(seed_num)
        # get first observation
        obs = env.reset()
        print('initial state',obs)
        return env, obs

    def generate_dist(self):
        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque(
            [self.obs] * self.obs_horizon, maxlen=self.obs_horizon)
        # save visualization and rewards
        rewards = list()
        done = False
        step_idx = 0
        alpha = 0.1

        #def cost_grad(nmean):

        with tqdm(total=self.max_steps, desc="Eval PushTStateEnv") as pbar:
            while not done:
                B = 1
                # stack the last obs_horizon (2) number of observations
                obs_seq = np.stack(obs_deque)
                # normalize observation
                nobs = self.normalize_data(obs_seq, stats=self.stats['obs'])
                # device transfer
                nobs = torch.from_numpy(nobs).to(self.device, dtype=torch.float32)

                # infer action
                with torch.no_grad():
                    # reshape observation to (B,obs_horizon*obs_dim)
                    obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (B, self.pred_horizon, self.action_dim), device=self.device)
                    naction = noisy_action

                    # init scheduler
                    self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

                    for k in self.noise_scheduler.timesteps:
                        # predict noise
                        noise_pred = self.noise_pred_net(
                            sample=naction,
                            timestep=k,
                            global_cond=obs_cond
                        )

                        # inverse diffusion step (remove noise)
                        naction = self.noise_scheduler.step(
                            model_output=noise_pred,
                            timestep=k,
                            sample=naction
                        ).prev_sample
                        var = self.noise_scheduler._get_variance(k)
                        nmean = naction - var*torch.ones(naction.shape).to(self.device)
                        naction = nmean + (var)*torch.ones(nmean.shape).to(self.device)
                        
                        if rewards==[]:
                            reward = torch.tensor(0)
                        else:
                            reward = rewards.pop()
                        
                        naction = naction.detach().to('cpu').numpy()
                        action = torch.tensor(self.unnormalize_data(naction, stats=self.stats['action'])).to(self.device)
                        action = action + 0.1*var*(torch.exp(reward-0.1))*torch.ones(nmean.shape).to(self.device)
                        action = action.detach().to('cpu').numpy()
                        naction = torch.tensor(self.normalize_data(action,stats = self.stats['action'])).to(self.device)
                        # (B, pred_horizon, action_dim)
                        
                        #- 0.000000001*var*torch.abs(reward-1)*torch.ones(nmean.shape).to(device)
                        #naction = naction.to(device) + 10*var*(torch.abs(obs[2]-256) + torch.abs(obs[3]-256))*torch.ones(nmean.shape).to(device)
                        #naction = naction + 0.001*var*(naction-torch.tensor([256,256,np.pi/4]))
                        
                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                # (B, pred_horizon, action_dim)
                naction = naction[0]
                action_pred = self.unnormalize_data(naction, stats=self.stats['action'])

                # only take action_horizon number of actions
                start = self.obs_horizon - 1
                end = start + self.action_horizon
                action = action_pred[start:end,:]
                # (action_horizon, action_dim)

                # execute action_horizon number of steps
                # without replanning
                for i in range(len(action)):
                    # stepping env
                    obs, coverage, reward, done, info = self.env.step(action[i])
                    # save observations
                    info = self.env._get_info()
                    shape = info['block_pose']
                    
                    obs_deque.append(obs)
                    # and reward/vis
                    rewards.append(reward)
                    #reward.backward()

                    # update progress bar
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    if step_idx > self.max_steps:
                        done = True
                    if done:
                        break
        # print out the maximum target coverage
        print('Score: ', max(rewards))
        Score= max(rewards)
        return Score

