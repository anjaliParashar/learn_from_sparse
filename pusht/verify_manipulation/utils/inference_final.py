#@markdown ### **Inference**
import sys
sys.path.append('/home/anjali/learn_from_sparse')

import numpy as np
import torch
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
import time

# env import
import os
import gdown
from pusht.verify_manipulation.utils.env_T import PushTEnv
from pusht.verify_manipulation.train.data_gen import PushTStateDataset
from pusht.verify_manipulation.train.network import ConditionalUnet1D
from torch.distributions.multivariate_normal import MultivariateNormal
import pickle

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
#noise_pred_net.load_state_dict(torch.load('/home/anjalip/push_T/verify_manipulation/models/push_T_diffusion_model'))

noise_pred_net = noise_pred_net.cuda()

ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())

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

def generate_dist(x0,y0,seed,sigma_1,sigma_2):
    # limit enviornment interaction to 200 steps before termination
    max_steps = 200
    env = PushTEnv(x0=x0,y0=y0,mass=1,friction=1,length=4)
    #env.seed(x0=220,y0=380)

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
    seed=seed
    torch.manual_seed(seed=seed)
    noisy_action = torch.randn(
                    (1, pred_horizon, action_dim), device=device)
    
    start_time=time.time()
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
                #print(noisy_action[0,1,0])
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

            # execute action_horizon number of steps without replanning

            #fixing the seed should also fix the seed for these sampling
            noise_1 = MultivariateNormal(torch.zeros(2), sigma_1*torch.eye(2)).sample()
            noise_2 = MultivariateNormal(torch.zeros(2), sigma_2*torch.eye(2)).sample()
            for i in range(len(action)):
                # stepping env
                obs, reward, done, obs_noise = env.step_noise(action[i],noise_1, noise_2)
                # save observations
                info = env._get_info()
                shape = info['block_pose']
                
                obs_deque.append(obs_noise)
                # and reward/vis
                rewards.append(reward)
                #print('reward',reward)

                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                #print(step_idx)
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if reward>=0.9:
                    done=True
                if step_idx > max_steps:
                    done = True
                if done:
                    break
            i_idx+=1

    # print out the maximum target coverage
    end_time=time.time()
    print('Score: ', max(rewards),"Time:",end_time-start_time)
    score = max(rewards)
    return score

def get_trajectory(x0,y0,seed=None):
    # limit enviornment interaction to 200 steps before termination
    max_steps = 200
    env = PushTEnv(x0=x0,y0=y0,mass=1,friction=1,length=4)
    #env.seed(x0=220,y0=380)

    # get first observation
    obs = env.reset()
    print(obs)
    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    rewards_maintain = list()
    done = False
    step_idx = 0
    i_idx=0
    #def cost_grad(nmean):
    if seed is None:
        seed=10000 #Fix the seed
    else:
        seed=seed
    torch.manual_seed(seed=seed)
    noisy_action = torch.randn(
                    (1, pred_horizon, action_dim), device=device)
    
    #noisy_action = noisy_action_list[0]
    start_time=time.time()
    obs_list = []
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
                #print(noisy_action[0,1,0])
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

            # execute action_horizon number of steps without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, info = env.step(action[i])
                obs_list.append(normalize_data(obs,stats=stats['obs']))
                # save observations
                info = env._get_info()
                shape = info['block_pose']
                
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                rewards_maintain.append(reward.detach().cpu())
                #print('reward',reward)

                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                #print(step_idx)
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if reward>0.8:
                    done=True
                if done:
                    break
            i_idx+=1

    # print out the maximum target coverage
    end_time=time.time()
    print('Score: ', max(rewards),"Time:",end_time-start_time)
    score = max(rewards)
    return score,obs_list, rewards_maintain


def generate_dist_multiple(x0,y0,l,n_seeds):
    max_steps = 200
    env = PushTEnv(x0=x0,y0=y0,mass=1,friction=1,length=l)
    obs = env.reset()


    obs_deque = np.zeros((n_seeds,2,5))
    rewards_list = [[] for i in range(n_seeds)]
    obs_list = [[] for i in range(n_seeds)]
    #rewards = torch.zeros((naction.shape[0],))
    # save visualization and rewards
    done = False
    step_idx = 0
    start_time = time.time()
        #def cost_grad(nmean):
    seed=1000
    torch.manual_seed(seed=seed)
    noisy_action = torch.randn(
                    (n_seeds, pred_horizon, action_dim), device=device)
    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            # stack the last obs_horizon (2) number of observations
            obs_seq = obs_deque
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.reshape((n_seeds,obs_horizon*obs_dim))
                #breakpoint()

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (noisy_action.shape[0], pred_horizon, action_dim), device=device)
                
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:

                    noise_pred = noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond= obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                    
                    # initialize action from Guassian noise
                    noisy_action = torch.randn(
                        (noisy_action.shape[0], pred_horizon, action_dim), device=device)
                    
                # unnormalize action
                naction = naction.detach().to('cpu').numpy()
                action_pred = unnormalize_data(naction.squeeze(), stats=stats['action'])
                start = obs_horizon - 1
                end = start + action_horizon
                action = action_pred[:,start:end,:] #(10,16,2)
                
                #obs_list = np.zeros((noisy_action.shape[0],len(action),self.obs_dim))

                #fixing the seed should also fix the seed for these sampling
                noise_1 = MultivariateNormal(torch.zeros(2), 1*torch.eye(2)).sample()
                noise_2 = MultivariateNormal(torch.zeros(2), 1*torch.eye(2)).sample()
                for i in range(action.shape[1]):
                    for j in range(naction.shape[0]):
                        obs, reward, done, info = env.step(action[j,i,:],noise_1, noise_2)
                        obs_list[j].append(obs)
                        rewards_list[j].append(reward.detach())
                    if step_idx > max_steps:
                        done = True
                    if done:
                        break
                    step_idx+=1
                    pbar.update(1)
                    pbar.set_postfix()
                obs_deque = np.stack(obs_list)[:,-2:,:] 
        #analysis
        end_time = time.time()
        score = np.mean(np.max(np.array(rewards_list),axis=1)) #average score
        print('total_time',end_time-start_time)
        #print('Score_list',rewards_list)
        #breakpoint()
        return score
