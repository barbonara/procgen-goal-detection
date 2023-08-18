#%%
import os
import time
import csv
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from collections import deque

import gym
from gym3 import ToBaselinesVecEnv
from procgen import ProcgenEnv, ProcgenGym3Env
from procgen_tools import visualization, maze
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)



hard500 = {
        'algo': 'ppo',
        'n_envs': 256,
        'n_steps': 256,
        'epoch': 3,
        'mini_batch_per_epoch': 8,
        'mini_batch_size': 8192,
        'gamma': 0.999,
        'lmbda': 0.95,
        'learning_rate': 0.0005,
        'grad_clip_norm': 0.5,
        'eps_clip': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'normalize_adv': True,
        'normalize_rew': True,
        'use_gae': True,
        'architecture': 'impala',
        'recurrent': False
    }
#%%
debug_params = {
    'algo': 'ppo',
    'n_envs': 1,
    'n_steps': 256,
    'epoch': 3,
    'mini_batch_per_epoch': 8,
    'mini_batch_size': 8192,
    'gamma': 0.999,
    'lmbda': 0.95,
    'learning_rate': 0.0005,
    'grad_clip_norm': 0.5,
    'eps_clip': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.01,
    'normalize_adv': True,
    'normalize_rew': True,
    'use_gae': True,
    'architecture': 'impala',
    'recurrent': False,
}

hyperparameters = hard500

#%%
# Utils

def xavier_uniform_init(module, gain=1.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

def orthogonal_init(module, gain=nn.init.calculate_gain('relu')):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight.data, gain)
        nn.init.constant_(module.bias.data, 0)
    return module

def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def get_n_params(model):
    return str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3)) + ' M params'



#%%
class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, start_dim=1)
    
class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.ReLU()(x)
        out = self.conv1(out)
        out = nn.ReLU()(out)
        out = self.conv2(out)
        return out + x

class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImpalaBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

scale = 1
class ImpalaModel(nn.Module):
    def __init__(self,
                 in_channels,
                 **kwargs):
        super(ImpalaModel, self).__init__()
        self.block1 = ImpalaBlock(in_channels=in_channels, out_channels=16*scale)
        self.block2 = ImpalaBlock(in_channels=16*scale, out_channels=32*scale)
        self.block3 = ImpalaBlock(in_channels=32*scale, out_channels=32*scale)
        self.fc = nn.Linear(in_features=32*scale * 8 * 8, out_features=256)

        self.output_dim = 256
        self.apply(xavier_uniform_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = nn.ReLU()(x)
        x = Flatten()(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        return x
    


#%%
class CategoricalPolicy(nn.Module):
    def __init__(self, 
                 embedder,
                 recurrent,
                 action_size):
        """
        embedder: (torch.Tensor) model to extract the embedding for observation
        action_size: number of the categorical actions
        """ 
        super(CategoricalPolicy, self).__init__()
        self.embedder = embedder
        # small scale weight-initialization in policy enhances the stability        
        self.fc_policy = orthogonal_init(nn.Linear(self.embedder.output_dim, action_size), gain=0.01)
        self.fc_value = orthogonal_init(nn.Linear(self.embedder.output_dim, 1), gain=1.0)

        self.recurrent = recurrent
        if self.recurrent:
            self.gru = GRU(self.embedder.output_dim, self.embedder.output_dim)  # Not defined because model isn't recurrent

    def is_recurrent(self):
        return self.recurrent

    def forward(self, x, hx, masks):
        hidden = self.embedder(x)
        if self.recurrent:
            hidden, hx = self.gru(hidden, hx, masks)
        logits = self.fc_policy(hidden)
        log_probs = F.log_softmax(logits, dim=1)
        p = Categorical(logits=log_probs)
        v = self.fc_value(hidden).reshape(-1)
        return p, v, hx



#%%
class Logger(object):

    def __init__(self, n_envs, logdir, use_wandb=False):
        self.start_time = time.time()
        self.n_envs = n_envs
        self.logdir = logdir
        self.use_wandb = use_wandb

        # training
        self.episode_rewards = []
        for _ in range(n_envs):
            self.episode_rewards.append([])

        self.episode_timeout_buffer = deque(maxlen = 40)
        self.episode_len_buffer = deque(maxlen = 40)
        self.episode_reward_buffer = deque(maxlen = 40)

        # validation
        self.episode_rewards_v = []
        for _ in range(n_envs):
            self.episode_rewards_v.append([])

        self.episode_timeout_buffer_v = deque(maxlen = 40)
        self.episode_len_buffer_v = deque(maxlen = 40)
        self.episode_reward_buffer_v = deque(maxlen = 40)

        time_metrics = ["timesteps", "wall_time", "num_episodes"] # only collected once
        episode_metrics = ["max_episode_rewards", "mean_episode_rewards", "min_episode_rewards",
                           "max_episode_len", "mean_episode_len", "min_episode_len",
                           "mean_timeouts"] # collected for both train and val envs
        self.log = pd.DataFrame(columns = time_metrics + episode_metrics + \
                                    ["val_"+m for m in episode_metrics])

        self.timesteps = 0
        self.num_episodes = 0

    def feed(self, rew_batch, done_batch, rew_batch_v=None, done_batch_v=None):
        steps = rew_batch.shape[0]
        rew_batch = rew_batch.T
        done_batch = done_batch.T

        valid = rew_batch_v is not None and done_batch_v is not None
        if valid:
            rew_batch_v = rew_batch_v.T
            done_batch_v = done_batch_v.T

        for i in range(self.n_envs):
            for j in range(steps):
                self.episode_rewards[i].append(rew_batch[i][j])
                if valid:
                    self.episode_rewards_v[i].append(rew_batch_v[i][j])

                if done_batch[i][j]:
                    self.episode_timeout_buffer.append(1 if j == steps-1 else 0)
                    self.episode_len_buffer.append(len(self.episode_rewards[i]))
                    self.episode_reward_buffer.append(np.sum(self.episode_rewards[i]))
                    self.episode_rewards[i] = []
                    self.num_episodes += 1
                if valid and done_batch_v[i][j]:
                    self.episode_timeout_buffer_v.append(1 if j == steps-1 else 0)
                    self.episode_len_buffer_v.append(len(self.episode_rewards_v[i]))
                    self.episode_reward_buffer_v.append(np.sum(self.episode_rewards_v[i]))
                    self.episode_rewards_v[i] = []

        self.timesteps += (self.n_envs * steps)

    def dump(self):
        wall_time = time.time() - self.start_time
        episode_statistics = self._get_episode_statistics()
        episode_statistics_list = list(episode_statistics.values())
        log = [self.timesteps, wall_time, self.num_episodes] + episode_statistics_list
        self.log.loc[len(self.log)] = log

        with open(self.logdir + '/log-append.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(self.log.columns)
            writer.writerow(log)

        with open(self.logdir + '/episode_rewards.csv', 'a') as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(["episode_rewards"])  # write the header
            for reward in self.episode_reward_buffer:
                writer.writerow([reward])  # write the rewards


        print(self.log.loc[len(self.log)-1])

        if self.use_wandb:
            wandb.log({k: v for k, v in zip(self.log.columns, log)})

    def _get_episode_statistics(self):
        episode_statistics = {}
        episode_statistics['Rewards/max_episodes']  = np.max(self.episode_reward_buffer, initial=0)
        episode_statistics['Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer)
        episode_statistics['Rewards/min_episodes']  = np.min(self.episode_reward_buffer, initial=0)
        episode_statistics['Len/max_episodes']  = np.max(self.episode_len_buffer, initial=0)
        episode_statistics['Len/mean_episodes'] = np.mean(self.episode_len_buffer)
        episode_statistics['Len/min_episodes']  = np.min(self.episode_len_buffer, initial=0)
        episode_statistics['Len/mean_timeout'] = np.mean(self.episode_timeout_buffer)

        # valid
        episode_statistics['[Valid] Rewards/max_episodes'] = np.max(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Rewards/mean_episodes'] = np.mean(self.episode_reward_buffer_v)
        episode_statistics['[Valid] Rewards/min_episodes'] = np.min(self.episode_reward_buffer_v, initial=0)
        episode_statistics['[Valid] Len/max_episodes'] = np.max(self.episode_len_buffer_v, initial=0)
        episode_statistics['[Valid] Len/mean_episodes'] = np.mean(self.episode_len_buffer_v)
        episode_statistics['[Valid] Len/min_episodes'] = np.min(self.episode_len_buffer_v, initial=0)
        episode_statistics['[Valid] Len/mean_timeout'] = np.mean(self.episode_timeout_buffer_v)
        return episode_statistics



#%%
class Storage():

    def __init__(self, obs_shape, hidden_state_size, num_steps, num_envs, device):
        self.obs_shape = obs_shape
        self.hidden_state_size = hidden_state_size
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self):
        self.obs_batch = torch.zeros(self.num_steps+1, self.num_envs, *self.obs_shape)
        self.hidden_states_batch = torch.zeros(self.num_steps+1, self.num_envs, self.hidden_state_size)
        self.act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.rew_batch = torch.zeros(self.num_steps, self.num_envs)
        self.done_batch = torch.zeros(self.num_steps, self.num_envs)
        self.log_prob_act_batch = torch.zeros(self.num_steps, self.num_envs)
        self.value_batch = torch.zeros(self.num_steps+1, self.num_envs)
        self.return_batch = torch.zeros(self.num_steps, self.num_envs)
        self.adv_batch = torch.zeros(self.num_steps, self.num_envs)
        self.info_batch = deque(maxlen=self.num_steps)
        self.step = 0

    def store(self, obs, hidden_state, act, rew, done, info, log_prob_act, value):
        self.obs_batch[self.step] = torch.from_numpy(obs.copy())
        self.hidden_states_batch[self.step] = torch.from_numpy(hidden_state.copy())
        self.act_batch[self.step] = torch.from_numpy(act.copy())
        self.rew_batch[self.step] = torch.from_numpy(rew.copy())
        self.done_batch[self.step] = torch.from_numpy(done.copy())
        self.log_prob_act_batch[self.step] = torch.from_numpy(log_prob_act.copy())
        self.value_batch[self.step] = torch.from_numpy(value.copy())
        self.info_batch.append(info)

        self.step = (self.step + 1) % self.num_steps

    def store_last(self, last_obs, last_hidden_state, last_value):
        self.obs_batch[-1] = torch.from_numpy(last_obs.copy())
        self.hidden_states_batch[-1] = torch.from_numpy(last_hidden_state.copy())
        self.value_batch[-1] = torch.from_numpy(last_value.copy())

    def compute_estimates(self, gamma=0.99, lmbda=0.95, use_gae=True, normalize_adv=True):
        rew_batch = self.rew_batch
        if use_gae:
            A = 0
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]
                value = self.value_batch[i]
                next_value = self.value_batch[i+1]

                delta = (rew + gamma * next_value * (1 - done)) - value
                self.adv_batch[i] = A = gamma * lmbda * A * (1 - done) + delta
        else:
            G = self.value_batch[-1]
            for i in reversed(range(self.num_steps)):
                rew = rew_batch[i]
                done = self.done_batch[i]

                G = rew + gamma * G * (1 - done)
                self.return_batch[i] = G

        self.return_batch = self.adv_batch + self.value_batch[:-1]
        if normalize_adv:
            self.adv_batch = (self.adv_batch - torch.mean(self.adv_batch)) / (torch.std(self.adv_batch) + 1e-8)


    def fetch_train_generator(self, mini_batch_size=None, recurrent=False):
        batch_size = self.num_steps * self.num_envs
        if mini_batch_size is None:
            mini_batch_size = batch_size
        # If agent's policy is not recurrent, data could be sampled without considering the time-horizon
        if not recurrent:
            sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                                   mini_batch_size,
                                   drop_last=True)
            for indices in sampler:
                obs_batch = torch.FloatTensor(self.obs_batch[:-1]).reshape(-1, *self.obs_shape)[indices].to(self.device)
                hidden_state_batch = torch.FloatTensor(self.hidden_states_batch[:-1]).reshape(-1, self.hidden_state_size).to(self.device)
                act_batch = torch.FloatTensor(self.act_batch).reshape(-1)[indices].to(self.device)
                done_batch = torch.FloatTensor(self.done_batch).reshape(-1)[indices].to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch).reshape(-1)[indices].to(self.device)
                value_batch = torch.FloatTensor(self.value_batch[:-1]).reshape(-1)[indices].to(self.device)
                return_batch = torch.FloatTensor(self.return_batch).reshape(-1)[indices].to(self.device)
                adv_batch = torch.FloatTensor(self.adv_batch).reshape(-1)[indices].to(self.device)
                yield obs_batch, hidden_state_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch
        # If agent's policy is recurrent, data should be sampled along the time-horizon
        else:
            num_mini_batch_per_epoch = batch_size // mini_batch_size
            num_envs_per_batch = self.num_envs // num_mini_batch_per_epoch
            perm = torch.randperm(self.num_envs)
            for start_ind in range(0, self.num_envs, num_envs_per_batch):
                idxes = perm[start_ind:start_ind+num_envs_per_batch]
                obs_batch = torch.FloatTensor(self.obs_batch[:-1, idxes]).reshape(-1, *self.obs_shape).to(self.device)
                # [0:1] instead of [0] to keep two-dimensional array
                hidden_state_batch = torch.FloatTensor(self.hidden_states_batch[0:1, idxes]).reshape(-1, self.hidden_state_size).to(self.device)
                act_batch = torch.FloatTensor(self.act_batch[:, idxes]).reshape(-1).to(self.device)
                done_batch = torch.FloatTensor(self.done_batch[:, idxes]).reshape(-1).to(self.device)
                log_prob_act_batch = torch.FloatTensor(self.log_prob_act_batch[:, idxes]).reshape(-1).to(self.device)
                value_batch = torch.FloatTensor(self.value_batch[:-1, idxes]).reshape(-1).to(self.device)
                return_batch = torch.FloatTensor(self.return_batch[:, idxes]).reshape(-1).to(self.device)
                adv_batch = torch.FloatTensor(self.adv_batch[:, idxes]).reshape(-1).to(self.device)
                yield obs_batch, hidden_state_batch, act_batch, done_batch, log_prob_act_batch, value_batch, return_batch, adv_batch

    def fetch_log_data(self):
        if 'env_reward' in self.info_batch[0][0]:
            rew_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                rew_batch.append([info['env_reward'] for info in infos])
            rew_batch = np.array(rew_batch)
        else:
            rew_batch = self.rew_batch.numpy()
        if 'env_done' in self.info_batch[0][0]:
            done_batch = []
            for step in range(self.num_steps):
                infos = self.info_batch[step]
                done_batch.append([info['env_done'] for info in infos])
            done_batch = np.array(done_batch)
        else:
            done_batch = self.done_batch.numpy()
        return rew_batch, done_batch



#%%
class BaseAgent(object):
    """
    Class for the basic agent objects.
    To define your own agent, subclass this class and implement the functions below.
    """

    def __init__(self, 
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 num_checkpoints,
                 env_valid=None,
                 storage_valid=None):
        """
        env: (gym.Env) environment following the openAI Gym API
        """
        self.env = env
        self.policy = policy
        self.logger = logger
        self.storage = storage
        self.device = device
        self.num_checkpoints = num_checkpoints
        self.env_valid = env_valid
        self.storage_valid = storage_valid
        self.t = 0

    def predict(self, obs):
        """
        Predict the action with the given input
        """
        pass

    def update_policy(self):
        """
        Train the neural network model
        """
        pass

    def train(self, num_timesteps):
        """
        Train the agent with collecting the trajectories
        """
        pass

    def evaluate(self):
        """
        Evaluate the agent
        """
        pass

class PPO(BaseAgent):
    def __init__(self,
                 env,
                 policy,
                 logger,
                 storage,
                 device,
                 n_checkpoints,
                 env_valid=None,
                 storage_valid=None,
                 n_steps=128,
                 n_envs=8,
                 epoch=3,
                 mini_batch_per_epoch=8,
                 mini_batch_size=32*8,
                 gamma=0.99,
                 lmbda=0.95,
                 learning_rate=2.5e-4,
                 grad_clip_norm=0.5,
                 eps_clip=0.2,
                 value_coef=0.5,
                 entropy_coef=0.01,
                 normalize_adv=True,
                 normalize_rew=True,
                 use_gae=True,
                 **kwargs):

        super(PPO, self).__init__(env, policy, logger, storage, device,
                                  n_checkpoints, env_valid, storage_valid)

        self.n_steps = n_steps
        self.n_envs = n_envs
        self.epoch = epoch
        self.mini_batch_per_epoch = mini_batch_per_epoch
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, eps=1e-5)
        self.grad_clip_norm = grad_clip_norm
        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.normalize_adv = normalize_adv
        self.normalize_rew = normalize_rew
        self.use_gae = use_gae

    def predict(self, obs, hidden_state, done):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(device=self.device)
            hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
            mask = torch.FloatTensor(1-done).to(device=self.device)
            dist, value, hidden_state = self.policy(obs, hidden_state, mask)
            act = dist.sample()
            log_prob_act = dist.log_prob(act)

        return act.cpu().numpy(), log_prob_act.cpu().numpy(), value.cpu().numpy(), hidden_state.cpu().numpy()

    def predict_w_value_saliency(self, obs, hidden_state, done):
        obs = torch.FloatTensor(obs).to(device=self.device)
        obs.requires_grad_()
        obs.retain_grad()
        hidden_state = torch.FloatTensor(hidden_state).to(device=self.device)
        mask = torch.FloatTensor(1-done).to(device=self.device)
        dist, value, hidden_state = self.policy(obs, hidden_state, mask)
        value.backward(retain_graph=True)
        act = dist.sample()
        log_prob_act = dist.log_prob(act)

        return act.detach().cpu().numpy(), log_prob_act.detach().cpu().numpy(), value.detach().cpu().numpy(), hidden_state.detach().cpu().numpy(), obs.grad.data.detach().cpu().numpy()

    def optimize(self):
        pi_loss_list, value_loss_list, entropy_loss_list = [], [], []
        batch_size = self.n_steps * self.n_envs // self.mini_batch_per_epoch
        if batch_size < self.mini_batch_size:
            self.mini_batch_size = batch_size
        grad_accumulation_steps = batch_size / self.mini_batch_size
        grad_accumulation_cnt = 1

        self.policy.train()
        for e in range(self.epoch):
            recurrent = self.policy.is_recurrent()
            generator = self.storage.fetch_train_generator(mini_batch_size=self.mini_batch_size,
                                                           recurrent=recurrent)
            for sample in generator:
                obs_batch, hidden_state_batch, act_batch, done_batch, \
                    old_log_prob_act_batch, old_value_batch, return_batch, adv_batch = sample
                mask_batch = (1-done_batch)
                dist_batch, value_batch, _ = self.policy(obs_batch, hidden_state_batch, mask_batch)

                # Clipped Surrogate Objective
                log_prob_act_batch = dist_batch.log_prob(act_batch)
                ratio = torch.exp(log_prob_act_batch - old_log_prob_act_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_batch
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped Bellman-Error
                clipped_value_batch = old_value_batch + (value_batch - old_value_batch).clamp(-self.eps_clip, self.eps_clip)
                v_surr1 = (value_batch - return_batch).pow(2)
                v_surr2 = (clipped_value_batch - return_batch).pow(2)
                value_loss = 0.5 * torch.max(v_surr1, v_surr2).mean()

                # Policy Entropy
                entropy_loss = dist_batch.entropy().mean()
                loss = pi_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                loss.backward()

                # Let model to handle the large batch-size with small gpu-memory
                if grad_accumulation_cnt % grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                grad_accumulation_cnt += 1
                pi_loss_list.append(-pi_loss.item())
                value_loss_list.append(-value_loss.item())
                entropy_loss_list.append(entropy_loss.item())

        summary = {'Loss/pi': np.mean(pi_loss_list),
                   'Loss/v': np.mean(value_loss_list),
                   'Loss/entropy': np.mean(entropy_loss_list)}
        return summary

    def train(self, num_timesteps):
        save_every = num_timesteps // self.num_checkpoints
        checkpoint_cnt = 0
        obs = self.env.reset()
        hidden_state = np.zeros((self.n_envs, self.storage.hidden_state_size))
        done = np.zeros(self.n_envs)

        if self.env_valid is not None:
            obs_v = self.env_valid.reset()
            hidden_state_v = np.zeros((self.n_envs, self.storage.hidden_state_size))
            done_v = np.zeros(self.n_envs)

        while self.t < num_timesteps:
            # Run Policy
            self.policy.eval()
            for _ in range(self.n_steps):
                act, log_prob_act, value, next_hidden_state = self.predict(obs, hidden_state, done)
                next_obs, rew, done, info = self.env.step(act)
                self.storage.store(obs, hidden_state, act, rew, done, info, log_prob_act, value)
                obs = next_obs
                hidden_state = next_hidden_state
            value_batch = self.storage.value_batch[:self.n_steps]
            _, _, last_val, hidden_state = self.predict(obs, hidden_state, done)
            self.storage.store_last(obs, hidden_state, last_val)
            # Compute advantage estimates
            self.storage.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            #valid
            if self.env_valid is not None:
                for _ in range(self.n_steps):
                    act_v, log_prob_act_v, value_v, next_hidden_state_v = self.predict(obs_v, hidden_state_v, done_v)
                    next_obs_v, rew_v, done_v, info_v = self.env_valid.step(act_v)
                    self.storage_valid.store(obs_v, hidden_state_v, act_v,
                                             rew_v, done_v, info_v,
                                             log_prob_act_v, value_v)
                    obs_v = next_obs_v
                    hidden_state_v = next_hidden_state_v
                _, _, last_val_v, hidden_state_v = self.predict(obs_v, hidden_state_v, done_v)
                self.storage_valid.store_last(obs_v, hidden_state_v, last_val_v)
                self.storage_valid.compute_estimates(self.gamma, self.lmbda, self.use_gae, self.normalize_adv)

            # Optimize policy & valueq
            summary = self.optimize()
            # Log the training-procedure
            self.t += self.n_steps * self.n_envs
            rew_batch, done_batch = self.storage.fetch_log_data()
            if self.storage_valid is not None:
                rew_batch_v, done_batch_v = self.storage_valid.fetch_log_data()
            else:
                rew_batch_v = done_batch_v = None
            self.logger.feed(rew_batch, done_batch, rew_batch_v, done_batch_v)
            self.logger.dump()
            self.optimizer = adjust_lr(self.optimizer, self.learning_rate, self.t, num_timesteps)
            # Save the model
            if self.t > ((checkpoint_cnt+1) * save_every):
                print("Saving model.")
                torch.save({'model_state_dict': self.policy.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                             self.logger.logdir + '/model_' + str(self.t) + '.pth')
                checkpoint_cnt += 1
        self.env.close()
        if self.env_valid is not None:
            self.env_valid.close()



#%%
class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = "ARGHH" #tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer

class VecEnvWrapper(VecEnv):
    """
    An environment wrapper that applies to an entire batch
    of environments at once.
    """

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.venv, name)
    
class VecEnvObservationWrapper(VecEnvWrapper):
    @abstractmethod
    def process(self, obs):
        pass

    def reset(self):
        obs = self.venv.reset()
        return self.process(obs)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self.process(obs), rews, dones, infos

class VecExtractDictObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        super().__init__(venv=venv,
            observation_space=venv.observation_space.spaces[self.key])

    def process(self, obs):
        return obs[self.key]

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)
        
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class VecNormalize(VecEnvWrapper):
    """
    A vectorized wrapper that normalizes the observations
    and returns from an environment.
    """

    def __init__(self, venv, ob=True, ret=True, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)

        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for i in range(len(infos)):
            infos[i]['env_reward'] = rews[i]
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

class TransposeFrame(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[2], obs_shape[0], obs_shape[1]), dtype=np.float32)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs.transpose(0,3,1,2), reward, done, info

    def reset(self):
        obs = self.venv.reset()
        return obs.transpose(0,3,1,2)

class ScaledFloatFrame(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(venv=env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return obs/255.0, reward, done, info

    def reset(self):
        obs = self.venv.reset()
        return obs/255.0
    


#%%
args = {
    "num_levels": 0,
    "start_level": 0,
    "distribution_mode": "hard",
    "num_threads": 8,
    "random_percent": 0,
    "step_penalty": 0,
    "key_penalty": 0,
    "rand_region": 0,
}

n_steps = hyperparameters["n_steps"]
n_envs = hyperparameters["n_envs"]

env_name = "maze_yellowstar_redgem"



#%%
# env = gym.make('procgen:procgen-maze_yellowstar_redgem-v0',
#                env_name=env_name,
#                num_levels=args["num_levels"],
#                start_level=args["start_level"],
#                distribution_mode=args["distribution_mode"],
#             #    num_threads=args["num_threads"]
# )

# obs = env.reset()
# plt.imshow(obs)

# venv = maze.create_venv(num=n_envs,
#                         start_level=args["start_level"], 
#                         num_levels=args["num_levels"],
#                         env_name=env_name)
# visualization.visualize_venv(venv, 0)



#%%

venv = ProcgenEnv(num_envs=n_envs,
                  env_name=env_name,
                  num_levels=args["num_levels"],
                  start_level=args["start_level"],
                  distribution_mode=args["distribution_mode"],
                  num_threads=args["num_threads"],
                  render_mode="rgb_array",
)

venv = VecExtractDictObs(venv, "rgb")
normalize_rew = hyperparameters["normalize_rew"]
if normalize_rew:
    venv = VecNormalize(venv, ob=False) # normalizing returns, but not
    #the img frames
venv = TransposeFrame(venv)
venv = ScaledFloatFrame(venv)

visualization.visualize_venv(venv, 0)




#%%
# def create_venv(args, hyperparameters, is_valid=False):
#     venv = ProcgenEnv(num_envs=n_envs,
#                         env_name=env_name,
#                         num_levels=args["num_levels"],
#                         start_level=args["start_level"],
#                         distribution_mode=args["distribution_mode"],
#                         num_threads=args["num_threads"],
#                         # random_percent=args["random_percent"],
#                         # step_penalty=args["step_penalty"],
#                         # key_penalty=args["key_penalty"],
#                         # rand_region=args["rand_region"]
#     )
#     venv.reset()
#     plt.imshow(env.render(mode='rgb_array'))
#     venv = VecExtractDictObs(venv, "rgb")
#     normalize_rew = hyperparameters["normalize_rew"]
#     if normalize_rew:
#         venv = VecNormalize(venv, ob=False) # normalizing returns, but not
#         #the img frames
#     venv = TransposeFrame(venv)
#     venv = ScaledFloatFrame(venv)
#     return venv
    
# env = create_venv(args, hyperparameters)

# visualization.visualize_venv(env, 0)



#%%
# Importing training code haphazardly from here

#%%
# Load model weights

#%%
# Initializing model

observation_space = venv.observation_space
observation_shape = observation_space.shape
in_channels = observation_shape[0]
action_space = venv.action_space

model = ImpalaModel(in_channels=in_channels)

# Discrete action space
recurrent = hyperparameters.get('recurrent', False)
if isinstance(action_space, gym.spaces.Discrete):
    action_size = action_space.n
    policy = CategoricalPolicy(model, recurrent, action_size)
else:
    raise NotImplementedError

# Load weights from saved checkpoint
checkpoint_path = "/home/paul/procgen-goal-detection/exploration/logs/train/maze_yellowstar_redgem/model_19900160.pth"
if os.path.exists(checkpoint_path):
    # Load the entire checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Extract the model's state_dict and load it into the model
    model_state_dict = checkpoint['model_state_dict']
    policy.load_state_dict(model_state_dict)

    #policy.load_state_dict(torch.load(checkpoint_path))
    print(f"Loaded model from {checkpoint_path}")
else:
    print(f"Checkpoint {checkpoint_path} not found. Starting training from scratch.")


policy.to(device)



#%%
# Initializing logger

logdir = os.path.join('logs', 'train', env_name)
run_name = time.strftime("%Y-%m-%d__%H-%M-%S")
logdir = os.path.join(logdir, run_name)
if not (os.path.exists(logdir)):
    os.makedirs(logdir)

logger = Logger(n_envs, logdir, use_wandb=False)



#%%
# Initializing Storage

hidden_state_dim = model.output_dim
storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)



#%%
# Initializing Agent

algo = hyperparameters["algo"]
if algo == 'ppo':
    AGENT = PPO
else:
    raise NotImplementedError

num_checkpoints = 10  # number of checkpoints to store

agent = AGENT(venv, policy, logger, storage, device,
              num_checkpoints, 
              **hyperparameters)



#%%
# Training

num_timesteps = 10000000
agent.train(num_timesteps)



# %%
# Load the episode rewards from the file where you saved them
episode_rewards_file = os.path.join(logdir, 'episode_rewards.csv') # Adjust the path accordingly
episode_rewards_df = pd.read_csv(episode_rewards_file)

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards_df['episode_rewards'], label='Training rewards')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Model Performance over Episodes')
plt.legend()
plt.show()
# %%
# Plot rolling average of rewards:

episode_rewards_file = os.path.join(logdir, 'episode_rewards.csv') # Adjust the path accordingly
episode_rewards_df = pd.read_csv(episode_rewards_file)

rolling_window = 70  # This is a common choice, but adjust based on your needs
rolling_mean = episode_rewards_df['episode_rewards'].rolling(window=rolling_window).mean()

plt.figure(figsize=(10, 6))
plt.plot(episode_rewards_df['episode_rewards'], label='Training rewards', alpha=0.6)
plt.plot(rolling_mean, label=f'Rolling mean (window={rolling_window})', color='red')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Model Performance over Episodes')
plt.legend()
plt.show()

#%%