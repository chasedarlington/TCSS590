import torch
import torch.nn as nn

import math
import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import distributions as pyd
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)


class PGPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=64, hidden_depth=2):
        super(PGPolicy, self).__init__()
        self.trunk = mlp(num_inputs, hidden_dim, num_outputs*2, hidden_depth)

    def forward(self, x):
        outs = self.trunk(x)
        mu, std, log_std = self.dist_create(outs)
        std = torch.exp(log_std)
        action = torch.normal(mu, std)
        self.mu = mu

        return action, std, log_std

    def dist_create(self, logits):
        min_log_std = -5
        max_log_std = 5
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        loc = torch.tanh(loc)

        log_std = torch.sigmoid(scale)
        log_std = min_log_std + log_std * (max_log_std - min_log_std)
        std = torch.exp(log_std)
        return loc, std, log_std


class PGBaseline(nn.Module):
    def __init__(self, num_inputs, hidden_dim=64, hidden_depth=2):
        super(PGBaseline, self).__init__()
        self.trunk = mlp(num_inputs, hidden_dim, 1, hidden_depth)

    def forward(self, x):
        v = self.trunk(x)
        return v

class ACPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=64, hidden_depth=2):
        super(ACPolicy, self).__init__()
        self.trunk = mlp(num_inputs, hidden_dim, num_outputs*2, hidden_depth)

    def forward(self, x):
        outs = self.trunk(x)
        mu, std, log_std = self.dist_create(outs)
        action = self.dist_sample_no_postprocess(mu, std)
        self.std = std
        self.mu = mu
        return action, std, log_std

    def dist_create(self, logits):
        min_log_std = -5
        max_log_std = 5
        loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
        loc = torch.tanh(loc)

        log_std = torch.sigmoid(scale)
        log_std = min_log_std + log_std * (max_log_std - min_log_std)
        std = torch.exp(log_std)
        return loc, std, log_std

    def dist_sample_no_postprocess(self, mu, std):
        action = torch.zeros((mu.shape[0], 1)).to(device)
        # TODO START
        # Hint: perform the reparameterization trick - action = mean + epsilon*std, where epsilon \sim N(0, I)
        # This will allow policy updates through gradient based updates via pathwise derivatives
        # TODO END
        return action



class QF(nn.Module):
    def __init__(self, num_inputs, hidden_dim=64, hidden_depth=2):
        super(QF, self).__init__()
        self.trunk = mlp(num_inputs, hidden_dim, 1, hidden_depth)

    def forward(self, x):
        v = self.trunk(x)
        return v



class TargetQF(nn.Module):
    def __init__(self, num_inputs, hidden_dim=64, hidden_depth=2):
        super(TargetQF, self).__init__()
        self.trunk = mlp(num_inputs, hidden_dim, 1, hidden_depth)

    def forward(self, x):
        v = self.trunk(x)
        return v


def collect_trajs(
        env,
        agent,
        replay_buffer,
        device,
        episode_length=math.inf,
        render=False,
):
    # Collect the following data
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    dones = []
    images = []

    path_length = 0

    o, info = env.reset() # (+info) UPDATED TO REFLECT NEW GYM API
    if render:
        env.render()

    while path_length < episode_length:
        o_for_agent = o

        action, _, _ = agent(torch.Tensor(o_for_agent).unsqueeze(0).to(device))
        action= action.cpu().detach().numpy()[0]

        # Step the simulation forward
        next_o, r, done, truncated, env_info = env.step(copy.deepcopy(action)) # (+truncated) UPDATED TO REFLECT NEW GYM API

        replay_buffer.add(o,
                          action,
                          r,
                          next_o,
                          done)

        # Render the environment
        if render:
            env.render()

        raw_obs.append(o)
        raw_next_obs.append(next_o)
        actions.append(action)
        rewards.append(r)
        dones.append(done)
        path_length += 1
        if done:
            break
        o = next_o

    # Prepare the items to be returned
    observations = np.array(raw_obs)
    next_observations = np.array(raw_next_obs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    dones = np.array(dones).reshape(-1, 1)

    # Return in the following format
    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=np.array(dones).reshape(-1, 1),
        images=np.array(images)
    )



def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk




def rollout(
        env,
        agent,
        episode_length=math.inf,
        render=False,
):
    # Collect the following data
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    dones = []
    images = []

    entropy = None
    log_prob = None
    agent_info = None
    path_length = 0

    o, info = env.reset() # (+info) UPDATED TO REFLECT NEW GYM API
    if render:
        env.render()

    while path_length < episode_length:
        o_for_agent = o # UPDATED TO REFLECT NEW GYM API

        action, _, _ = agent(torch.Tensor(o_for_agent).unsqueeze(0).to(device))
        action = action.cpu().detach().numpy()[0]

        # Step the simulation forward
        next_o, r, done, truncated, env_info = env.step(copy.deepcopy(action)) # (+truncated) UPDATED TO REFLECT NEW GYM API

        # Render the environment
        if render:
            env.render()

        raw_obs.append(o)
        raw_next_obs.append(next_o)
        actions.append(action)
        rewards.append(r)
        dones.append(done)
        path_length += 1
        if done:
            break
        obs = next_o

    # Prepare the items to be returned
    observations = np.array(raw_obs)
    next_observations = np.array(raw_next_obs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    dones = np.array(dones).reshape(-1, 1)

    # Return in the following format
    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=np.array(dones).reshape(-1, 1),
        images=np.array(images)
    )
