import torch
import torch.nn as nn
import copy
import numpy as np
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_size, action_size, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, obs_size), dtype=np.float32)
        self.next_obses = np.empty((capacity, obs_size), dtype=np.float32)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        idxs = np.arange(self.idx, self.idx + obs.shape[0]) % self.capacity
        self.obses[idxs] = copy.deepcopy(obs)
        self.actions[idxs] = copy.deepcopy(action)
        self.rewards[idxs] = copy.deepcopy(reward)
        self.next_obses[idxs] = copy.deepcopy(next_obs)
        self.not_dones[idxs] = 1.0 - copy.deepcopy(done)

        self.full = self.full or (self.idx + obs.shape[0] >= self.capacity)
        self.idx = (self.idx + obs.shape[0]) % self.capacity

    def sample(self, batch_size):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        return obses, actions, rewards, next_obses, not_dones

def reward_fn_reacher(state, action):
        cos_theta = state[:, :2]
        sin_theta = state[:, 2:4]
        qpos = state[:, 4:6]
        qvel = state[:, 6:8]
        vec = state[:, 8:11]

        reward_dist = -torch.norm(vec, dim=1)
        reward_ctrl = -torch.square(action).sum(dim=1)

        reward = reward_dist + reward_ctrl
        reward = reward[:, None]
        return reward

def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DeterministicDynamicsModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=64, hidden_depth=2):
        super(DeterministicDynamicsModel, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.trunk = mlp(num_inputs, hidden_dim, num_outputs, hidden_depth)

    def forward(self, x):
        v = self.trunk(x)
        v = v + x[:, :v.shape[1]]
        return v


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

