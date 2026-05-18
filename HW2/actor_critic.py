import torch
from torch import optim
import copy
import numpy as np
from utils import collect_trajs

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



def compute_losses(policy, qf, target_qf, obs_t, actions_t, rewards_t, next_obs_t, not_dones_t, device, discount=0.99):
    obs_t = torch.as_tensor(obs_t, device=device, dtype=torch.float32)
    actions_t = torch.as_tensor(actions_t, device=device, dtype=torch.float32)
    rewards_t = torch.as_tensor(rewards_t, device=device, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_t, device=device, dtype=torch.float32)
    not_dones_t = torch.as_tensor(not_dones_t, device=device, dtype=torch.float32)

    # Policy loss:
    a_sampled_t, _, _ = policy(obs_t) # get (differentiable) action samples a_sampled_t from the policy using policy.forward
    policy_q = qf(torch.cat([obs_t, a_sampled_t], dim=-1)) #Compute the Q values as qf(obs_t, a_sampled_t)
    policy_loss = -policy_q.mean() #Policy loss is the mean over negative Q values

    # QF loss:
    q_pred = qf(torch.cat([obs_t, actions_t], dim=-1)) #Compute q predictions using obs_t, actions_t
    with torch.no_grad(): # Compute q targets using reward + target_qf for next_obs_t and new actions sampled from the policy
        next_a_t, _, _ = policy(next_obs_t)
        target_q_values = target_qf(torch.cat([next_obs_t, next_a_t], dim=-1))
        q_target = rewards_t + not_dones_t * discount * target_q_values
    qf_loss = torch.nn.functional.mse_loss(q_pred, q_target) # Compute Bellman error as mean squared error between q_predictions and q_targets

    # TODO END

    return policy_loss, qf_loss


def soft_update_target(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def simulate_policy_ac(
        env,
        policy,
        qf,
        target_qf,
        replay_buffer,
        device,
        episode_length: int = 100,
        num_epochs: int = 200,
        batch_size=32,
        target_weight= 5e-3,
        num_update_steps=100,
        render = False,
        print_freq=10,
        learning_rate = 3e-4,
):
    env.reset()

    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    qf_optimizer = optim.Adam(qf.parameters(), lr=learning_rate)

    # Copy parameters initially
    soft_update_target(qf, target_qf, 1.0)

    for iter_num in range(num_epochs):
        sample_trajs = []

        # Sampling trajectories
        for it in range(batch_size):
            sample_traj = collect_trajs(env, policy, replay_buffer, device, episode_length=episode_length, render=render)
            sample_trajs.append(sample_traj)

        if iter_num % print_freq == 0:
            rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num, rewards_np, path_length))

        for update_num in range(num_update_steps):
            obs_t, actions_t, rewards_t, next_obs_t, not_dones_t = replay_buffer.sample(batch_size)

            policy_loss, qf_loss = compute_losses(policy, qf, target_qf, obs_t, actions_t, rewards_t, next_obs_t,
                                                  not_dones_t, device)

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            qf_optimizer.zero_grad()
            qf_loss.backward()
            qf_optimizer.step()


            soft_update_target(qf, target_qf, target_weight)
