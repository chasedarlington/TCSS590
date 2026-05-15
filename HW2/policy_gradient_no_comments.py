import os
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from utils import rollout, log_density

def train_model(policy,baseline,trajs,policy_optim,baseline_optim,device,gamma=0.99,baseline_train_batch_size=64,baseline_num_epochs=5):
    states_all = []
    actions_all = []
    returns_all = []
    for traj in trajs:
        states_singletraj = traj['observations']
        actions_singletraj = traj['actions']
        rewards_singletraj = traj['rewards']
        returns_singletraj = np.zeros_like(rewards_singletraj)
        running_return = 0.0
        for t in reversed(range(len(rewards_singletraj))):                                              # reversed(rewards_singletraj):
            running_return = rewards_singletraj[t] + gamma * running_return                             # reward + gamma * running_return 
            returns_singletraj[t] = running_return                                                      # np.insert(returns_singletraj, 0, running_return)
        states_all.append(states_singletraj)
        actions_all.append(actions_singletraj)
        returns_all.append(returns_singletraj)
    states = np.concatenate(states_all)
    actions = np.concatenate(actions_all)
    returns = np.concatenate(returns_all).astype(np.float32)                                            # np.concatenate(returns_all)
    returns = (returns - returns.mean())/(returns.std() + 1e-8)                                         #####################################################
    n = len(states)
    indices = np.arange(n)
    for _ in range(baseline_num_epochs):
        np.random.shuffle(indices)
        for i in range(0, n, baseline_train_batch_size):                                                # range(n // baseline_train_batch_size):
            batch_indices = indices[i : i + baseline_train_batch_size]                                  # indices[baseline_train_batch_size * i : baseline_train_batch_size * (i + 1)]
            batch_indices = torch.LongTensor(batch_indices).to(device)
            obs_batch = torch.from_numpy(states[batch_indices.cpu()]).float().to(device)
            returns_batch = torch.from_numpy(returns[batch_indices.cpu()]).float().to(device)
            returns_batch = returns_batch.squeeze(-1)                                                   # add squeeze
            baseline_pred = baseline(obs_batch).squeeze(-1)
            baseline_loss = torch.nn.functional.mse_loss(baseline_pred, returns_batch)
            baseline_optim.zero_grad()
            baseline_loss.backward()
            #torch.nn.utils.clip_grad_norm_(baseline.parameters(), max_norm=1.0)                         #####################################################
            baseline_optim.step()
    with torch.no_grad():
        baseline_values = baseline(torch.from_numpy(states).float().to(device)).squeeze(-1)
        advantages = torch.from_numpy(returns).float().to(device).squeeze(-1) - baseline_values         # torch.from_numpy(returns).float().to(device) - baseline_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    rollout_actions = torch.from_numpy(actions).float().to(device)                                      #####################################################
    _, std, logstd = policy(torch.from_numpy(states).float().to(device))                                # policy(torch.Tensor(states).to(device)) 
    log_policy = log_density(rollout_actions, policy.mu, std, logstd)                                   # log_density(torch.Tensor(actions).to(device), policy.mu, std, logstd)
                                                                                                        # baseline_pred = baseline(torch.from_numpy(states).float().to(device))
                                                                                                        # advantages = returns - baseline_pred.detach()
    policy_loss = -(log_policy * advantages.detach()).mean()                                            # -(log_policy * advantages)
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()
    del states, actions, returns, states_all, actions_all, returns_all

def simulate_policy_pg(env, policy, baseline, num_epochs, max_path_length, batch_size,gamma, baseline_train_batch_size, baseline_num_epochs, print_freq, device = "cuda", render=False):
    policy_optim = optim.Adam(policy.parameters())
    baseline_optim = optim.Adam(baseline.parameters())
    for iter_num in range(num_epochs):
        sample_trajs = []
        for _ in range(batch_size):                                                                     # to _
            sample_traj = rollout(env,policy,device,episode_length=max_path_length)                            # remove render arg
            sample_trajs.append(sample_traj)
        if iter_num % print_freq == 0:
            rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num, rewards_np, path_length))
        train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, device, gamma=gamma,
                    baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs)
