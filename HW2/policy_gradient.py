import os
import torch
import numpy as np
from torch import nn
from torch import optim
import argparse
import collections
import functools
import math
import time
from typing import Any, Callable, Dict, Optional, Sequence, List
import gymnasium as gym
from gymnasium import utils
import torch.nn.functional as F
import copy
from typing import Tuple, Optional, Union
import matplotlib.pyplot as plt
from utils import rollout, log_density

"""
Train one policy-gradient update using collected trajectory data.
Also trains the baseline network to predict returns.
"""
def train_model(
        policy, 
        baseline, 
        trajs, 
        policy_optim, 
        baseline_optim, 
        device, 
        gamma=0.99, 
        baseline_train_batch_size=64,
        baseline_num_epochs=5
):
    
    states_all = [] # OBSERVATIONS
    actions_all = [] # ACTIONS
    returns_all = [] # DISCOUNTED SUM OF FUTURE REWARDS (RETURN TO GO)
    # log_probs_all = [] # LOG PROBABILITIES OF ACTIONS UNDER THE POLICY

    ### FLATTEN TRAJECTORIES INTO STATES, ACTIONS, REWARDS, AND DISCOUNTED RETURNS !!!
    for traj in trajs:
        states_singletraj = traj['observations'] # OBSERVATIONS !!
        actions_singletraj = traj['actions'] # ACTIONS !!
        rewards_singletraj = traj['rewards'] # REWARDS !!
        returns_singletraj = np.zeros_like(rewards_singletraj) # DISCOUNTED RETURNS !!
        # log_probs_singletraj = traj['log_probs'] # LOG PROBABILITIES OF ACTIONS UNDER THE POLICY !!
        
        running_return = 0.0
        for reward in reversed(rewards_singletraj):

            # note: running return is the current reward + gamma * prior sum of rewards
            running_return = reward + gamma * running_return 
            
            # note: inject running return at the front of the returns NumPy array, 
            #   so that returns_singletraj[t] is the RETURN TO GO from time step t
            returns_singletraj = np.insert(returns_singletraj, 0, running_return)

        states_all.append(states_singletraj) # append observations
        actions_all.append(actions_singletraj) # append actions
        returns_all.append(returns_singletraj) # append discounted returns
        # log_probs_all.append(log_probs_singletraj) # append log probabilities of actions under the policy

    ## for NumPy arrays...
    states = np.concatenate(states_all) # concatenate all observations (into one NumPy array)
    actions = np.concatenate(actions_all) # concatenate all actions (into one NumPy array)
    returns = np.concatenate(returns_all) # concatenate all discounted returns (into one NumPy array)
    # log_probs = np.concatenate(log_probs_all) # concatenate all log probabilities of actions under the policy (into one NumPy array)

    ## for PyTorch tensors...
    # states = torch.tensor(states, dtype=torch.float32, device=device).to(device) 
    # actions = torch.tensor(actions, dtype=torch.float32, device=device).to(device)
    # returns = torch.tensor(returns_all, dtype=torch.float32, device=device).to(device) 
    # log_probs = torch.stack(all_log_probs).to(device)
        
    #### TRAIN BASELINE NETWORK TO PREDICT RETURNS !!!
    
    ## if states is a NumPy array:
    n = len(states)  
    indices = np.arange(n)
    
    ## ??? ? ?? ? ? ? ? ? 
    # criterion = torch.nn.MSELoss() 
    
    ## if states is a PyTorch tensor:
    # n = states.shape[0]

    for _ in range(baseline_num_epochs):
        
        ## DO THE SHUFFLE !!
        
        ## if n were a numpy array:
        np.random.shuffle(indices)
 
        ## if n were a tensor: 
        #   indices = torch.randperm(n, device=device) 
        
        for i in range(n // baseline_train_batch_size): # or range(0, n, baseline_train_batch_size)
            
            ## send concatenated states and returns to device, but only for the current batch of data (batch_indices)
            
            ## if states & actions are numpy arrays: --> CONVERT TO TENSOR !! 
            batch_indices = indices[baseline_train_batch_size * i : baseline_train_batch_size * (i + 1)] # get batch indices for current batch
            batch_indices = torch.LongTensor(batch_indices).to(device) # convert batch indices to tensor and send to device
            obs_batch = torch.from_numpy(states[batch_indices.cpu()]).float().to(device) # get batch observations (states) and send to device
            returns_batch = torch.from_numpy(returns[batch_indices.cpu()]).float().to(device) # get batch returns and send to device

            ## alternate to the above ? ? ? ? ? ? ? ? ?  ? ? ? ? ? ?  ? ? ? ?? 
            # end = i + baseline_train_batch_size
            # batch_indices = indices[i:end] 
            # obs_batch = torch.from_numpy(states[batch_indices]).float().to(device)
            # returns_batch = torch.from_numpy(returns[batch_indices]).float().to(device)

            ## if states & actions are tensors: 
            # end = i + baseline_train_batch_size
            # batch_indices = indices[i:end]
            # obs_batch = states[batch_indices])
            # returns_batch = returns[batch_indices]
            
            ## Pass observations (states) to baseline network, then remove final singleton dimension
            baseline_pred = baseline(obs_batch).squeeze(-1) 

            ## compute MSE loss (baseline predictions vs actual returns) for returns batch
            baseline_loss = torch.nn.functional.mse_loss(baseline_pred, returns_batch)

            baseline_optim.zero_grad()
            baseline_loss.backward()
            baseline_optim.step()

    #### COMPUTE ADVANTAGES
    with torch.no_grad():

        ## if observations (states) were a numpy array:
        baseline_values = baseline(torch.from_numpy(states).float().to(device)).squeeze(-1) # compute baseline predictions for all states (observations) in the trajectories
        advantages = torch.from_numpy(returns).float().to(device) - baseline_values # compute advantages (returns - baseline predictions)

        ## if observations (states) were a tensor: 
        #   baseline_values = baseline(observations_tensor).squeeze(-1)
        #   advantages = returns_tensor - baseline_values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize advantages (zero mean, unit variance)

    #### TRAIN POLICY NETWORK TO OPTIMIZE SURROGATE OBJECTIVE !!! 

    ### CALCULATE LOSS W/ CALCULATED LOG PROBS USING LOG_DENSITY FXN:

    ## if observations & actions are numpy arrays:
    actions, std, logstd = policy(torch.Tensor(states).to(device))
    log_policy = log_density(torch.Tensor(actions).to(device), policy.mu, std, logstd)
    baseline_pred = baseline(torch.from_numpy(states).float().to(device))
    advantages = returns - baseline_pred.detach() # compute advantages (returns - baseline predictions) and detach baseline predictions from computation graph (so that we don't backprop through the baseline network when updating the policy network)
    
    ## if observations & actions are tensors:
    # action, std, logstd = policy(observations_tensor.to(device))
    # log_policy = log_density(actions_tensor.to(device), policy.mu, std, logstd
    # baseline_pred = baseline(observations_tensor.to(device))
    # advantages = returns - baseline_pred.detach()
    
    ## calc surrogate objective (negative b/c we want to maximize):
    policy_loss = -(log_policy * advantages) # sum over all data points in the batch (instead of averaging, as below)
    # policy_loss = -(log_policy * advantages).mean() # average over all data points in the batch (instead of summing, as above)

    ### CALCULATING LOSS W/ LOG PROBS OF ACTIONS UNDER THE POLICY: 
    #   (instead of calculating log probs from scratch using log_density fxn, as above)
    # policy_loss = -(log_probs * advantages) # sum over all data points in the batch (instead of averaging, as below)
    # policy_loss = -(log_probs * advantages).mean() # average over all data points in the batch (instead of summing, as above)

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    ### OPTIONAL: DELETE VARIABLES TO FREE UP MEMORY
    del states, actions, returns, states_all, actions_all, returns_all 

def simulate_policy_pg(env, policy, baseline, num_epochs=200, max_path_length=200, batch_size=100,
                       gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5, print_freq=10, device = "cuda", render=False):
    
    policy_optim = optim.Adam(policy.parameters())
    baseline_optim = optim.Adam(baseline.parameters())

    for iter_num in range(num_epochs):
        sample_trajs = []
        for it in range(batch_size):
            sample_traj = rollout(
                env,
                policy,
                episode_length=max_path_length,
                render=False)
            sample_trajs.append(sample_traj)

        # Logging returns occasionally
        if iter_num % print_freq == 0:
            rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num, rewards_np, path_length))

        # Training model
        train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, device, gamma=gamma,
                    baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs)
