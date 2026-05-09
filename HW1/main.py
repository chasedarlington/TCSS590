"""
DO NOT MODIFY BESIDES HYPERPARAMETERS
"""

import sys
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import argparse

from utils import generate_paths, get_expert_data, PolicyGaussian, PolicyAutoRegressiveModel

from bc import simulate_policy_bc
from dagger import simulate_policy_dagger
import pytorch_utils as ptu
from evaluate import evaluate
from reach_goal.envs.pointmaze_env import PointMazeEnv
from reach_goal.envs.pointmaze_expert import WaypointController

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

#torch.manual_seed(0)
import random
#random.seed(0)
#np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='reacher', help='choose env (reacher/pointmaze)')
    parser.add_argument('--train', type=str, default='behavior_cloning', help='choose training method (behavior cloning/dagger)')
    parser.add_argument('--policy', type=str, default='gaussian', help='choose policy class (gaussian/autoregressive)')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--render',  action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0) # added for experiment
    parser.add_argument('--episode_length', type=int, default=None) # added for experiment
    parser.add_argument('--num_epochs', type=int, default=None) # added for experiment
    parser.add_argument('--batch_size', type=int, default=None)  # added for experiment
    parser.add_argument("--num_validation_runs", type=int, default=100) # added for experiment

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.render and sys.platform.startswith("linux"):
        import os
        os.environ.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libGLEW.so")

    # Get the expert data
    if args.env == 'reacher':
        file_path = 'data/reacher_expert_data.pkl'
    elif args.env == 'pointmaze':
        file_path = 'data/pointmaze_expert_data.pkl'
    else:
        raise ValueError('Invalid environment')
    expert_data = get_expert_data(file_path)

    flattened_expert = {'observations': [],
                        'actions': []}

    for expert_path in expert_data:
        for k in flattened_expert.keys():
            flattened_expert[k].append(expert_path[k])

    for k in flattened_expert.keys():
        flattened_expert[k] = np.concatenate(flattened_expert[k])

    # Define environment
    if args.env == 'reacher':
        env = gym.make("Reacher-v4", render_mode='human' if args.render else None)
    elif args.env == 'pointmaze':
        env = PointMazeEnv(render_mode='human' if args.render else 'rgb_array')
    else:
        raise ValueError('Invalid environment')

    # Define policy
    hidden_dim = 128
    hidden_depth = 2
    obs_size = env.observation_space.shape[0]
    ac_size = env.action_space.shape[0]
    ac_margin = 0.1

    if args.policy == 'gaussian':
        policy = PolicyGaussian(num_inputs=obs_size, num_outputs=ac_size, hidden_dim=hidden_dim, hidden_depth=hidden_depth)
    elif args.policy == 'autoregressive':
        num_buckets = 10
        policy = PolicyAutoRegressiveModel(num_inputs=obs_size, num_outputs=ac_size, hidden_dim=hidden_dim,
                                            hidden_depth=hidden_depth, num_buckets=num_buckets,
                                            ac_low=flattened_expert['actions'].min(axis=0) - ac_margin,
                                            ac_high=flattened_expert['actions'].max(axis=0) + ac_margin)
    policy.to(device)

    # Training hyperparameters for BC
    if args.env == 'reacher':
        episode_length = 50 #episode_length, num_epochs, batch_size
        num_epochs = 500
        batch_size = 32
        num_validation_runs = 100
    elif args.env == 'pointmaze':
        episode_length = 300
        num_epochs = 10
        batch_size = 128
        num_validation_runs = 100
    else:
        raise ValueError('Invalid environment')
    if args.episode_length is not None: #experiment
        episode_length = args.episode_length #experiment
    if args.num_epochs is not None: #experiment
        num_epochs = args.num_epochs #experiment
    if args.batch_size is not None: #experiment
        batch_size = args.batch_size #experiment
    if args.num_validation_runs is not None: #experiment
        num_validation_runs = args.num_validation_runs #experiment

    if not args.test:
        if args.train == 'behavior_cloning':
            # Train behavior cloning
            simulate_policy_bc(env, policy, expert_data, num_epochs=num_epochs, episode_length=episode_length,
                            batch_size=batch_size)
        elif args.train == 'dagger':
            if args.env == 'reacher':
                # Load interactive expert
                expert_policy = torch.load('data/reacher_expert_policy.pkl', map_location=torch.device(device), weights_only=False)
                print("Expert policy loaded")
                expert_policy.to(device)
                ptu.set_gpu_mode(torch.cuda.is_available())
            elif args.env == 'pointmaze':
                expert_policy = WaypointController(env.maze)
            else:
                raise ValueError('Invalid environment')

            # Training hyperparameters for DAgger
            num_dagger_iters=10
            num_epochs = int(num_epochs/num_dagger_iters)
            num_trajs_per_dagger=10
            # Train DAgger
            simulate_policy_dagger(env, policy, expert_data, expert_policy, num_epochs=num_epochs, episode_length=episode_length,
                            batch_size=batch_size, num_dagger_iters=num_dagger_iters, num_trajs_per_dagger=num_trajs_per_dagger)
        else:
            raise ValueError('Invalid training method')
        # Code for saving a policy to a checkpoint
        torch.save(policy.state_dict(), f'{args.policy}_{args.env}_{args.train}_final.pth')
    else:
        # Code for loading a policy from a checkpoint
        policy.load_state_dict(torch.load(f'{args.policy}_{args.env}_{args.train}_final.pth', map_location=device))

    # Code for policy evaluation post training
    evaluate(env, policy, args.train, num_validation_runs=num_validation_runs, episode_length=episode_length, render=args.render, env_name=args.env)