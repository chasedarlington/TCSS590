import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch

from utils import DeterministicDynamicsModel, set_random_seed, reward_fn_reacher
from train_mbrl import simulate_mbrl
from rollouts import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='single', help='choose type of model: single or ensemble')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--plan_mode', type=str, default='random_mpc', help='choose planning method: random_mpc or mppi')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    set_random_seed(args.seed)

    # Environment and reward definition
    env = gym.make("Reacher-v4", render_mode='human' if args.render else None)
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    max_path_length = 50

    # Define dynamics model
    hidden_dim_model = 64
    hidden_depth_model = 2
    if args.model_type == 'single':
        model = DeterministicDynamicsModel(env.observation_space.shape[0] + env.action_space.shape[0], env.observation_space.shape[0], hidden_dim=hidden_dim_model, hidden_depth=hidden_depth_model)
        model.to(device)
    elif args.model_type == 'ensemble':
        num_ensembles = 5
        model = []
        for model_id in range(num_ensembles):
            curr_model = DeterministicDynamicsModel(env.observation_space.shape[0] + env.action_space.shape[0], env.observation_space.shape[0], hidden_dim=hidden_dim_model, hidden_depth=hidden_depth_model)
            curr_model.to(device)
            model.append(curr_model)
    else:
        raise NotImplementedError("No other model types implemented")

    # Training hyperparameters
    num_epochs = 15
    batch_size = 250
    num_agent_train_epochs_per_iter = 10
    num_traj_per_iter = batch_size // max_path_length
    gamma = 0.99
    print_freq = 1
    capacity = 100000
    mpc_horizon = 10
    n_samples_mpc = 1000

    if not args.test:
        # Training and model saving code
        simulate_mbrl(env, model, plan_mode=args.plan_mode, num_epochs=num_epochs, max_path_length=max_path_length, mpc_horizon=mpc_horizon,
                      n_samples_mpc=n_samples_mpc, batch_size=batch_size, num_agent_train_epochs_per_iter=num_agent_train_epochs_per_iter,
                      capacity=capacity, num_traj_per_iter=num_traj_per_iter, gamma=gamma, print_freq=print_freq, device=device, reward_fn=reward_fn_reacher)
        if isinstance(model, list):
            for model_idx, curr_model in enumerate(model):
                torch.save(curr_model.state_dict(), f'{args.model_type}_{args.plan_mode}_final_{model_idx}.pth')
        else:
            torch.save(model.state_dict(), f'{args.model_type}_{args.plan_mode}_final.pth')
    else:
        print('loading pretrained mbrl')
        if isinstance(model, list):
            for model_idx in range(len(model)):
                model[model_idx].load_state_dict(torch.load(f'{args.model_type}_{args.plan_mode}_final_{model_idx}.pth', map_location=device))
        else:
            model.load_state_dict(torch.load(f'{args.model_type}_{args.plan_mode}_final.pth', map_location=device))

    evaluate(env, model, plan_mode=args.plan_mode, mpc_horizon=mpc_horizon, n_samples_mpc=n_samples_mpc, num_validation_runs=100, episode_length=max_path_length, render=args.render, reward_fn=reward_fn_reacher)
