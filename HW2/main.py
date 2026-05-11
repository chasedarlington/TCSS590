import gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import argparse
from policy_gradient import simulate_policy_pg
from actor_critic import simulate_policy_ac, ReplayBuffer
from utils import ACPolicy, QF, TargetQF, PGPolicy, PGBaseline
from evaluate import evaluate
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='policy_gradient', help='choose task')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--render',  action='store_true', default=False)
    parser.add_argument('--double_q', action='store_true', default=False)
    args = parser.parse_args()
    if args.render:
        import os
        os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libGLEW.so"

    env = gym.make("InvertedPendulum-v2")

    if args.task == 'policy_gradient':
        # Define policy and value function
        hidden_dim_pol = 64
        hidden_depth_pol = 2
        hidden_dim_baseline = 64
        hidden_depth_baseline = 2
        policy = PGPolicy(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=hidden_dim_pol, hidden_depth=hidden_depth_pol)
        baseline = PGBaseline(env.observation_space.shape[0], hidden_dim=hidden_dim_baseline, hidden_depth=hidden_depth_baseline)
        policy.to(device)
        baseline.to(device)

        # Training hyperparameters
        num_epochs=200
        max_path_length=200
        batch_size=100
        gamma=0.99
        baseline_train_batch_size=64
        baseline_num_epochs=5
        print_freq=10

        if not args.test:
            # Train policy gradient
            simulate_policy_pg(env, policy, baseline, num_epochs=num_epochs, max_path_length=max_path_length, batch_size=batch_size,
                            gamma=gamma, baseline_train_batch_size=baseline_train_batch_size, device = device, baseline_num_epochs=baseline_num_epochs, print_freq=print_freq, render=args.render)
            torch.save(policy.state_dict(), 'pg_final.pth')
        else:
            print('loading pretrained pg')
            policy.load_state_dict(torch.load(f'pg_final.pth'))
        evaluate(env, policy,  num_validation_runs=100, episode_length=max_path_length, render=args.render)

    if args.task == 'actor_critic':
        # Define replay buffer
        hidden_dim = 64
        hidden_depth = 2

        obs_size = env.observation_space.shape[0]
        ac_size = env.action_space.shape[0]

        capacity=10000
        replay_buffer = ReplayBuffer(obs_size,
                                     ac_size,
                                     capacity,
                                     device)

        episode_length = 200
        num_epochs = 200
        discount = 0.99
        print_freq = 10
        num_update_steps = 100
        batch_size = 64

        policy = ACPolicy(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)

        if not args.test:
            # Train actor critic
            qf = QF(env.observation_space.shape[-1] + env.action_space.shape[-1], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)
            target_qf = TargetQF(env.observation_space.shape[-1] + env.action_space.shape[-1], hidden_dim=hidden_dim,
                                   hidden_depth=hidden_depth).to(device)
            simulate_policy_ac(env, policy, qf, target_qf, replay_buffer, device,
                               episode_length=episode_length,
                               num_epochs=num_epochs, batch_size=batch_size, num_update_steps=num_update_steps,
                               print_freq=print_freq,
                               render=args.render)
            torch.save(policy.state_dict(), 'ac_final.pth')
        else:
            print('loading pretrained ac')
            policy.load_state_dict(torch.load(f'ac_final.pth'))

        evaluate(env, policy,  num_validation_runs=100, episode_length=episode_length, render=args.render)
