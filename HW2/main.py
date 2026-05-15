import gymnasium as gym
import numpy as np
import torch
import argparse
# from policy_gradient import simulate_policy_pg
from policy_gradient_no_comments import simulate_policy_pg
from actor_critic import simulate_policy_ac, ReplayBuffer
from utils import ACPolicy, QF, TargetQF, PGPolicy, PGBaseline
from evaluate import evaluate
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# main arguments: task, test, render, and double_q 
#   task: policy_gradient or actor_critic
#   test: True to use .pth, False to train and save .pth
#   render: True to display environment, False to not display environment
#   double_q: True to use double Q-learning in actor-critic, False to not use double Q-learning in actor-critic

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

    env = gym.make("InvertedPendulum-v4", render_mode="human" if args.render else None) # trying v4 instead of v2
    
    print(type(env))
    print(env)
    print(type(env.unwrapped))
    print(env.unwrapped)

    if args.task == 'policy_gradient':

        # NEURAL NET ARCHITECTURE (POLICY + BASELINE)
        hidden_dim_pol = 64 
        hidden_depth_pol = 2 
        hidden_dim_baseline = 64
        hidden_depth_baseline = 2

        # PG POLICY: self.trunk = mlp(num_inputs, hidden_dim, num_outputs*2, hidden_depth)
        #   where NUM_INPUTS: input size, HIDDEN_DIM: hidden layer size, NUM_OUTPUTS: output size, HIDDEN_DEPTH: number of hidden layers
        policy = PGPolicy(num_inputs=env.observation_space.shape[0], 
                          num_outputs=env.action_space.shape[0], 
                          hidden_dim=hidden_dim_pol, 
                          hidden_depth=hidden_depth_pol).to(device)
        
        # PG BASELINE: self.trunk = mlp(num_inputs, hidden_dim, 1, hidden_depth)
        #   where NUM_INPUTS: input size, HIDDEN_DIM: hidden layer size, NUM_OUTPUTS: output size, HIDDEN_DEPTH: number of hidden layers
        baseline = PGBaseline(env.observation_space.shape[0], 
                              hidden_dim=hidden_dim_baseline, 
                              hidden_depth=hidden_depth_baseline).to(device)

        # HYPERPARAMETERS
        num_epochs=200
        max_path_length=200
        batch_size=100
        gamma=0.99
        baseline_train_batch_size=64
        baseline_num_epochs=5
        print_freq=10
        eval_ep_count=100

        # TRAIN: run policy in env, train/update policy, and use baseline netowrk for stability
        #   then save trained policy weights (in .pth file)
        if not args.test:
            
            # SIMULATE: run the policy gradient training loop; update policy + baseline
            simulate_policy_pg(env, policy, baseline, num_epochs=num_epochs, max_path_length=max_path_length, 
                            batch_size=batch_size, gamma=gamma, baseline_train_batch_size=baseline_train_batch_size, 
                            device=device, baseline_num_epochs=baseline_num_epochs, print_freq=print_freq, render=args.render)
            
            # SAVE: save trained policy weights (in .pth file)
            torch.save(policy.state_dict(), 'pg_final.pth')
        
        # TEST: load .pth
        else:
            policy.load_state_dict(torch.load(f'pg_final.pth'))
        
        # EVALUATE: Run the policy in the environment x times, for up to max_path_length steps each time, optionally render the environment
        evaluate(env, policy,  num_validation_runs=eval_ep_count, episode_length=max_path_length, render=args.render)

    if args.task == 'actor_critic':

        # REPLAY BUFFER
        obs_size = env.observation_space.shape[0]
        ac_size = env.action_space.shape[0]
        capacity=10000  
        replay_buffer = ReplayBuffer(obs_size, ac_size, capacity, device)

        # NEURAL NET ARCHITECTURE (POLICY)
        hidden_dim = 64
        hidden_depth = 2

        # HYPERPARAMETERS
        discount = 0.99
        max_path_length = 200
        num_epochs = 200
        batch_size = 64
        num_update_steps = 100
        print_freq = 10
        eval_ep_count=100

        # ACTOR CRITIC POLICY: self.trunk = mlp(num_inputs, hidden_dim, num_outputs*2, hidden_depth)
        policy = ACPolicy(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)

        # TRAIN: run policy in env, store exp in replay buffer, train Q-network (critic) + update policy, and use target Q-network for stability 
        #   then save trained policy weights (in .pth file)
        if not args.test:
            
            # DEFINE: main Q-network + duplicate for computing target values in critic update
            qf = QF(env.observation_space.shape[-1] + env.action_space.shape[-1], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)
            target_qf = TargetQF(env.observation_space.shape[-1] + env.action_space.shape[-1], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)
            
            # SIMULATE: run the actor-critic training loop; update policy + Q-networks, and use replay buffer to store experience 
            simulate_policy_ac(env, policy, qf, target_qf, replay_buffer, device,
                               episode_length=max_path_length, num_epochs=num_epochs, 
                               batch_size=batch_size, num_update_steps=num_update_steps,
                               print_freq=print_freq, render=args.render)
            
            # SAVE: save trained policy weights (in .pth file)
            torch.save(policy.state_dict(), 'ac_final.pth')
        
        # TEST: load .pth         
        else:
            policy.load_state_dict(torch.load(f'ac_final.pth'))

        # EVALUATE: Run the policy in the environment x times, for up to max_path_length steps each time, optionally render the environment
        evaluate(env, policy,  num_validation_runs=eval_ep_count, episode_length=episode_length, render=args.render)
