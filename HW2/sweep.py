import gymnasium as gym
import numpy as np
import torch
from sweep_pg import simulate_policy_pg
from sweep_ac import simulate_policy_ac, ReplayBuffer
from utils import ACPolicy, QF, TargetQF, PGPolicy, PGBaseline
import random
from datetime import datetime

## DEVICE AND ENV SETUP !!
device = torch.device('cpu')
env = gym.make("InvertedPendulum-v4", render_mode=None)

## NEURAL NET SETUP !!
hidden_dim = 64
hidden_depth = 2

## REPLAY BUFFER SETUP !!
obs_size = env.observation_space.shape[0]
ac_size = env.action_space.shape[0]
capacity = 10000
replay_buffer = ReplayBuffer(obs_size, ac_size, capacity, device)

## POLICY DEFINITIONS !!
pg_policy = PGPolicy(num_inputs=env.observation_space.shape[0], num_outputs=env.action_space.shape[0], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)
pg_baseline = PGBaseline(env.observation_space.shape[0], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)
ac_policy = ACPolicy(env.observation_space.shape[0], env.action_space.shape[0], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)
qf = QF(env.observation_space.shape[-1] + env.action_space.shape[-1], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)
target_qf = TargetQF(env.observation_space.shape[-1] + env.action_space.shape[-1], hidden_dim=hidden_dim, hidden_depth=hidden_depth).to(device)

#### SWEEP !!!!

### SWEEP PARAMETERS !!!
seeds = [0, 1, 2]
normalize_sweep = [True, False]
learning_rate_sweep = [1e-4, 3e-4, 1e-3]
num_epochs_sweep = [100, 200, 500]
path_len_limit_sweep = [200]
batch_size_sweep = [64]
discount_sweep = [0.99]
policy_sweep = ['pg','ac']

## PG !!
baseline_train_batch_size_sweep = [64]
baseline_num_epochs_sweep = [100]

## AC !!
ac_update_steps_sweep = [100]

for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for normalize in normalize_sweep:
        for learning_rate in learning_rate_sweep:
            for num_epochs in num_epochs_sweep:
                for path_len_limit in path_len_limit_sweep:
                    for batch_size in batch_size_sweep:
                        for discount in discount_sweep:
                            for policy_name in policy_sweep:
                                if(policy_name=='pg'):
                                    run_id = f"{policy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    for baseline_train_batch_size in baseline_train_batch_size_sweep:
                                        for baseline_num_epochs in baseline_num_epochs_sweep:
                                            simulate_policy_pg(
                                                run_id, seed, env, pg_policy, pg_baseline, normalize=normalize, learning_rate=learning_rate,
                                                num_epochs=num_epochs,path_len_limit=path_len_limit, batch_size=batch_size, discount=discount,
                                                baseline_train_batch_size=baseline_train_batch_size, device=device, baseline_num_epochs=baseline_num_epochs,
                                                csv_path="logs/policy_gradient_sweep_metrics.csv"
                                            )

                                else:
                                    run_id = f"{policy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                    for ac_update_steps in ac_update_steps_sweep:
                                            simulate_policy_ac(
                                                run_id, seed, env, ac_policy, qf, target_qf, replay_buffer,
                                                device, path_len_limit=path_len_limit,
                                                num_epochs=num_epochs, batch_size=batch_size,
                                                ac_update_steps=ac_update_steps,
                                                csv_path="logs/actor_critic_sweep_metrics.csv"
                                            )
