"""
DO NOT MODIFY BESIDES HYPERPARAMETERS 
"""
import torch
import numpy as np
from utils import rollout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(env, policy, device,num_validation_runs, episode_length, render=False):
    success_count = 0
    rewards_suc = 0
    rewards_all = 0
    for k in range(num_validation_runs):
        o, info = env.reset() # (+info) UPDATED TO REFLECT NEW GYM API
        path = rollout(
                env,
                policy,
                device,
                episode_length=episode_length,
                render=render)
        success = len(path['done_arr']) == episode_length

        if success:
            success_count += 1
            rewards_suc += np.sum(path['reward_arr'])
        rewards_all += np.sum(path['reward_arr'])
        print(f"test {k}, success {success}, reward {np.sum(path['reward_arr'])}")
    print("Success rate: ", success_count/num_validation_runs)
    print("Average reward (success only): ", rewards_suc/max(success_count, 1))
    print("Average reward (all): ", rewards_all/num_validation_runs)
