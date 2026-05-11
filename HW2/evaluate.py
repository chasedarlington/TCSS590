"""
DO NOT MODIFY BESIDES HYPERPARAMETERS 
"""
import torch
import numpy as np

from utils import rollout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(env, policy, num_validation_runs=10, episode_length=200, render=False):
    success_count = 0
    rewards_suc = 0
    rewards_all = 0
    for k in range(num_validation_runs):
        o = env.reset()
        path = rollout(
                env,
                policy,
                episode_length=episode_length,
                render=render)
        success = len(path['dones']) == episode_length

        if success:
            success_count += 1
            rewards_suc += np.sum(path['rewards'])
        rewards_all += np.sum(path['rewards'])
        print(f"test {k}, success {success}, reward {np.sum(path['rewards'])}")
    print("Success rate: ", success_count/num_validation_runs)
    print("Average reward (success only): ", rewards_suc/max(success_count, 1))
    print("Average reward (all): ", rewards_all/num_validation_runs)