"""
DO NOT MODIFY BESIDES HYPERPARAMETERS
"""
import math
import torch
import numpy as np

from planning import plan_model_mppi, plan_model_random_shooting
import copy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def planning_agent(env, o_for_agent, model, reward_fn, plan_mode, mpc_horizon=None, n_samples_mpc=None):
    if plan_mode == 'random':
        # Taking random actions
        action = torch.Tensor(env.action_space.sample()[None]).to(device)
    elif plan_mode == 'random_mpc':
        # Taking actions via random shooting + MPC
        action, _ = plan_model_random_shooting(env, o_for_agent, env.action_space.shape[0], mpc_horizon, model,
                                               reward_fn, n_samples_mpc=n_samples_mpc)
    elif plan_mode == 'mppi':
        action, _ = plan_model_mppi(env, o_for_agent, env.action_space.shape[0], mpc_horizon, model, reward_fn,
                                    n_samples_mpc=n_samples_mpc)
    else:
        raise NotImplementedError("Other planning methods not implemented")
    return action

def collect_traj_MBRL(
        env,
        model,
        plan_mode,
        replay_buffer=None,
        device=device,
        episode_length=math.inf,
        reward_fn=None, #Reward function to evaluate
        render=False,
        mpc_horizon=None,
        n_samples_mpc=None
):
    # Collect the following data
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    dones = []
    images = []

    path_length = 0
    o, _ = env.reset()
    if render:
        env.render()
    with torch.no_grad():
        while path_length < episode_length:
            o_for_agent = o

            # Using the planning agent to take actions
            action = planning_agent(env, o_for_agent, model, reward_fn, plan_mode, mpc_horizon=mpc_horizon, n_samples_mpc=n_samples_mpc)
            if len(action.shape) == 1:
                action = action.unsqueeze(0)
            action = action.cpu().detach().numpy()[0]

            # Step the simulation forward
            next_o, r, done, trunc, env_info = env.step(copy.deepcopy(action))
            done = done or trunc
            if replay_buffer is not None:
                replay_buffer.add(o,
                                action,
                                r,
                                next_o,
                                done)
            # Render the environment
            if render:
                env.render()

            raw_obs.append(o)
            raw_next_obs.append(next_o)
            actions.append(action)
            rewards.append(r)
            dones.append(done)
            path_length += 1
            if done:
                break
            o = next_o

    # Prepare the items to be returned
    observations = np.array(raw_obs)
    next_observations = np.array(raw_next_obs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    dones = np.array(dones).reshape(-1, 1)

    # Return in the following format
    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=np.array(dones).reshape(-1, 1),
        images=np.array(images)
    )



def evaluate(env, model, plan_mode, num_validation_runs=10, episode_length=200, render=False, mpc_horizon=None, n_samples_mpc=None, reward_fn=None):
    success_count = 0
    rewards_suc = 0
    rewards_all = 0
    for k in range(num_validation_runs):
        env.reset()
        path = collect_traj_MBRL(
            env,
            model,
            plan_mode,
            episode_length=episode_length,
            render=render,
            mpc_horizon=mpc_horizon,
            n_samples_mpc=n_samples_mpc,
            device=device,
            reward_fn=reward_fn
        )
        success = np.linalg.norm(env.unwrapped.get_body_com("fingertip") - env.unwrapped.get_body_com("target")) < 0.1

        if success:
            success_count += 1
            rewards_suc += np.sum(path['rewards'])
        rewards_all += np.sum(path['rewards'])
        print(f"test {k}, success {success}, reward {np.sum(path['rewards'])}")
    print("Success rate: ", success_count/num_validation_runs)
    print("Average reward (success only): ", rewards_suc/max(success_count, 1))
    print("Average reward (all): ", rewards_all/num_validation_runs)
