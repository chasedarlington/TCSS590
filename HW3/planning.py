import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rollout_model(
        model,
        initial_states,
        actions,
        horizon,
        reward_fn):
    # Collect the following data
    all_states = []
    all_rewards = []
    curr_state = initial_states # Starting from the initial state
    #========== TODO: start ==========
    # Hint1: concatenate current state and action pairs as the input for the model and predict the next observation
    # for horizon number of steps
    # Hint2: get the predicted reward using reward_fn()



    #========== TODO: end ==========
    all_states_full = torch.cat([state[:, None, :] for state in all_states], dim=1).cpu().detach().numpy()
    all_rewards_full = torch.cat(all_rewards, dim=-1).cpu().detach().numpy()
    return all_states_full, all_rewards_full


def get_ensemble_rewards(model, state_repeats, random_actions, horizon, reward_fn):
    """This method will generate the average reward rolled out over the ensemble of models

    Args:
        model List: The list of models
        state_repeats: The initial state repeated over the num random action sequence dimension
        random_actions: The random actions to be taken
        horizon: How long to roll out the model
        reward_fn: Used to get the reward
    """
    #========== TODO: start ==========
    # For each model in the list of models, rollout the model and get the rewards using the rollout_model
    # function. Take the mean of the rewards over each time step to get the average reward to return for
    # the passed action sequence. The output should be an array of the length of the number of random actions



    #========== TODO: end ==========
    return all_rewards


def plan_model_random_shooting(env, state, ac_size, horizon, model, reward_fn, n_samples_mpc=100):
    #========== TODO: start ==========
    # Hint1: randomly sample actions in the action space
    # Hint2: rollout model based on current state and random action using the rollout_model function.
    # Then select the best action that maximize the sum of the reward



    #========== TODO: end ==========
    return best_ac, random_actions[best_ac_idx]


def plan_model_mppi(env, state, ac_size, horizon, model, reward_fn, n_samples_mpc=100, n_iter_mppi=10, gaussian_noise_scales=[1.0, 1.0, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1, 0.01, 0.01]):
    assert len(gaussian_noise_scales) == n_iter_mppi
    # Rolling forward random actions through the model
    state_repeats = torch.from_numpy(np.repeat(state[None], n_samples_mpc, axis=0)).to(device)
    # Sampling random actions in the range of the action space
    random_actions = torch.FloatTensor(n_samples_mpc, horizon, ac_size).uniform_(env.action_space.low[0], env.action_space.high[0]).to(device).float()
    # Rolling forward through the mdoel for horizon steps
    if not isinstance(model, list):
        all_states, all_rewards = rollout_model(model, state_repeats, random_actions, horizon, reward_fn)
    else:
        # NOTE: Implement this branch in part 4 of the writeup, not in part 3. This is for the ensemble model
        # Use the get_ensemble_rewards function to get the rewards for the ensemble model
        pass


    all_returns = all_rewards.sum(axis=-1)
    # Take first action from best trajectory
    best_ac_idx = np.argmax(all_rewards.sum(axis=-1))
    best_ac = random_actions[best_ac_idx, 0] # Take the first action from the best trajectory

    # Run through a few iterations of MPPI

    for iter in range(n_iter_mppi):
        #========== TODO: start ==========
        # Hint1: Compute weights based on exponential of returns
        # Hint2: sample actions based on the weight, and compute average return over models
        # Hint3: if model type is a list, then implement ensemble mppi
        # Hint4: Refer to the psudeocode in the writeup for more details

        # Weight trajectories by exponential of returns


        # Rolling forward through the model for horizon steps (or ensemble) to update the rewards
        if not isinstance(model, list):
            # Rolling forward through the mdoel for horizon steps
            pass # Fill this in
        else:
            # NOTE: Implement this branch in part 4 of the writeup, not in part 3. This is for the ensemble model.
            # Use the get_ensemble_rewards function to get the rewards for the ensemble model
            pass
        #========== TODO: end ==========

    # Finally take first action from best trajectory
    best_ac_idx = np.argmax(all_rewards.sum(axis=-1))
    best_ac = random_actions[best_ac_idx, 0] # Take the first action from the best trajectory
    return best_ac, random_actions[best_ac_idx]
