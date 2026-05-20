import torch
import numpy as np
from torch import nn
from torch import optim
from utils import rollout, log_density

"""
Train one policy-gradient update using collected trajectory data.
Also trains the baseline network to predict returns.
"""
def train_model(
        policy,
        baseline,
        trajectories,
        policy_optim,
        baseline_optim,
        device,
        gamma=0.99,
        baseline_train_batch_size=64,
        baseline_num_epochs=5
):

# ---------[INITIALIZE]-------------------------------------------------------------------------------------------------

    states_all = [] # STATES
    actions_all = [] # ACTIONS
    returns_all = [] # DISCOUNTED SUM OF FUTURE REWARDS (RETURN TO GO)
    # log_probs_all = [] # LOG PROBABILITIES OF ACTIONS UNDER THE POLICY

    #### FLATTEN TRAJECTORIES INTO STATES, ACTIONS, REWARDS, RETURNS (TO GO) !!!!

    for traj in trajectories:
        states_traj = traj['state_arr'] # STATES !
        actions_traj = traj['action_arr'] # ACTIONS !
        rewards_traj = traj['reward_arr'] # REWARDS !
        returns_traj = np.zeros_like(rewards_traj) # DISCOUNTED RETURNS (TO GO) !
        # log_probs_traj = traj['log_probs'] # LOG PROBABILITIES OF ACTIONS UNDER THE POLICY !

        ###  CALCULATE AND INJECT RUNNING RETURN (TO GO) !!!

        running_return = 0.0
        for t in reversed(range(len(rewards_traj))):

            ## CALCULATE RUNNING RETURN (CURRENT REWARD + GAMMA * PRIOR SUM OF REWARDS) TO GO !!

            running_return = rewards_traj[t] + gamma * running_return # running_return is return to go

            ## INJECT RUNNING RETURN (TO GO) IN RETURNS TRAJECTORY !!

            returns_traj[t] = running_return # returns_traj[t] is the running return (return to go) from time step t

        states_all.append(states_traj) # append states
        actions_all.append(actions_traj) # append action_arr
        returns_all.append(returns_traj) # append discounted returns
        # log_probs_all.append(log_probs_traj) # append log probabilities of action_arr under the policy

    ## for NumPy arrays...
    # states = np.concatenate(states_all) # concatenate all states (into one NumPy array)
    # action_arr = np.concatenate(actions_all) # concatenate all action_arr (into one NumPy array)
    # returns = np.concatenate(returns_all) # concatenate all discounted returns (into one NumPy array)
    # returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize returns (zero mean, unit variance)
    # log_probs = np.concatenate(log_probs_all) # concatenate all log probabilities of action_arr under the policy (into one NumPy array)

    ## for PyTorch tensors...
    states = np.concatenate(states_all, axis=0)
    actions = np.concatenate(actions_all, axis=0)
    returns = np.concatenate(returns_all, axis=0)
    states = torch.tensor(states, dtype=torch.float32, device=device) # join all states
    actions = torch.tensor(actions, dtype=torch.float32, device=device) # join all action_arr
    returns = torch.tensor(returns, dtype=torch.float32, device=device) # join all returns

    returns = (returns - returns.mean()) / (returns.std() + 1e-8) # normalize returns (zero mean, unit variance)
    # COME BACK HERE
    # returns = returns.view(-1, 1) # ensure returns are shaped like baseline output

    # log_probs = torch.stack(log_probs_all).to(device) # join log probabilities

# ---------[TRAIN BASELINE]---------------------------------------------------------------------------------------

    #### TRAIN BASELINE NETWORK TO PREDICT RETURNS !!!!

    ### DEFINE NUMBER OF SAMPLE STATES !!!

    ## if states is a NumPy array:
    # num_samples = len(states)
    # idx_arr = np.arange(num_samples)

    ## if states is a PyTorch tensor:
    num_samples = states.shape[0]

    ### SHUFFLE & SAMPLE STATES; PASS TO BASELINE NEURAL NET; COMPARE BASELINE NEURAL NET PREDICTIONS TO REAL RETURNS !!!
    ## note: num_samples ~ number of rows (samples) in "states"

    for epoch in range(baseline_num_epochs): # baseline iterations ~ num of baselines (i.e. max training iterations per baseline)

        ## SHUFFLE INDICES FOR SAMPLING !!

        ## if states is a numpy array:
        # np.random.shuffle(idx_arr) # shuffle index array

        ## if states is a tensor:
        idx_arr = torch.randperm(num_samples, device=device) # shuffle index array; torch.randperm(5) returns rand permutation of [0, 1, 2, 3, 4]
        for i in range(num_samples // baseline_train_batch_size): # “How many full batches can I make from the total number of samples?”

            ## SAMPLE ON INDICES !!

            ## if states & action_arr are numpy arrays:
            # batch_indices = torch.LongTensor(idx_arr[(i) * baseline_train_batch_size : (i+1) * baseline_train_batch_size]).to(device) # get indices for current batch (from shuffled indices);  convert/send to device
            # batch_states = torch.from_numpy(states[batch_indices.cpu()]).float().to(device) # get shuffled states; convert/send to device
            # batch_returns = torch.from_numpy(returns[batch_indices.cpu()]).float().to(device) # get shuffled returns; convert/send to device

            ## if states & action_arr are tensors:
            batch_indices = idx_arr[(i) * baseline_train_batch_size : (i+1) * baseline_train_batch_size] # get indices for current batch (from shuffled indices)
            batch_states = states[batch_indices] # get shuffled states
            batch_returns = returns[batch_indices] # get shuffled returns

            ## PASS SAMPLE STATES TO BASELINE NEURAL NET (and rm final singleton dimension from output & batch_returns) !!

            batch_baseline = baseline(batch_states).squeeze(-1) # baseline chooses action
            batch_returns = batch_returns.squeeze(-1)

            ## COMPUTE SAMPLE LOSS (baseline predictions vs sample returns) !!

            baseline_loss = torch.nn.functional.mse_loss(batch_baseline, batch_returns) # normalize advantages (zero mean, unit variance); mse_loss ~ (x - x.mean()) / (x.std() + 1e-8)

            ## UPDATE BASELINE !!

            baseline_optim.zero_grad() # reset gradients to zero
            baseline_loss.backward() # compute new gradients from current loss
            baseline_optim.step() # update model weights using those ^ gradients

# ---------[TRAIN POLICY]-----------------------------------------------------------------------------------------

    #### TRAIN POLICY NETWORK TO OPTIMIZE SURROGATE OBJECTIVE !!!!

    """
    Pass states through the policy neural network to compute the current Gaussian policy distribution: mu, std, and log_std; ignore the newly sampled action (_). 
    Then log_density computes the log probability of the previously collected rollout action_arr under that distribution. 
    That log probability is used in the policy-gradient objective so PyTorch can backpropagate into the policy network.
    """

    ### CALCULATE POLICY LOG DISTRIBUTION PARAMETERS USING LOG_DENSITY FXN !!!
    ## >> policy(states): "how good did the current policy expect those states to be?"
    ## >> policy_log: "how likely did the current policy think those taken action_arr were?”

    ## if states & action_arr are numpy arrays:
    # _, std, log_std = policy(torch.from_numpy(states).float().to(device))
    # policy_log = log_density(torch.from_numpy(action_arr).float().to(device), policy.mu, std, log_std)

    ## if states & action_arr are tensors:
    _, std, log_std = policy(states)
    policy_log = log_density(actions, policy.mu, std, log_std)

    ### COMPUTE FINAL BASELINE RETURNS; COMPARE POLICY RETURNS !!!
    ## >> returns: "how good was the outcome from those states?"
    ## >> baseline(states): "how good did the baseline expect those states to be?”
    ## >> .detach(): do not build a gradient path, i.e. do not include in computation graph
    #           (otherwise, policy_loss.backward() will compute gradients for baseline network weights)

    with torch.no_grad(): # force torch to not build computation graph (alternative to .detach())
        baseline_results = baseline(states)
    advantages = returns - baseline_results # "was the actual result better or worse than expected?"
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # normalize advantages
    # COME BACK HERE

    ### COMPUTE LOSS (OUR SURROGATE OBJECTIVE) W/ POLICY LOG * ADVANTAGE !!

    policy_loss = -(policy_log * advantages).mean()

    ## UPDATE POLICY: step towards less  !!!

    policy_optim.zero_grad() # reset gradients to zero
    policy_loss.backward() # compute new gradients from current loss
    policy_optim.step() # update model weights using those ^ gradients

    ### OPTIONAL: DELETE VARIABLES TO FREE UP MEMORY !!!

    del states, actions, returns, states_all, actions_all, returns_all

def simulate_policy_pg(env, policy, baseline, num_epochs=200, max_path_length=200, batch_size=100,
                       gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5, print_freq=10, device = "cuda", render=False):
    
    policy_optim = optim.Adam(policy.parameters())
    baseline_optim = optim.Adam(baseline.parameters())

    history = {
        "episode": [],
        "avg_reward": [],
        "max_path_length": [],
        "policy_loss": [],
        "qf_loss": [],
    }

    for iter_num in range(num_epochs):
        sample_trajs = []
        for _ in range(batch_size):
            sample_traj = rollout(env,policy,device,episode_length=max_path_length)
            sample_trajs.append(sample_traj)

        #if len(sample_trajs) > 0:
        epoch_avg_reward = np.mean(np.asarray([traj['reward_arr'].sum() for traj in sample_trajs]))
        epoch_max_path_len = np.max(np.asarray([traj['reward_arr'].shape[0] for traj in sample_trajs]))

        history["episode"].append(iter_num)
        history["avg_reward"].append(epoch_avg_reward)
        history["max_path_length"].append(epoch_max_path_len)

        if iter_num % print_freq == 0:
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num,epoch_avg_reward,epoch_max_path_len,))

        # Logging returns occasionally
        #if iter_num % print_freq == 0:
        #    epoch_avg_reward = np.mean(np.asarray([traj['return'].sum() for traj in sample_trajs]))
        #    epoch_max_path_len = np.max(np.asarray([traj['reward_arr'].shape[0] for traj in sample_trajs]))
        #    print("Episode: {}, reward: {}, max path length: {}".format(iter_num, epoch_avg_reward, epoch_max_path_len))

        # Training model
        train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, device, gamma=gamma,
                    baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs)
    return history