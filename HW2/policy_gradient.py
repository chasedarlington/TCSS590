import torch
import numpy as np
from torch import nn
from torch import optim
from utils import rollout, log_density

# WHERE SHALL WE USE THIS? with torch.no_grad():
#       means: do not build computation graph... we are not using .backward() within

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
        states_traj = traj['states'] # STATES !
        actions_traj = traj['actions'] # ACTIONS !
        rewards_traj = traj['rewards'] # REWARDS !
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
        states_all.append(states_traj) # append states
        actions_all.append(actions_traj) # append actions
        returns_all.append(returns_traj) # append discounted returns
        # log_probs_all.append(log_probs_traj) # append log probabilities of actions under the policy

    ## for NumPy arrays...
    # states = np.concatenate(states_all) # concatenate all states (into one NumPy array)
    # actions = np.concatenate(actions_all) # concatenate all actions (into one NumPy array)
    # returns = np.concatenate(returns_all) # concatenate all discounted returns (into one NumPy array)
    # returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # normalize returns (zero mean, unit variance)
    # log_probs = np.concatenate(log_probs_all) # concatenate all log probabilities of actions under the policy (into one NumPy array)

    ## for PyTorch tensors...
    states = torch.tensor(states_all, dtype=torch.float32, device=device).to(device) # join all states
    actions = torch.tensor(actions_all, dtype=torch.float32, device=device).to(device) # join all actions
    returns = torch.tensor(returns_all, dtype=torch.float32, device=device).to(device) # join all returns

    # ????????????????

    # returns = torch.nn.functional.mse_loss(returns, returns.mean()) # normalize returns (zero mean, unit variance)

    # returns = returns.view(-1, 1) # ensure returns are shaped like baseline output

    # log_probs = torch.stack(log_probs_all).to(device) # join log probabilities

# ---------[TRAIN BASELINE]---------------------------------------------------------------------------------------

    #### TRAIN BASELINE NETWORK TO PREDICT RETURNS !!!!

    ### DEFINE NUMBER OF SAMPLE STATES !!!

    ## if states is a NumPy array:
    # num_samples = len(states)
    # indices = np.arange(num_samples)

    ## if states is a PyTorch tensor:
    num_samples = states.shape[0]

    ### SHUFFLE & SAMPLE STATES; PASS TO BASELINE NEURAL NET; COMPARE BASELINE NEURAL NET PREDICTIONS TO REAL RETURNS !!!

    for _ in range(baseline_num_epochs): # baseline iterations ~ num of baselines (i.e. max training iterations per baseline)

        ## SHUFFLE INDICES FOR SAMPLING !!

        ## if states is a numpy array:
        # np.random.shuffle(indices) # shuffle indices

        ## if states is a tensor:
        indices = torch.randperm(num_samples, device=device) # shuffle indices

        for i in range(num_samples // baseline_train_batch_size): # iterations per baseline ~ num of samples per baseline (i.e. max path length per training iteration)

            ## SAMPLE ON INDICES !!

            ## if states & actions are numpy arrays:
            # batch_indices = torch.LongTensor(indices[i : i + baseline_train_batch_size]).to(device) # get shuffled indices for batch;  convert/send to device
            # batch_states = torch.from_numpy(states[batch_indices.cpu()]).float().to(device) # get shuffled states; convert/send to device
            # batch_returns = torch.from_numpy(returns[batch_indices.cpu()]).float().to(device) # get shuffled returns; convert/send to device

            ## if states & actions are tensors:
            batch_indices = indices[i : i + baseline_train_batch_size] # get shuffled indices for batch
            batch_states = states[batch_indices] # get shuffled states
            batch_returns = returns[batch_indices] # get shuffled returns

            ## PASS SAMPLE STATES TO BASELINE NEURAL NET (and rm final singleton dimension from output & batch_returns) !!

            batch_baseline = baseline(batch_states).squeeze(-1) # baseline chooses action
            batch_returns = batch_returns.squeeze(-1)

            ## COMPUTE SAMPLE LOSS (baseline predictions vs sample returns) !!

            baseline_loss = torch.nn.functional.mse_loss(batch_baseline, batch_returns) # normalize advantages (zero mean, unit variance); mse_loss ~ (x - x.mean()) / (x.std() + 1e-8)

            ## UPDATE BASELINE !!

            baseline_optim.zero_grad()
            baseline_loss.backward()
            baseline_optim.step()

# ---------[TRAIN POLICY]-----------------------------------------------------------------------------------------

    #### TRAIN POLICY NETWORK TO OPTIMIZE SURROGATE OBJECTIVE !!!!

    """
    Pass states through the policy neural network to compute the current Gaussian policy distribution: mu, std, and log_std; ignore the newly sampled action (_). 
    Then log_density computes the log probability of the previously collected rollout actions under that distribution. 
    That log probability is used in the policy-gradient objective so PyTorch can backpropagate into the policy network.
    """

    ### CALCULATE POLICY LOG DISTRIBUTION PARAMETERS USING LOG_DENSITY FXN !!!
    ## >> policy(states): "how good did the current policy expect those states to be?"
    ## >> policy_log: "how likely did the current policy think those taken actions were?”

    ## if states & actions are numpy arrays:
    # _, std, log_std = policy(torch.from_numpy(states).float().to(device))
    # policy_log = log_density(torch.from_numpy(actions).float().to(device), policy.mu, std, log_std)

    ## if states & actions are tensors:
    _, std, log_std = policy(states)
    policy_log = log_density(actions, policy.mu, std, log_std)

    ### COMPUTE FINAL BASELINE RETURNS; COMPARE POLICY RETURNS !!!
    ## >> returns: "how good was the outcome from those states?"
    ## >> baseline(states): "how good did the baseline expect those states to be?”

    advantages = returns - baseline(states) # "was the actual result better or worse than expected?"

    ### COMPUTE LOSS (OUR SURROGATE OBJECTIVE) W/ POLICY LOG * ADVANTAGE !!
    ##

    policy_loss = -(policy_log * advantages).mean()

    ## UPDATE POLICY: step towards less  !!!

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    ### OPTIONAL: DELETE VARIABLES TO FREE UP MEMORY !!!

    del states, actions, returns, states_all, actions_all, returns_all

def simulate_policy_pg(env, policy, baseline, num_epochs=200, max_path_length=200, batch_size=100,
                       gamma=0.99, baseline_train_batch_size=64, baseline_num_epochs=5, print_freq=10, device = "cuda", render=False):
    
    policy_optim = optim.Adam(policy.parameters())
    baseline_optim = optim.Adam(baseline.parameters())

    for iter_num in range(num_epochs):
        sample_trajs = []
        for _ in range(batch_size):
            sample_traj = rollout(env,policy,device,episode_length=max_path_length)
            sample_trajs.append(sample_traj)

        # Logging returns occasionally
        if iter_num % print_freq == 0:
            rewards_np = np.mean(np.asarray([traj['rewards'].sum() for traj in sample_trajs]))
            path_length = np.max(np.asarray([traj['rewards'].shape[0] for traj in sample_trajs]))
            print("Episode: {}, reward: {}, max path length: {}".format(iter_num, rewards_np, path_length))

        # Training model
        train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, device, gamma=gamma,
                    baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs)
