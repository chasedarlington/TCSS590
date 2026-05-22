import torch
import numpy as np
from torch import optim
from utils import rollout, log_density
from csv_logger import CSVLogger

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
        discount=0.99,
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

            running_return = rewards_traj[t] + discount * running_return # running_return is return to go

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

def simulate_policy_pg(

        ## POLICY
        env, # gym.make("InvertedPendulum-v4", render_mode="human" if args.render else None)
        policy, # policy (PGPolicy) neural network
        baseline, # policy (PGBaseline) neural network

        ## HYPERPARAMETERS
        learning_rate = 0.001, # optim.Adam learning rate ~ step size for .backward() and .step()
        num_epochs = 200, # outer iterations; for training loop
        batch_size = 100, # inner iterations; rollouts/trajectories per epoch
        path_len_limit = 200, # maximum steps per rollout
        discount = 0.99, # discount factor; how much are future rewards worth right now?
        baseline_train_batch_size = 64, # number of rollout trajectories per epoch
        baseline_num_epochs = 5, # epochs for baseline

        ## SYSTEM & LOGGING
        print_freq = 10, # print frequency
        device = "cuda", # device (cuda or cpu)
        render = False, # render the gym.make env?
        csv_path = None # path for CSV log
):
    
    policy_optim = optim.Adam(policy.parameters(),lr=learning_rate)
    baseline_optim = optim.Adam(baseline.parameters(),lr=learning_rate)

    data_fields = [
        "env",
        "policy",
        "episode",
        "avg_reward",
        "max_path_length",
        "learning_rate",
        "num_epochs",
        "batch_size",
        "path_len_limit",
        "discount",
        "baseline_train_batch_size",
        "baseline_num_epochs"
    ]

    csv_log = CSVLogger(csv_path, data_fields) if csv_path else None # INITIALIZE

    if csv_log:
        csv_log = csv_log.__enter__() # ENTER: FILE OPEN

    try:

        ### ROLLOUT !!!

        for iter_num in range(num_epochs): # NUM EPOCHS
            sample_trajs = []
            for _ in range(batch_size): # NUM TRAJECTORIES
                sample_traj = rollout(env, policy, device, episode_length=path_len_limit)
                sample_trajs.append(sample_traj)

            ### LOGGING !!!

            ## CALCULATIONS !!

            episode_returns = np.fromiter((float(traj["reward_arr"].sum()) for traj in sample_trajs),dtype=np.float32)
            path_lengths = np.fromiter((traj["reward_arr"].shape[0] for traj in sample_trajs),dtype=np.int32)
            epoch_avg_reward = float(episode_returns.mean())
            epoch_max_path_len = int(path_lengths.max())

            ## PRINT !!

            if iter_num % print_freq == 0:
                print("Episode: {}, reward: {}, max path length: {}".format(iter_num,epoch_avg_reward,epoch_max_path_len,))

            ## ALTERNATE PRINT !!

            #if iter_num % print_freq == 0:
            #epoch_avg_reward = np.mean(np.asarray([traj['return'].sum() for traj in sample_trajs]))
            #epoch_max_path_len = np.max(np.asarray([traj['reward_arr'].shape[0] for traj in sample_trajs]))
            #    print("Episode: {}, reward: {}, max path length: {}".format(iter_num, epoch_avg_reward, epoch_max_path_len))

            ## TRAIN MODEL !!

            train_model(policy, baseline, sample_trajs, policy_optim, baseline_optim, device, discount=discount,
                                  baseline_train_batch_size=baseline_train_batch_size, baseline_num_epochs=baseline_num_epochs)

            ## WRITE TO LOG !!

            if csv_log:
                csv_log.write({
                    "env": env,
                    "policy": policy,
                    "episode": iter_num,
                    "avg_reward": epoch_avg_reward,
                    "max_path_length": epoch_max_path_len,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "path_len_limit": path_len_limit,
                    "discount": discount,
                    "baseline_train_batch_size": baseline_train_batch_size,
                    "baseline_num_epochs": baseline_num_epochs
                })
    finally:
        if csv_log:
            csv_log.__exit__(None, None, None) # TRY AND FINALLY W/ EXIT: FILE CLOSE
    return