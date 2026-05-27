import torch
from torch import optim
import torch.nn.functional as F
import copy
import numpy as np
from utils import collect_trajs
from csv_logger import CSVLogger
import os

file_name = os.path.basename(__file__)

class ReplayBuffer(object):
    """Buffer to store environment transitions."""

    def __init__(self, obs_size, action_size, capacity, device):
        self.capacity = capacity
        self.device = device
        self.state_arr = np.empty((capacity, obs_size), dtype=np.float32)
        self.next_state_arr = np.empty((capacity, obs_size), dtype=np.float32)
        self.action_arr = np.empty((capacity, action_size), dtype=np.float32)
        self.reward_arr = np.empty((capacity, 1), dtype=np.float32)
        self.not_done_arr = np.empty((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.last_save = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, state, action, reward, next_state, done):
        state = np.asarray(state, dtype=np.float32).reshape(-1, self.state_arr.shape[1])
        action = np.asarray(action, dtype=np.float32).reshape(-1, self.action_arr.shape[1])
        reward = np.asarray(reward, dtype=np.float32).reshape(-1, 1)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(-1, self.next_state_arr.shape[1])
        done = np.asarray(done, dtype=np.float32).reshape(-1, 1)

        num_samples = state.shape[0]
        if not (action.shape[0] == reward.shape[0] == next_state.shape[0] == done.shape[0] == num_samples):
            raise ValueError("ReplayBuffer.add received inputs with different batch sizes")

        idx_arr = np.arange(self.idx, self.idx + num_samples) % self.capacity
        self.state_arr[idx_arr] = copy.deepcopy(state)
        self.action_arr[idx_arr] = copy.deepcopy(action)
        self.reward_arr[idx_arr] = copy.deepcopy(reward)
        self.next_state_arr[idx_arr] = copy.deepcopy(next_state)
        self.not_done_arr[idx_arr] = 1.0 - done

        self.full = self.full or (self.idx + num_samples >= self.capacity)
        self.idx = (self.idx + num_samples) % self.capacity

    def sample(self, batch_size):
        idx_arr = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)
        state_arr = torch.as_tensor(self.state_arr[idx_arr], device=self.device).float()
        action_arr = torch.as_tensor(self.action_arr[idx_arr], device=self.device)
        reward_arr = torch.as_tensor(self.reward_arr[idx_arr], device=self.device)
        next_state_arr = torch.as_tensor(self.next_state_arr[idx_arr],
                                     device=self.device).float()
        not_done_arr = torch.as_tensor(self.not_done_arr[idx_arr], device=self.device)

        return state_arr, action_arr, reward_arr, next_state_arr, not_done_arr

def compute_losses(policy, qf, target_qf, state_arr, action_arr, reward_arr, next_state_arr, not_done_arr, device, discount=0.99):

    ## FORMAT STATES, ACTIONS, REWARDS, NEXT STATES, AND NOT DONE - PULL NUMPY ARRAYS AS TENSORS
    state_arr = torch.as_tensor(state_arr, device=device).float()
    action_arr = torch.as_tensor(action_arr, device=device).float()
    reward_arr = torch.as_tensor(reward_arr, device=device).float()
    next_state_arr = torch.as_tensor(next_state_arr, device=device).float()
    not_done_arr = torch.as_tensor(not_done_arr, device=device).float()

    ## COMPUTE POLICY LOSS
    a_sampled_t, _, _ = policy(state_arr) # Get (differentiable) action samples a_sampled_t from the policy using policy.forward
    q_pi = qf(torch.cat([state_arr, a_sampled_t], dim=1)) # Score them with the current Q-function
    policy_loss = -q_pi.mean() # Maximise Q  →  minimise its negation

    ## COMPUTE Q FUNCTION LOSS
    q_predictions = qf(torch.cat([state_arr, action_arr], dim=1)) # Compute q predictions
    with torch.no_grad():
        next_a_t, _, _ = policy(next_state_arr)
        q_next = target_qf(torch.cat([next_state_arr, next_a_t], dim=1))
        q_targets = reward_arr + not_done_arr * discount * q_next # Compute q targets
    qf_loss = F.mse_loss(q_predictions, q_targets)  # compute bellman error

    return policy_loss, qf_loss

def soft_update_target(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def simulate_policy_ac(

        ## POLICY
        run_id,
        seed,
        env, # gym.make("InvertedPendulum-v4", render_mode="human" if args.render else None)
        policy, # policy (ACPolicy) neural network
        qf, # policy (QF) neural network
        target_qf, # policy (targetQF) neural network
        replay_buffer, # ReplayBuffer(obs_size, ac_size, capacity, device)

        ## HYPERPARAMETERS
        learning_rate = 3e-4, # optim.Adam learning rate ~ step size for .backward() and .step()
        num_epochs: int = 200, # number of outer iterations for training loop
        batch_size = 32, # number of rollout trajectories, and repeats per replay buffer
        path_len_limit: int = 100, # maximum steps per rollout
        discount = 0.99, # discount factor; how much are future rewards worth right now?
        target_weight = 5e-3, # soft update rate; how quickly does target_qf move towards qf?
        ac_update_steps = 100, # number of gradient update rounds per epoch; how many update rounds per epoch?

        ## SYSTEM AND LOGGING
        print_freq = 10,  # print frequency
        device = "cuda",  # device (cuda or cpu)
        render = False,  # render the gym.make env?
        csv_path = None  # path for CSV log
):

    env.reset()

    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    qf_optimizer = optim.Adam(qf.parameters(), lr=learning_rate)

    # Copy parameters initially
    soft_update_target(qf, target_qf, 1.0)

    data_fields = [
        "env",
        "policy",
        "run_id",
        "seed",
        "epoch",
        "avg_reward",
        "max_path_length",
        "avg_policy_loss",
        "avg_qf_loss",
        "learning_rate",
        "num_epochs",
        "batch_size",
        "path_len_limit",
        "discount",
        "target_weight",
        "ac_update_steps"
    ]

    csv_log = CSVLogger(csv_path, data_fields) if csv_path else None  # INITIALIZE

    if csv_log:
        csv_log = csv_log.__enter__() # ENTER: FILE OPEN

    try:

        ### ROLLOUT !!!

        for iter_num in range(num_epochs): # NUM EPOCHS
            sample_trajs = []
            for _ in range(batch_size): # NUM TRAJECTORIES
                sample_traj = collect_trajs(env, policy, replay_buffer, device, episode_length=path_len_limit,render=render) # EQUIV TO ROLLOUT!
                sample_trajs.append(sample_traj)

            epoch_policy_losses = []
            epoch_qf_losses = []

            for update_num in range(ac_update_steps):
                state_arr, action_arr, reward_arr, next_state_arr, not_done_arr = replay_buffer.sample(batch_size)

                policy_loss, qf_loss = compute_losses(policy, qf, target_qf, state_arr,
                                                      action_arr, reward_arr, next_state_arr,
                                                      not_done_arr, device, discount)

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                qf_optimizer.zero_grad()
                qf_loss.backward()
                qf_optimizer.step()

                soft_update_target(qf, target_qf, target_weight)

                epoch_policy_losses.append(policy_loss.item())
                epoch_qf_losses.append(qf_loss.item())

            ### LOGGING !!!

            ## CALCULATIONS !!

            episode_returns = np.fromiter((float(traj["reward_arr"].sum()) for traj in sample_trajs),dtype=np.float32)
            path_lengths = np.fromiter((traj["reward_arr"].shape[0] for traj in sample_trajs), dtype=np.int32)
            epoch_avg_reward = float(episode_returns.mean())
            epoch_max_path_len = int(path_lengths.max())
            epoch_avg_p_loss = float(np.mean(epoch_policy_losses))
            epoch_avg_q_loss = float(np.mean(epoch_qf_losses))

            ## PRINT !!

            if iter_num % print_freq == 0:
                print(
                    "Episode: {}, reward: {}, max path length: {}, policy loss: {:.4f}, qf loss: {:.4f}".format(
                        iter_num,
                        epoch_avg_reward,
                        epoch_max_path_len,
                        epoch_avg_p_loss,
                        epoch_avg_q_loss
                    )
                )

            ## WRITE TO LOG !!

            if csv_log:
                csv_log.write({
                    "env": env.spec.id,
                    "policy": file_name,
                    "run_id": run_id,
                    "seed": seed,
                    "epoch": iter_num,
                    "avg_reward": epoch_avg_reward,
                    "max_path_length": epoch_max_path_len,
                    "avg_policy_loss": epoch_avg_p_loss,
                    "avg_qf_loss": epoch_avg_q_loss,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "path_len_limit": path_len_limit,
                    "discount": discount,
                    "target_weight": target_weight,
                    "ac_update_steps": ac_update_steps
                })
    finally:
        if csv_log:
            csv_log.__exit__(None, None, None)  # TRY AND FINALLY W/ EXIT: FILE CLOSE