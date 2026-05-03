import torch
import torch.optim as optim
import numpy as np

from utils import rollout, relabel_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_dagger(env, policy, expert_paths, expert_policy=None, num_epochs=500, episode_length=50,
                            batch_size=32, num_dagger_iters=10, num_trajs_per_dagger=10):

    # Optimizer code
    optimizer = optim.Adam(list(policy.parameters()))
    losses = []
    returns = []

    trajs = list(expert_paths)
    # Dagger iterations
    for dagger_itr in range(num_dagger_iters):
        iter_losses = []
        # Flatten current dataset for supervised training
        flat_obs = np.concatenate([t['observations'] for t in trajs])
        flat_ac = np.concatenate([t['actions'] for t in trajs])
        obs_all = torch.from_numpy(flat_obs).float().to(device)
        acs_all = torch.from_numpy(flat_ac).float().to(device)
        N = obs_all.shape[0]
        num_batches =  max(1, N // batch_size)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(num_batches):
                optimizer.zero_grad()
                idx = np.random.randint(0, N, batch_size)
                obs_batch = obs_all[idx]
                acs_batch = acs_all[idx]
                log_prob = policy.log_prob(obs_batch, acs_batch)
                loss = -log_prob.mean()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            # if epoch % 10 == 0:
            print('[%d, %5d] loss: %.8f' %(epoch + 1, i + 1, running_loss))
            iter_losses.append(loss.item())
        losses.append(iter_losses)

        # Collecting more data for dagger
        trajs_recent = []
        for k in range(num_trajs_per_dagger):
            traj = rollout(env, policy, episode_length)
            traj = relabel_action(traj, expert_policy)
            trajs_recent.append(traj)
        trajs += trajs_recent
        mean_return = np.mean(np.array([traj['rewards'].sum() for traj in trajs_recent]))
        print("Average DAgger return is " + str(mean_return))
        returns.append(mean_return)
    return losses, returns
