"""
TODO: MODIFY TO FILL IN YOUR DAGGER IMPLEMENTATION
"""
import torch
import torch.optim as optim
import numpy as np

from utils import rollout, relabel_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_dagger(env, policy, expert_paths, expert_policy=None, num_epochs=500, episode_length=50,
                            batch_size=32, num_dagger_iters=10, num_trajs_per_dagger=10):

    # Fill in your dagger implementation in this function.

    # Hint: Loop through num_dagger_iters iterations, at each iteration train a policy on the current dataset.
    # Then rollout the policy, use relabel_action to relabel the actions along the trajectory with "expert_policy" and then add this to current dataset
    # Repeat this so the dataset grows with states drawn from the policy, and relabeled actions using the expert.

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

        idxs = np.array(range(len(trajs)))
        num_batches = len(idxs)*episode_length // batch_size
        # Train the model with Adam
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(num_batches):
                optimizer.zero_grad()
                # TODO start: Fill in your standard behavior cloning implementation here
                # Sample a minibatch of (obs, action) pairs from the current aggregated dataset,
                # compute the negative log-likelihood of the actions under the policy,
                # and assign it to `loss`.
                loss = None
                # TODO end
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            # if epoch % 10 == 0:
            print('[%d, %5d] loss: %.8f' %(epoch + 1, i + 1, running_loss))
            iter_losses.append(loss.item())
        losses.append(iter_losses)

        # Collecting more data for dagger
        trajs_recent = []
        for k in range(num_trajs_per_dagger):
            env.reset()
            # TODO start: Rollout the policy on the environment to collect more data, relabel them,
            #             and then add them into trajs_recent
            pass
            # TODO end

        trajs += trajs_recent
        mean_return = np.mean(np.array([traj['rewards'].sum() for traj in trajs_recent]))
        print("Average DAgger return is " + str(mean_return))
        returns.append(mean_return)
    return losses, returns
