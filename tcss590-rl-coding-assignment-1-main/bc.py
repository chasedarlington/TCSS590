"""
TODO: MODIFY TO FILL IN YOUR BC IMPLEMENTATION
"""
import torch
import torch.optim as optim
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50,
                       batch_size=32):

    # Fill in your BC implementation in this function.

    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy.
    flattened = {'observations': [], 'actions': []}
    for path in expert_data:
        for k in flattened.keys():
            flattened[k].append(path[k])
    for k in flattened.keys():
        flattened[k] = np.concatenate(flattened[k])

    obs_all = torch.from_numpy(flattened['observations']).float().to(device)
    acs_all = torch.from_numpy(flattened['actions']).float().to(device)
    N = obs_all.shape[0]

    optimizer = optim.Adam(list(policy.parameters()), lr=1e-4)
    idxs = np.array(range(len(expert_data)))
    num_batches = len(idxs)*episode_length // batch_size
    losses = []
    for epoch in range(num_epochs):
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            # TODO start: Fill in your behavior cloning implementation here, just maximize log likelihood!
            # Sample a minibatch of (obs, action) pairs, compute the negative log-likelihood
            # of the actions under the policy, and assign it to `loss`.
            loss = None
            # TODO end
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # if epoch % 10 == 0:
        print('[%d] loss: %.8f' %
            (epoch, running_loss / 10.))
        losses.append(loss.item())
    return losses
