import torch
import torch.optim as optim
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50,
                       batch_size=32):

    # --------------------------
    # FLATTEN EXPERT DATASET
    # --------------------------
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
    idxs = np.array(range(len(expert_data))) #idxs = np.arange(N) SIMPLER
    num_batches = len(idxs)*episode_length // batch_size #num_batches = N // batch_size SIMPLER
    losses = []
    for epoch in range(num_epochs):
        np.random.shuffle(idxs) # shuffle indices
        running_loss = 0.0 # baseline loss
        for i in range(num_batches): # for each batch
            optimizer.zero_grad() # zero the gradient
            batch_idx = idxs[i*batch_size:(i+1)*batch_size] #  NECESSARY TO COMPUTE IDX?
            obs=obs_all[i] # note index
            acs=acs_all[i]
            log_prob = policy.log_prob(obs, acs)
            loss = -log_prob.mean()
            # TODO start: Fill in your behavior cloning implementation here,
            # just maximize log likelihood!
            # Sample a minibatch of (obs, action) pairs, compute the negative
            # log-likelihood
            # of the actions under the policy, and assign it to `loss`.
            # TODO end
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # if epoch % 10 == 0:
        # avg_loss = running_loss / num_batches
        print('[%d] loss: %.8f' %
            (epoch, running_loss / 10.))
        losses.append(loss.item())
    return losses
