import torch
import torch.optim as optim
import numpy as np
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
    idxs = np.arange(N)
    num_batches = N // batch_size
    losses = []
    for epoch in range(num_epochs):
        np.random.shuffle(idxs) # shuffle indices
        running_loss = 0.0 # baseline loss
        losses = []
        for i in range(num_batches): # for each batch
            optimizer.zero_grad() # zero the gradient
            #batch_idx = idxs[i*batch_size:(i+1)*batch_size] #  NECESSARY TO COMPUTE IDX?
            obs=obs_all[i] # note index
            acs=acs_all[i]
            log_prob = policy.log_prob(obs, acs)
            loss = -log_prob.mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            running_loss += loss.item()
        # if epoch % 10 == 0:
        #avg_loss = running_loss / num_batches
        print('[%d] running_loss: %.8f' %
            (epoch, running_loss / 10.))
        #print('[%d] avg_loss: %.8f' %
        #    (epoch, avg_loss / 10.))
    return losses
