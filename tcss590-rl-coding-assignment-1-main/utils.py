import torch
import torch.nn as nn
import math
import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import distributions as pyd
import torch.optim as optim
from torch.distributions import Categorical

# static values w/i .utils module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8 # epsilon! resolve log(0)

# feedforward neural network: configurable multi-layer perceptron (MLP) using pyTorch
def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)] # single linear transform: *each output neuron looks at all input values, multiplies them by learned weights, adds them up, then adds a bias*
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)] # linear transform + activate (ReLU)
        for i in range(hidden_depth - 1): # for each hidden layer: linear transform + activate (ReLU)
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim)) # final transform
    if output_mod is not None:
        mods.append(output_mod) # optional output modifier
    trunk = nn.Sequential(*mods) # pipe sequential layers
    return trunk # ready-to-use neural network

#class PolicyBase(nn.Module):
#    def __init__(self):
#        pass

#    def forward(self, state):
#        """
#        Input: state to perform forward inference
#        Output: (action_sample from policy distribution, log_prob of sampled action)
#        """
#       pass
#
#    def log_prob(self, state, action):
#        """
#        Input: state to perform forward inference, action to evaluate log probability
#        Output: Log probability of action distribution under the policy distribution using state
#        """
#        pass

class PolicyGaussian(nn.Module): # select ideal policy x% of the time, otherwise explore (state ? all action dimensions at once)

    def __init__(self, num_inputs, num_outputs, hidden_dim=65, hidden_depth=2): # initialize (see below)
        super(PolicyGaussian, self).__init__()
        self.trunk = mlp(num_inputs, hidden_dim, num_outputs*2, hidden_depth) # network outputs twice the action size; (1) mean (?) (2) log standard deviation (log ?)

    def forward(self, state):
        outs = self.trunk(state) # feed state through neural network
        mu, logstd = torch.split(outs, outs.shape[-1] // 2, dim=-1) # split self.trunk(state) into mu and log
        std = torch.exp(logstd) + EPS # convert log stdev to [normal] stdev
        ac_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), reinterpreted_batch_ndims=1) # one total log probability <- .Normal() creates a Gaussian per action dimension; .Independent() treats all dimensions as one joint distribution
        ac = ac_dist.sample() # generate exploratory action!
        return ac, ac_dist.log_prob(ac) # return explore action and log probability!

    def log_prob(self, state, action): # evaluate how likely a given action is!
        outs = self.trunk(state)
        mu, logstd = torch.split(outs, outs.shape[-1] // 2, dim=-1)
        std = torch.exp(logstd) + EPS
        ac_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), reinterpreted_batch_ndims=1)
        return ac_dist.log_prob(action)

class PolicyAutoRegressiveModel(nn.Module): # predict a multi-dimensional action one component at a time (each later action dimension depends on the earlier sampled dimensions; state ? action_dim_0 | state + action_dim_0 ? action_dim_1 | state + action_dim_0 + action_dim_1 ? action_dim_2 | ...)
    def __init__(self, num_inputs, num_outputs, hidden_dim=65, hidden_depth=2, num_buckets=10, ac_low=-1, ac_high=1):
        super(PolicyAutoRegressiveModel, self).__init__()
        self.eps = 1e-8

        # one neural network per action dimension (i.e. discretizes each action dimension into buckets : outputs num_buckets unnormalized log-probabilities)
        self.trunks = nn.ModuleList([mlp(num_inputs, hidden_dim, num_buckets, hidden_depth)] + [mlp(num_inputs + j + 1, hidden_dim, num_buckets, hidden_depth) for j in range(num_outputs - 1)])

        self.num_dims = num_outputs
        self.ac_low = torch.tensor(ac_low).to(device)
        self.ac_high = torch.tensor(ac_high).to(device)
        self.num_buckets = num_buckets
        self.bucket_size = torch.tensor((ac_high - ac_low) / num_buckets).to(device)

    def discretize(self, ac): # real-valued action -> bucket index
        bucket_idx = (ac - self.ac_low) // (self.bucket_size + self.eps)
        return torch.clip(bucket_idx, 0, self.num_buckets - 1)

    def undiscretize(self, bucket_idx, dimension): # bucket index -> action value (use center of bucket)
        return_val = bucket_idx[:, None]*self.bucket_size + self.ac_low + self.bucket_size*0.5
        return return_val[:, dimension]

    def forward(self, state): # for sampling a new action (explore)
        vals = []
        log_prob = 0 # optional: initialize as torch.zeros(state.shape[0], device=state.device)
        for j in range(self.num_dims): # build input to the j-th neural network; sample one dimension at a time
          if j == 0: # first dimension only uses state
            trunk_input = state
          else: # subsequent dimensions use state and prior actions
            prev_actions = torch.cat(vals, dim=-1) # map dependencies between action dimensions?
            trunk_input = torch.cat([state, prev_actions], dim=-1)
          logits = self.trunks[j](trunk_input) # get unnormalized log-probabilities for bucket probabilities
          distribution = Categorical(logits=logits) # logits -> distribution
          bucket_idx = distribution.sample() # distribution -> random sample bucket index
          log_prob += distribution.log_prob(bucket_idx) # accummulate log probability for that bucket index
          action_value = self.undiscretize(bucket_idx, j) # bucket index -> action value (use center of bucket) ### UNSQUEEZE?
          vals.append(action_value[:,None]) # collect sample action values!
        vals = torch.cat(vals, dim=-1)
        return vals, log_prob

    def log_prob(self, state, action): # for evaluating a known action: given curr state and [existing] action -> probability action on-policy?
        vals = []
        log_prob = 0. # optional: initialize as torch.zeros(state.shape[0], device=state.device)
        ac_discretized = self.discretize(action)
        for j in range(self.num_dims):
          if j == 0: # first dimension only uses state
            trunk_input = state
          else: # subsequent dimensions use state and prior actions
            prev_actions = action # use GIVEN action, and include dimensions according to j index
            trunk_input = torch.cat([state, prev_actions], dim=-1)
          logits = self.trunks[j](trunk_input) # get unnormalized log-probabilities (logits) for bucket probabilities
          distribution = Categorical(logits=logits) # logits -> distribution
          log_prob += distribution.log_prob(ac_discretized) # accummulate log probability for real-value bucket index
        return log_prob

def rollout(
        env,
        agent,
        agent_name, # Should be bc, dagger, pg
        episode_length=math.inf,
        render=False,
):
    # Collect the following data
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    dones = []
    images = []

    entropy = None
    log_prob = None
    agent_info = None
    path_length = 0

    reset_out = env.reset()
    o = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    if render:
        env.render()

    while path_length < episode_length:
        o_for_agent = o.copy()
        o_for_agent = torch.from_numpy(o_for_agent[None]).to(device).float()
        action, _ = agent(o_for_agent) # TODO: May need to convert to numpy
        action = action.cpu().detach().numpy()
        # Step the simulation forward
        step_action = action.squeeze() if action.ndim > 1 else action
        step_out = env.step(copy.deepcopy(step_action))
        if len(step_out) == 5:
            next_o, r, terminated, truncated, env_info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            next_o, r, done, env_info = step_out

        # Render the environment
        if render:
            env.render()

        raw_obs.append(o)
        raw_next_obs.append(next_o)
        actions.append(action)
        rewards.append(r)
        dones.append(done)
        path_length += 1
        if done:
            break
        o = next_o

    # Prepare the items to be returned
    observations = np.array(raw_obs)
    next_observations = np.array(raw_next_obs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    dones = np.array(dones).reshape(-1, 1)

    # Return in the following format
    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=np.array(dones).reshape(-1, 1),
        images = np.array(images)
    )

def generate_paths(env, expert_policy, episode_length, num_paths, file_path):
    # Initial data collection
    paths = []
    for j in range(num_paths):
        path = rollout(
            env,
            expert_policy,
            agent_name='bc',
            episode_length=episode_length,
            render=False)
        print("return is " + str(path['rewards'].sum()))
        paths.append(path)

    with open(file_path, 'wb') as fp:
        pickle.dump(paths, fp)
    print('Paths has been save to the file')

def get_expert_data(file_path):
    with open(file_path, 'rb') as fp:
        expert_data = pickle.load(fp)
    print('Imported Expert data successfully')
    return expert_data

def relabel_action(path, expert_policy):
    observation = path['observations']
    expert_action = expert_policy.get_action(observation)
    path['actions'] = expert_action[0]
    return path

def combine_sample_trajs(sample_trajs):
    assert len(sample_trajs) > 0

    my_dict = {k: [] for k in sample_trajs[0]}
    sample_trajs[0].keys()
    for sample_traj in sample_trajs:
        for key, value in sample_traj.items():
            my_dict[key].append(value)

    for key, value in my_dict.items():
        my_dict[key] = np.array(value)

    return my_dict
