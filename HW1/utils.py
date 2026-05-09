import torch
import torch.nn as nn
import math
import copy
import numpy as np
import pickle
#import matplotlib.pyplot as plt
#import torch.nn.functional as F
from torch import distributions as pyd
#import torch.optim as optim
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
    return nn.Sequential(*mods) # sequential layers -> ready-to-use neural network

def ensure_2d_tensor(x, expected_last_dim=None, name="tensor"):
    """
    Convert x to a 2D torch tensor.

    Valid inputs:
        shape (D,)    -> shape (1, D)
        shape (N, D)  -> shape (N, D)

    This prevents errors like:
        IndexError: too many indices for tensor of dimension 1
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)

    if x.ndim == 1:
        x = x.unsqueeze(0)

    if x.ndim != 2:
        raise ValueError(
            f"{name} must be 1D or 2D, but got shape {tuple(x.shape)}"
        )

    if expected_last_dim is not None and x.shape[-1] != expected_last_dim:
        raise ValueError(
            f"{name} last dimension must be {expected_last_dim}, "
            f"but got shape {tuple(x.shape)}"
        )

    return x.float()

class PolicyGaussian(nn.Module): # select ideal policy x% of the time, otherwise explore (state → all action dimensions at once)
    def __init__(self, num_inputs, num_outputs, hidden_dim=65, hidden_depth=2): # initialize (see below)
        #super(PolicyGaussian, self).__init__()
        super().__init__() # experiment
        self.num_inputs = num_inputs #test
        self.num_outputs = num_outputs #test
        self.trunk = mlp(num_inputs, hidden_dim, num_outputs*2, hidden_depth) # network outputs twice the action size; (1) mean (μ) (2) log standard deviation (log σ)

    def forward(self, state):
        state = ensure_2d_tensor(state, self.num_inputs, name = "state") #unit test
        outs = self.trunk(state) # feed state through neural network
        mu, logstd = torch.split(outs, outs.shape[-1] // 2, dim=-1) # split self.trunk(state) into mu and log
        std = torch.exp(logstd) + EPS # convert log stdev to [normal] stdev
        ac_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), reinterpreted_batch_ndims=1) # one total log probability <- .Normal() creates a Gaussian per action dimension; .Independent() treats all dimensions as one joint distribution
        ac = ac_dist.sample() # generate exploratory action!
        return ac, ac_dist.log_prob(ac) # return explore action and log probability!

    def log_prob(self, state, action): # evaluate how likely a given action is!
        state = ensure_2d_tensor(state, self.num_inputs, name="state") #unit test
        action = ensure_2d_tensor(action, self.num_outputs, name="action") #unit test
        outs = self.trunk(state)
        mu, logstd = torch.split(outs, outs.shape[-1] // 2, dim=-1)
        std = torch.exp(logstd) + EPS
        ac_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), reinterpreted_batch_ndims=1)
        return ac_dist.log_prob(action)


class PolicyAutoRegressiveModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=65, hidden_depth=2, num_buckets=10, ac_low=-1, ac_high=1):
        super().__init__() #experiment
        #super(PolicyAutoRegressiveModel, self).__init__()
        self.eps = 1e-8
        self.num_inputs = num_inputs #test
        self.num_dims = num_outputs #test
        self.num_buckets = num_buckets #test

        self.trunks = nn.ModuleList(
            [mlp(num_inputs, hidden_dim, num_buckets, hidden_depth)] +
            [mlp(num_inputs + j + 1, hidden_dim, num_buckets, hidden_depth)
             for j in range(num_outputs - 1)]
        )

        self.ac_low = torch.tensor(ac_low).to(device)
        self.ac_high = torch.tensor(ac_high).to(device)
        #self.ac_low = torch.tensor(ac_low, dtype=torch.float32, device=device) #experimetn
        #self.ac_high = torch.tensor(ac_high, dtype=torch.float32, device=device) #experimetn
        #self.num_dims = num_outputs
        if self.ac_low.ndim == 0: # unit test
            self.ac_low = self.ac_low.repeat(num_outputs)
        if self.ac_high.ndim == 0: # unit test
            self.ac_high = self.ac_high.repeat(num_outputs)
        if self.ac_low.shape[-1] != num_outputs: # unit test
            raise ValueError(f"ac_low must have length {num_outputs}, got shape {tuple(self.ac_low.shape)}")
        if self.ac_high.shape[-1] != num_outputs: # unit test
            raise ValueError(f"ac_high must have length {num_outputs}, got shape {tuple(self.ac_high.shape)}")

        #self.num_buckets = num_buckets
        self.bucket_size = (self.ac_high - self.ac_low) / self.num_buckets
            #torch.tensor((ac_high - ac_low) / num_buckets).to(device)

    def discretize(self, ac):
        ac = ensure_2d_tensor(ac, self.num_dims, name="action") # unit test
        bucket_idx = (ac - self.ac_low) // (self.bucket_size + self.eps)
        return bucket_idx.long()
        #return torch.clip(bucket_idx, 0, self.num_buckets - 1)

    def undiscretize(self, bucket_idx, dimension):
        if bucket_idx.ndim != 1: # unit test
            raise ValueError(f"bucket_idx must be 1D with shape (batch_size,), got {tuple(bucket_idx.shape)}")
        return_val = bucket_idx.float() * self.bucket_size[dimension] + self.ac_low[dimension] + self.bucket_size[dimension]*0.5
        #return_val = bucket_idx[:, None] * self.bucket_size + self.ac_low + self.bucket_size * 0.5
        #return return_val[:, dimension]
        return return_val

    def forward(self, state):
        state = ensure_2d_tensor(state, self.num_inputs, name="state") # unit test
        vals = []
        log_probs = torch.zeros(state.shape[0], device=state.device) # test
        #log_probs = 0

        for j in range(self.num_dims):
            # use previous sampled actions
            if j == 0:
                inp = state
            else:
                inp = torch.cat([state] + vals, dim=-1)

            logits = self.trunks[j](inp)
            dist = Categorical(logits=logits)

            sample = dist.sample()
            log_probs = log_probs + dist.log_prob(sample)

            action_j = self.undiscretize(sample, j).unsqueeze(-1)
            vals.append(action_j)

        vals = torch.cat(vals, dim=-1)
        return vals, log_probs

    def log_prob(self, state, action):
        """
                Evaluate log probability of expert action under autoregressive policy.

                Expected:
                    state:  shape (batch_size, state_dim) or (state_dim,)
                    action: shape (batch_size, action_dim) or (action_dim,)

                Returns:
                    log_prob: shape (batch_size,)
                """
        state = ensure_2d_tensor(state, self.num_inputs, name="state") # unit test
        action = ensure_2d_tensor(action, self.num_dims, name="action") # unit test
        if state.shape[0] != action.shape[0]: # unit test
            raise ValueError(f"state and action batch sizes must match. state shape={tuple(state.shape)}, action shape={tuple(action.shape)}")
        log_prob = torch.zeros(state.shape[0], device=state.device) #test
        #log_prob = 0.0
        ac_discretized = self.discretize(action)

        prev_actions = []

        for j in range(self.num_dims):

            # use TRUE previous actions (teacher forcing)
            if j == 0:
                inp = state
            else:
                inp = torch.cat([state] + prev_actions, dim=-1)

            logits = self.trunks[j](inp)
            dist = Categorical(logits=logits)

            log_prob = log_prob + dist.log_prob(ac_discretized[:, j])

            prev_actions.append(action[:, j:j + 1])

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
        with torch.no_grad():
            action, _ = agent(o_for_agent)
        action = action.detach().cpu().numpy()
        step_action = action.squeeze(0) if action.ndim > 1 else action
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
        #actions.append(action)
        actions.append(step_action) #experiment
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

# ---------------------------------------------------------------------
# Unit tests
# Run with:
#     python utils.py
# ---------------------------------------------------------------------

def test_autoregressive_log_prob_batched_action():
    torch.manual_seed(0)

    batch_size = 4
    state_dim = 11
    action_dim = 2

    policy = PolicyAutoRegressiveModel(
        num_inputs=state_dim,
        num_outputs=action_dim,
        hidden_dim=16,
        hidden_depth=1,
        num_buckets=10,
        ac_low=np.array([-1.0, -1.0]),
        ac_high=np.array([1.0, 1.0]),
    ).to(device)

    state = torch.randn(batch_size, state_dim, device=device)
    action = torch.rand(batch_size, action_dim, device=device) * 2 - 1

    log_prob = policy.log_prob(state, action)

    assert log_prob.shape == (batch_size,), (
        f"Expected log_prob shape {(batch_size,)}, got {tuple(log_prob.shape)}"
    )

    assert torch.isfinite(log_prob).all(), "log_prob contains NaN or Inf"


def test_autoregressive_log_prob_single_action_vector():
    torch.manual_seed(0)

    state_dim = 11
    action_dim = 2

    policy = PolicyAutoRegressiveModel(
        num_inputs=state_dim,
        num_outputs=action_dim,
        hidden_dim=16,
        hidden_depth=1,
        num_buckets=10,
        ac_low=np.array([-1.0, -1.0]),
        ac_high=np.array([1.0, 1.0]),
    ).to(device)

    state = torch.randn(state_dim, device=device)
    action = torch.tensor([0.25, -0.50], device=device)

    log_prob = policy.log_prob(state, action)

    assert log_prob.shape == (1,), (
        f"Expected log_prob shape {(1,)}, got {tuple(log_prob.shape)}"
    )

    assert torch.isfinite(log_prob).all(), "log_prob contains NaN or Inf"


def test_autoregressive_log_prob_rejects_wrong_action_dim():
    torch.manual_seed(0)

    state_dim = 11
    action_dim = 2

    policy = PolicyAutoRegressiveModel(
        num_inputs=state_dim,
        num_outputs=action_dim,
        hidden_dim=16,
        hidden_depth=1,
        num_buckets=10,
        ac_low=np.array([-1.0, -1.0]),
        ac_high=np.array([1.0, 1.0]),
    ).to(device)

    state = torch.randn(4, state_dim, device=device)
    bad_action = torch.randn(4, 3, device=device)

    try:
        policy.log_prob(state, bad_action)
    except ValueError as e:
        assert "action last dimension must be 2" in str(e)
    else:
        raise AssertionError("Expected ValueError for wrong action dimension")


def test_autoregressive_forward_shape():
    torch.manual_seed(0)

    batch_size = 5
    state_dim = 11
    action_dim = 2

    policy = PolicyAutoRegressiveModel(
        num_inputs=state_dim,
        num_outputs=action_dim,
        hidden_dim=16,
        hidden_depth=1,
        num_buckets=10,
        ac_low=np.array([-1.0, -1.0]),
        ac_high=np.array([1.0, 1.0]),
    ).to(device)

    state = torch.randn(batch_size, state_dim, device=device)

    action, log_prob = policy(state)

    assert action.shape == (batch_size, action_dim), (
        f"Expected action shape {(batch_size, action_dim)}, got {tuple(action.shape)}"
    )

    assert log_prob.shape == (batch_size,), (
        f"Expected log_prob shape {(batch_size,)}, got {tuple(log_prob.shape)}"
    )

    assert torch.isfinite(action).all(), "action contains NaN or Inf"
    assert torch.isfinite(log_prob).all(), "log_prob contains NaN or Inf"


def run_unit_tests():
    tests = [
        test_autoregressive_log_prob_batched_action,
        test_autoregressive_log_prob_single_action_vector,
        test_autoregressive_log_prob_rejects_wrong_action_dim,
        test_autoregressive_forward_shape,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print("All utils.py unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()