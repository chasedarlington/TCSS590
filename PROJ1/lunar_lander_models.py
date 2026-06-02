import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        #basic actor model
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_size):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.net(state)


# combined Actor Critic model for PPO Agent

class AgentPPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(AgentPPO, self).__init__()
        self.actor = Actor(state_dim , action_dim)
        self.critic = Critic(state_dim)

    def act(self, state):
        state = torch.as_tensor(state, device = self.device, dtype = torch.float32)
        #state = torch.FloatTensor(state.cpu()).to(self.device)
        action_logits = self.actor(state) # output of actor nn
        #SoftMax = nn.Softmax(dim=-1)
        #action_probs = SoftMax(action_logits)# outputs -> probabilities
        #dist = Categorical(probs = action_probs)# distribution according to the probabilities
        dist = Categorical(logits=action_logits)
        selected_action = dist.sample()# random action sampling from the distribution
        log_prob = dist.log_prob(selected_action)# log of the probability of the selected action (for calculating loss)
        state_value = self.critic(state) # expected value of the current
        return selected_action , log_prob, state_value


    def evaluate(self, state, action):
        #state = torch.FloatTensor(state.cpu()).to(self.device)
        action_logits = self.actor(state) # output of actor
        # SoftMax = nn.Softmax(dim=-1)
        #action_probs = SoftMax(action_logits)# outputs -> probabilities
        #dist = Categorical(probs = action_probs)# discrete distribution from the probabilities
        dist = Categorical(logits = action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs , state_values, dist_entropy

    def save_model(self, filepath: str):
        torch.save({
            'Actor_state_dict': self.actor.state_dict(),
            'Critic_state_dict': self.critic.state_dict(),
        }, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['Actor_state_dict'])
        self.critic.load_state_dict(checkpoint['Critic_state_dict'])