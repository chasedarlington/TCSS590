"""
This file defines the PPO training algorithm. It owns the PPOAgent class, which handles:

PPO hyperparameters:
 - actor/critic optimizer
 - trajectory memory
 - action selection
 - reward/return computation
 - PPO clipped policy update
 - training loop
 - model save/load wrappers
"""
import torch
import torch.nn.functional as F
import numpy as np
from lunar_lander_models import AgentPPO

## HYPERPARAMETERS !!
TIMESTEP = 2048 #1000 # update policy every n timesteps
EPOCHS = 10 # update policy for n epochs
EPSILON = 0.2 # clip log prob ratio to 1 +/- epsilon (PPO clip parameter)
GAMMA = 0.99 # discount factor
TAU = 1e-3 # for soft update of target parameters
LR_ACTOR = 0.0005 # learning rate of the actor
LR_CRITIC = 0.0005 # learning rate of the critic
EPISODES = 2000
MINI_BATCH_SIZE = 64 # 1000


# CLASS: PPOAgent for discrete action spaces (e.g. LunarLander-v3)
class PPOAgent:
    
    """
    METHOD: __init__: initialize the PPO agent
    INPUTS: 
      state_size: dimension of the observation space, 
      action_size: dimension of the action space
      device: torch.device() -> cuda or cpu
    OUTPUTS: None
    """
    def __init__(self , state_size, action_size, device):
        # self.max_training_timesteps = int(5e5) # break training loop if timeteps > max_training_timesteps
        self.update_timestep = TIMESTEP
        self.epochs = EPOCHS
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.device = device
        self.state_dim = state_size
        self.action_dim = action_size
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.dones = []
        self.policy = AgentPPO(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': LR_ACTOR},
                        {'params': self.policy.critic.parameters(), 'lr': LR_CRITIC}
                    ])

        self.policy_old = AgentPPO(self.state_dim, self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def empty_lists(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.dones[:]

    def act(self, state):
        #return action given a state
        with torch.no_grad():
          state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
          action,action_logprob,state_val = self.policy_old.act(state)

        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.state_values.append(state_val)

        return action.item()

    def evaluate(self, state, action):
        action_logits = self.actor(state)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

    def update(self , rewards): # detach() to prevent backpropagation through old policy's states/actions/logprobs/state_values

        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.state_values, dim=0)).detach().to(self.device)

        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.epochs):

            ### EVALUATE OLD ACTIONS AND VALUES USING NEW POLICY
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach()) # compute log probability ratios for the two (old and new) policies. 
            ## note: detach() to prevent backpropagation through old policy's log probabilities

            ### CALCULATE SURROGATE LOSS
            surr1 = ratios * advantages # surrogate loss without clipping
            surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages # clipped surrogate loss
            state_values = torch.squeeze(state_values) # squeeze states for loss fxn (match rewards tensor dimensions)
            loss = -torch.min(surr1, surr2) + 0.5 * F.smooth_l1_loss(state_values, rewards) - 0.01 * dist_entropy # final loss of PPO surrogate objective
            ## note: smoothing with Huber loss for value function, and adding an entropy bonus to encourage exploration. The critic loss is weighted by 0.5 and the entropy bonus is weighted by 0.01. The negative sign in front of the surrogate loss is because we want to maximize it, but optimizers minimize the loss.
            
            self.optimizer.zero_grad() # take gradient step
            loss.mean().backward() # back propagate the loss
            self.optimizer.step() # update the network parameters

        self.policy_old.load_state_dict(self.policy.state_dict())

    def learn(self):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        self.update(rewards)
        self.empty_lists()

    def train_agent(self, env, num_episodes=EPISODES, batch_size=MINI_BATCH_SIZE):
        time_step = 0
        i_episode = 1
        scores = []

        for i_episode in range(1, num_episodes+1):
            state = env.reset()[0]
            current_ep_reward = 0
            for t in range(1, batch_size+1):
                action = self.act(state)
                state, reward, terminated , truncated, _ = env.step(action)
                done = terminated or truncated
                self.rewards.append(reward)
                self.dones.append(done)
                time_step +=1
                current_ep_reward += reward
                if time_step % self.update_timestep == 0:
                    self.learn()
                if done:
                    break
            scores.append(current_ep_reward)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores[-100:])), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores[-100:])))
            if np.mean(scores[-100:]) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,np.mean(scores[-100:])))
                break
        return scores

    def save(self, filepath: str):
        self.policy.save_model(filepath)

    def load(self, filepath: str):
        self.policy.load_model(filepath, self.device)
