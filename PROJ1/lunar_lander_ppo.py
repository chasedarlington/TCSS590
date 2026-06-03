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
TIMESTEP = 1000 # update policy every n timesteps
EPOCHS = 10 # update policy for n epochs
EPSILON = 0.2 # clip log prob ratio to 1 +/- epsilon (PPO clip parameter)
GAMMA = 0.99 # discount factor
LR_ACTOR = 0.0003 # learning rate of the actor
LR_CRITIC = 0.001 # learning rate of the critic
EPISODES = 2000
EP_MAX_STEPS = 500
EP_REWARD_PENALTY = -0.00001
EP_TIMEOUT_PENALTY = -10.00


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
        #return action given args state
        with torch.no_grad():
          state = torch.as_tensor(state, device=self.device, dtype=torch.float32)
          action,action_logprob,state_val = self.policy_old.act(state) # STOCHASTIC ACTION SELECTION

        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.state_values.append(state_val)

        return action.item()

    def update(self , rewards, writer=None, global_step=None): # detach() to prevent backpropagation through old policy's states/actions/logprobs/state_values

        old_states = torch.squeeze(torch.stack(self.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.state_values, dim=0)).detach().to(self.device)

        ## COMPUTE ADVANTAGES !!

        advantages = rewards.detach() - old_state_values.detach()

        ## NORMALIZE ADVANTAGES !!

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for _ in range(self.epochs):

            ### EVALUATE OLD ACTIONS AND VALUES USING NEW POLICY
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs.detach()) # compute log probability ratios for the two (old and new) policies. 
            ## note: detach() to prevent backpropagation through old policy's log probabilities

            ### CALCULATE SURROGATE LOSS
            surr1 = ratios * advantages # surrogate loss without clipping
            surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages # clipped surrogate loss
            state_values = torch.squeeze(state_values) # squeeze states for loss fxn (match rewards tensor dimensions)
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = F.smooth_l1_loss(state_values, rewards)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy # final loss of PPO surrogate objective


            ## note: smoothing with Huber loss for value function, and adding an entropy bonus to encourage exploration. The critic loss is weighted by 0.5 and the entropy bonus is weighted by 0.01. The negative sign in front of the surrogate loss is because we want to maximize it, but optimizers minimize the loss.
            
            self.optimizer.zero_grad() # take gradient step
            loss.mean().backward() # back propagate the loss
            self.optimizer.step() # update the network parameters

            ## LOGGING !!
            if writer is not None and global_step is not None:
                writer.add_scalar("Loss/Total", loss.mean().item(), global_step)
                writer.add_scalar("Loss/Actor", actor_loss.mean().item(), global_step)
                writer.add_scalar("Loss/Critic", critic_loss.item(), global_step)
                writer.add_scalar("Policy/Entropy", dist_entropy.mean().item(), global_step)

        self.policy_old.load_state_dict(self.policy.state_dict())

    def learn(self, writer=None, global_step=None):
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
        self.update(rewards=rewards,writer=writer,global_step=global_step)
        self.empty_lists()

    def train_agent(self, env, num_episodes=EPISODES, max_steps=EP_MAX_STEPS, writer=None):
        time_step = 0
        episode_length = 0
        scores = []

        for current_episode in range(1, num_episodes+1):
            state = env.reset()[0]
            current_ep_reward = 0

            for t in range(1, max_steps+1): # max steps per episode

                action = self.act(state)
                state, reward, terminated , truncated, _ = env.step(action)
                reward += EP_REWARD_PENALTY # ADDING THIS PENALTY TO ENCOURAGE LANDING QUICKLY
                done = terminated or truncated

                if t == max_steps and not done:
                    reward += EP_TIMEOUT_PENALTY
                    done = True

                self.rewards.append(reward)
                self.dones.append(done)

                time_step +=1
                current_ep_reward += reward
                episode_length = t

                if time_step % self.update_timestep == 0:
                    self.learn(writer=writer,global_step=time_step)

                if done:
                    break

            ## LOGGING !!
            scores.append(current_ep_reward)
            avg_score_100 = np.mean(scores[-100:])
            if writer is not None:
                writer.add_scalar("Train/Episode_Return", current_ep_reward, current_episode)
                writer.add_scalar("Train/Average_Return_100", avg_score_100, current_episode)
                writer.add_scalar("Train/Episode_Length", episode_length, current_episode)
                writer.add_scalar("Train/Total_Timesteps", time_step, current_episode)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(current_episode, np.mean(scores[-100:])), end="")
            if current_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(current_episode, np.mean(scores[-100:])))
            if np.mean(scores[-100:]) >= 200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(current_episode,np.mean(scores[-100:])))
                break
        return scores

    def save(self, filepath: str):
        self.policy.save_model(filepath)

