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
import numpy as np
import torch.nn.functional as F
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from matplotlib import pyplot as plt

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

    ## ACTOR NETWORK ---(STATE)--> ONE LOGIT PER DISCRETE ACTION
    ## CRITIC NETWORK ---(STATE)--> BASELINE VALUE (FOR ADVANTAGE FXN)
    def act(self, state):
        #state = torch.as_tensor(state, device = self.device, dtype = torch.float32)
        #state = torch.FloatTensor(state.cpu()).to(self.device)
        action_logits = self.actor(state) ## output of actor nn
        #SoftMax = nn.Softmax(dim=-1)
        #action_probs = SoftMax(action_logits) ## outputs -> probabilities
        #dist = Categorical(probs = action_probs) ## distribution according to the probabilities

        ## STOCHASTIC ACTION SELECTION !!
        dist = Categorical(logits=action_logits)
        selected_action = dist.sample() ## random action sampling from the distribution
        log_prob = dist.log_prob(selected_action) ## log of the probability of the selected action (for calculating loss)
        state_value = self.critic(state) ## expected value of the current

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
            
            self.optimizer.zero_grad() # take gradient time_step
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
            state = env.reset(seed=43)[0]
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


"""
This is the project entry point.
It creates the Gymnasium LunarLander environment,
creates the PPO agent, trains the agent,
saves the trained model, and closes the environment.
"""


def main():
    ## INITIALIZE THE ENVIRONMENT !!

    writer = SummaryWriter(log_dir="runs/ppo_lunar_lander")
    env = gym.make("LunarLander-v2")

    ## CREATE THE AGENT !!      using env observation and action dimensions

    agent = PPOAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    ## TRAIN THE AGENT !!

    scores = agent.train_agent(env=env, writer=writer)

    ## PLOT SCORES !!

    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO LunarLander Training Scores")
    plt.show()

    agent.save("ppo_lunar_lander.pt")
    writer.close()
    env.close()

    """

    # Reset the environment to generate the first observation
    observation, info = env.reset(seed=42)

    # iterate for 1000 episodes

    for _ in range(1000):
        agent = PPOAgent(
            state_size = env.observation_space.shape[0] # 8?
            , action_size = env.action_space.shape[0] # 4?
            , device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        action = env.action_space.sample()

        # time_step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated

        observation, reward, terminated, truncated, info = env.time_step(action)


        # If the episode has ended then we can reset to start args new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    """


def render(model_path="ppo_lunar_lander.pt", episodes=5):
    env = gym.make("LunarLander-v2", render_mode="human")
    agent = PPOAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    checkpoint = torch.load(model_path)
    agent.policy.actor.load_state_dict(checkpoint["Actor_state_dict"])
    agent.policy.critic.load_state_dict(checkpoint["Critic_state_dict"])

    agent.policy.eval()

    for episode in range(episodes):
        state, info = env.reset(seed=43)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            ## STOCHASTIC SAMPLING !!

            # with torch.no_grad():
            #    action_logits = agent.policy.actor(state_tensor)
            #    dist = torch.distributions.Categorical(logits=action_logits)
            #    action = dist.sample().item()

            ## DETERMINISTIC SAMPLING !!
            with torch.no_grad():
                action_logits = agent.policy.actor(state_tensor)
                action = torch.argmax(action_logits).item()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    ## CREATE THE AGENT AND TRAIN THE MODEL !!
    main()

    ## RENDER PREVIOUSLY TRAINED AGENT !! (USE .PT FILE)
    # render()

    """ 
    TENSOR BOARD TAGS

        Train/Episode_Return
        Train/Average_Return_100
        Train/Episode_Length
        Train/Total_Timesteps

        Loss/Total
        Loss/Actor
        Loss/Critic

        Policy/Entropy

    PLOTS

        Average_Return_100
        Episode_Return
        Episode_Length
        Loss/Actor
        Loss/Critic
        Policy/Entropy

    """