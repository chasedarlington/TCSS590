"""
This is the project entry point.
It creates the Gymnasium LunarLander environment,
creates the PPO agent, trains the agent,
saves the trained model, and closes the environment.
"""

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

from lunar_lander_ppo import PPOAgent

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## INITIALIZE THE ENVIRONMENT !!
    writer = SummaryWriter(log_dir="runs/ppo_lunar_lander")
    env = gym.make("LunarLander-v2")

    ## CREATE THE AGENT !!      using env observation and action dimensions

    agent = PPOAgent(
        state_size = env.observation_space.shape[0],
        action_size = env.action_space.n,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    scores = agent.train_agent(env, writer)
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

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated

        observation, reward, terminated, truncated, info = env.step(action)


        # If the episode has ended then we can reset to start a new episode
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
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            with torch.no_grad():
                action_logits = agent.policy.actor(state_tensor)

                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample().item()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

    env.close()




if __name__ == "__main__":

    ## CREATE THE AGENT AND TRAIN THE MODEL !!
    #main()

    ## RENDER PREVIOUSLY TRAINED AGENT !! (USE .PT FILE)
    render()


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