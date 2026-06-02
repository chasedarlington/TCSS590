"""
This is the project entry point.
It creates the Gymnasium LunarLander environment,
creates the PPO agent, trains the agent,
saves the trained model, and closes the environment.
"""

import gymnasium as gym
import torch
from lunar_lander_ppo import PPOAgent

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## INITIALIZE THE ENVIRONMENT !!

    env = gym.make("LunarLander-v2", render_mode="human")

    ## CREATE THE AGENT !!      using env observation and action dimensions

    agent = PPOAgent(
        state_size = env.observation_space.shape[0],
        action_size = env.action_space.n,
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    scores = agent.train_agent(env)

    agent.save("ppo_lunar_lander.pt")
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

if __name__ == "__main__":
    main()