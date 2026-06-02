main.py:
  creates Gymnasium env
  creates PPOAgent
  calls train_agent()

lunar_lander_ppo.py:
  defines PPOAgent
  manages training loop, memory, returns, PPO loss, optimizer
  uses AgentPPO model

lunar_lander_models.py:
  defines Actor neural network
  defines Critic neural network