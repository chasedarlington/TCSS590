---
title: "Lunar Landing with Reinforcement Learning: Milestone Report"
author: "Cole Buckingham (colbuc@uw.edu), Chase Darlington (darlchas@uw.edu)"
date: "May 11, 2026"
output: 
  html_document: 
    css: styles.css 
bibliography: references.bib
---

**Course:** TCSS 590, Special Topics ??? Reinforcement Learning  
**School:** University of Washington Tacoma  
**Professor:** Professor Xutong Liu  
**Quarter:** Spring 2026  

---


## Summary
The goal of this project is to develop a reinforcement learning agent that is capable of safely landing a spacecraft within a simulated environment. We???re doing these by using the ???Lunar Landing??? environment framework which is perfect for focusing on decision making that???s autonomous. In this the agent must learn how to be able to balance movement and fuel while interacting with a physics-based environment. 

Our original proposal for the project focused on applying reinforcement learning concepts commonly used in robotics and navigation systems. Since then, the direction of the project has shifted more to focusing on the optimal control and trajectory in addition to reinforcement learning.

The environment models spacecraft dynamics such as position and velocity, so the RL agent learns through repeated interactions within the environment to determine what is the correct option in the scenario. Successful actions allow the RL agent to receive rewards while failing actions penalitize it depending on the given behavior. 

This project is implemented in Python using reinforcement learning libraries inspired by OpenAI Gym and has used ???Lunar Landing??? as both the base and example on how to demonstrate reinforcement learning. 


## Progress So Far
Since the proposal, we have completed several major portions of the project that we hoped to set up before the first check-in with this Milestone. 
===================================================================================
Environment and Physics Setup:
We were able to design for and research the Lunar Landing environment using the reinforcement learning tools. Right now the environment contains basic physics and has action spaces consisting of???

Action Spaces: (Left, Right, Up Thrust, Left/Up Thrust, Right/Up Thrust and No Action).

The current state representations include:

State Representation: (Horizontal and vertical positions/velocity, land angle and remaining fuel).

Both the action spaces and state representations allow the agent to observe the full landing state before selecting a given action to perform and use to the best of their ability. 
===================================================================================
Reinforcement Learning Agent:
We were able to implement a very early framework for the reinforcement learning agent and reward system. Right now the reward structure only really includes:

Positive rewards for safe landings.
Penalties for crashing.
Penalties for fuel consumption.

Right now reward shaping is designed to encourage improvement using repeated episodes of the agent. 
===================================================================================
Optimal Control Research:
In addition to reinforcement learning implementation, Chase has searched up control methods strongly related to Lunar Landing such as Pontryagin???s Maximum Principle. Pontryagin???s Maximum Principle is a fundamental theorem in optimal control theory, stemming from rocket terminal speed maximization, that describes the best possible control for taking a dynamical system, evolving with time, from one state to another. 

Essentially, in Pontryagin???s framework, an agent chooses a control signal over time that minimizes cost subject to physical motion equations. For an optimal trajectory, there is a set of variables, costates or adjoint variables, which inform how valuable or costly each state variable is at a given time. 

Pontryagin's minimum principle states that the optimal state trajectory x, optimal control u, and corresponding Lagrange multiplier vector ??, must minimize the Hamiltonian H so that

for all time t???[0,T] and for all permissible control inputs u???U. Here, the trajectory of the Lagrangian multiplier vector ?? is the solution to the costate equation:

where ??T is the transpose of ??. Final state and time not being fixed results in the following conditions:


The project formulation currently models the landing state as:
x(t) = [px, py, vx, vy, 0, w, fuel]
With control actions represented as:
u(t) = [left, right, up, left + up, right + up, none]

This minimizes total landing cost subjected to spacecraft.

Preliminary Testing:
We have begun initial training and debugging of the agent with the environment and have verified:
Most physic interactions have functioned correctly. 
The action inputs properly affect motion of the ship.
Rewards/penalties are being applied.

Though the model is still early in training, everything is functioning properly at the moment.

## Challenges
Currently we have had a few challenges regarding development.
===================================================================================
Reward Balancing:
One of the hardest challenges was determining an effective reward/penalizing structure without going all in one direction. We tried to do these via small adjustments to reward values without significantly changing the behavior of the agent. It appears more often than not that the agent prioritized survival without landing efficiently while in others it seems to overuse thrust and waste fuel. Fine-tuning the RL agent was difficult in this and we ended up making tiny micro-adjustments until we believed it???s satisfactory.
===================================================================================
Training:
Training has also been challenging. Early learning attempts produced inconsistent landing behavior across the episodes. The agent occasionally converged towards strategies we thought were undesirable in getting accurate and consistent data from. 
===================================================================================
Environment:
Our environment has had some challenges in having terrarian that is suitable for the reinforcement learning task. We tried to make the environment both simple and robust enough for the ship to land while still actually learning. So far we have done that but we may need some improvement for it going forward. 


## Help Needed
At this stage, we would appreciate some feedback and some guidance regarding:
Suggestions for improving reward shaping and training.
Advice on reinforcement learning baselines appropriate for Lunar Landing environments.
Feedback on integrating deterministic optimal control concepts with reinforcement learning methods.
Recommendations for balancing exploration versus exploitation during training.

With these we???d believe we would make the most optimal version of our project in our eyes.


## Updated Timeline

**Wk 1** (Apr 20 - 26): Configure the game environment. COMPLETE 
**Wk 2** (Apr 27- May 3): Implement the RL agent and basic reward system. COMPLETE 
**Wk 3** (May 4 - 10): Deploy the agent and debug learning behavior.  COMPLETE
**Wk 4** (May 11 - 17): Add environmental challenges.  
**Wk 5** (May 18 - 24): Perform code-level optimization.  
**Wk 6** (May 25 - 31): Finalize results.  
**Wk 7** (Jun 3): Present and demonstrate RL outcomes.

