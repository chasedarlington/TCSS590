---
title: "Coding Assignment 2: Policy Gradient and Actor Critic"
author: "Cole Buckingham (colbuc@uw.edu), Chase Darlington (darlchas@uw.edu)"
date: "May 12, 2026"
output:
  html_document:
    css: styles.css
---

**Course:** TCSS 590, Special Topics — Reinforcement Learning  
**School:** University of Washington Tacoma  
**Professor:** Professor Xutong Liu  
**Quarter:** Spring 2026  

---

## Introduction

d

---

## Behavior Cloning (BC)

### Method Overview

Behavior Cloning is a supervised learning approach that trains a policy, $\pi_\theta(\cdot \mid s)$, to mimic expert actions using maximum likelihood estimation.

The main objective is to maximize the log-likelihood of expert actions. Equivalently, the loss is the negative log probability of expert actions under the policy.

### Implementation Details

For our implementation in `bc.py`, the expert dataset was flattened by combining all trajectories into one large dataset. 
This allowed the model to train on individual state-action pairs rather than treating trajectories separately.

During training, the dataset was shuffled each epoch to randomize the batches.
We used a batch size of 32 for sampling. 
For each batch, the policy computed the log-likelihood of the expert actions using:
```python policy.log_prob(obs, acs)
```

The loss function was defined as the negative mean log-likelihood: $L <- (1 / N) * sum(log_pi_theta_a_given_s)$
The optimizer updated the model parameters after each batch.

The average training loss across all batches was recorded after each epoch. 
This follows the standard Behavior Cloning framework, where the policy learns to imitate the expert by maximizing the likelihood of expert actions.

### Training Plot

Below is the training plot from running the code.

knitr::include_graphics("images/bc_gaussian_reacher.png")
knitr::include_graphics("images/bc_gaussian_pointmaze.png")

### Results

#### Reacher
Success Rate: 0.30 (+/- 0.10)
Average Reward, success only: -7.84 (+/- 2.00)
Average Reward, all episodes: -11.67 (+/- 2.00)

Behavior Cloning performed moderately well in Reacher and exceeded the expected success-rate threshold of 0.20. 
However, failures still occurred because the learned policy may drift away from the expert trajectory over time.

The data also suggests performance collapse as batch size increases, especially beyond batch_size = 32, such as at batch_size >= 64. 
Larger batch sizes may lead to fewer gradient updates per epoch, meaning the policy receives fewer weight updates.

#### PointMaze
Success Rate: 0.16
Success Rate Range: 0.85
Average Reward, success only: 1.00
Average Reward, all episodes: 0.16
Average Reward Range: 0.85

Behavior Cloning performed worse in PointMaze. 
This environment is more sensitive to compounding error because small deviations in movement can prevent the agent from reaching the goal.

### Hyperparameter Experiment

Hyperparameter chosen: *Number of training epochs*

We chose the number of training epochs because it determines how many times the model trains over the expert dataset. We selected 100 epochs as a middle ground.

Too few epochs can cause underfitting, while more epochs can improve imitation performance by reducing training loss. Therefore, this hyperparameter strongly affects Behavior Cloning performance.

## DAgger

### Method Overview

DAgger, or Dataset Aggregation, improves Behavior Cloning by iteratively collecting data from the learned policy and relabeling visited states using the expert policy.
Unlike standard Behavior Cloning, which learns only from the original expert dataset, DAgger updates the training data with states encountered by the learner. 
This helps reduce compounding error caused by drifting into unseen states.

### Implementation Details

For our implementation in dagger.py, the initial training dataset began with the provided expert demonstrations.
At each iteration, the current learned policy was rolled out in the environment to collect new observations from visited states. 
These generated states were then relabeled by the expert policy, which provided the correct actions for each observation.
After aggregation, the policy was retrained using supervised learning with the same negative log-likelihood loss used in Behavior Cloning. 
The optimizer updated the model parameters after each batch during retraining.
This process was repeated over several DAgger iterations, allowing the model to continuously improve by learning how to recover from its own mistakes.

### Training Plot

Below is the training plot from running the code.

knitr::include_graphics("images/dagger_gaussian_reacher.png")

knitr::include_graphics("images/dagger_gaussian_pointmaze.png")

### Results

#### Reacher
Success Rate: 1.00
Average Reward, success only: -4.40
Average Reward, all episodes: -4.40

DAgger performed extremely well in Reacher and greatly exceeded the expected success-rate threshold of 0.20. 
Although the reward is still negative due to the environment’s distance-based penalty, the agent consistently completed the task.

#### PointMaze
Success Rate: 0.90
Average Reward, success only: 1.00
Average Reward, all episodes: 0.90

DAgger performed well in PointMaze. 
Compared to Behavior Cloning, DAgger improved robustness by learning from its own mistakes, 
which is especially important in navigation tasks where small errors can lead to failure.

### Hyperparameter Experiment

Hyperparameter chosen: *Number of DAgger iterations*
We chose a moderate number of DAgger iterations because it provides a balance between performance improvement and computational cost.
Too few iterations may not expose the policy to enough of its own mistakes, limiting the benefit over standard Behavior Cloning. 
Too many iterations can increase training time while producing diminishing returns once the policy has already converged.
This hyperparameter affects DAgger performance because each iteration improves the dataset by including states encountered by the learned policy.

## Autoregressive Policy

### Method Overview

Autoregressive policies model actions sequentially rather than predicting all action dimensions at once. 
Instead of outputting one full action vector directly, the model predicts one action component at a time. 
Each new prediction depends on the current state and previously predicted action components.

This allows the policy to capture relationships between action dimensions, which can improve coordination and produce smoother control behavior.

For training, the expert demonstrations were loaded successfully, and the state-action pairs were used from the provided expert dataset. 
The model was trained using mini-batch gradient descent, and negative log-likelihood loss was minimized to match expert actions.

### Implementation Details

For our implementation in utils.py, the continuous action space was discretized into a fixed number of buckets. 
The autoregressive network then predicted over the buckets for each action dimension in sequence.

The recorded training losses were:

Epoch 0: 491.88
Epoch 1: 258.49
Epoch 2: 203.58
Epoch 3: 177.92
Epoch 4: 162.71
Epoch 5: 152.57
Epoch 6: 145.27
Epoch 7: 139.35
Epoch 8: 134.48
Epoch 9: 130.45

The training loss steadily decreased across epochs, showing that the policy learned to imitate the expert demonstrations over time.

### Training Plot

Below is the training plot from running the code.

knitr::include_graphics("images/dagger_autoregressive_reacher.png")

knitr::include_graphics("images/dagger_autoregressive_pointmaze.png")

The training curve shows a strong downward trend in loss for both the Behavior Cloning and DAgger versions of the autoregressive policy. However, as optimization continued, the Behavior Cloning model better maintained expert actions, causing losses to decrease more steadily than in the DAgger version.

### Results

#### Reacher
Success Rate: 0.84
Average Reward, success only: 1.00
Average Reward, all episodes: 0.84

The autoregressive policy performed very well in Reacher and greatly exceeded the expected success-rate threshold of 0.20.
The policy succeeded in 84 out of 100 test episodes. 
Since successful episodes received full reward, this indicates that the model learned effective and reliable reaching behavior.
Because Reacher requires coordinated joint movement, the autoregressive structure likely helped by modeling dependencies between action outputs.

#### PointMaze
Success Rate: 0.72
Average Reward, success only: 1.00
Average Reward, all episodes: 0.72

The autoregressive policy performed well in PointMaze and exceeded the expected success-rate threshold of 0.20.
The policy succeeded in 72 out of 100 test episodes. 
Since successful episodes received full reward, this indicates that the model learned reasonably effective navigation behavior.
Because PointMaze requires consistent sequential decision-making, the autoregressive structure likely helped by producing more structured action sequences. 
However, some failures still occurred due to compounding errors, where small deviations from the expert trajectory caused the agent to miss the goal.
Compared to Gaussian policies, autoregressive policies better capture dependencies between action dimensions, leading to improved coordination. 
However, they still do not directly address distribution shift in the same way as DAgger.

## Conclusion

In this assignment, we explored three learning approaches: Behavior Cloning, DAgger, and autoregressive policies. 
We evaluated them across the Reacher and PointMaze environments. 
The results highlight key differences in how each method handles errors and action prediction.

Behavior Cloning provided a strong baseline and performed reasonably well in simpler settings such as Reacher. 
However, its performance degraded in PointMaze due to compounding errors. 
This occurred because the policy was trained only on expert data and could not reliably recover from states outside the training distribution.

DAgger significantly improved performance in both environments by iteratively collecting data from the policy’s own rollouts and correcting those states with expert labels. 
This allowed the model to learn how to recover from its own mistakes, leading to near-perfect success rates in Reacher and strong performance in PointMaze.

Autoregressive policies also showed strong performance, particularly in Reacher, by modeling dependencies between action dimensions. 
This led to more coordinated and structured actions compared to standard Gaussian policies. 
In PointMaze, the autoregressive model outperformed standard Behavior Cloning but still lagged behind DAgger because it did not address compounding error through dataset aggregation.

Overall, this assignment demonstrated that while model architecture can improve action prediction, addressing distribution shift often has a larger impact on performance in decision-making tasks. 
This is an important consideration for future reinforcement learning and imitation learning models.