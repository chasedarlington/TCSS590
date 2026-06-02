"""
================================================================================
  PPO.RL.py  —  Production-Grade Proximal Policy Optimization
================================================================================

WHAT IS PPO? (Explain Like I'm 5)
----------------------------------
Imagine you're teaching a dog (the "agent") to navigate a maze (the "environment").
Every step the dog takes, it gets a treat (+reward) or a smack (-reward).

The dog has two "brains":
  🧠 Actor  → decides WHAT action to take ("go left / right / jump")
  🧠 Critic → judges HOW GOOD the current situation is ("this position is worth 10 treats")

PPO (Proximal Policy Optimization) is the training algorithm that teaches both brains.
Its key idea: "Don't change your mind too drastically in one step."
If the dog accidentally learns "always go right" too strongly, it might forget how to go left.
PPO clips (limits) how much the policy can change per update — like a safety leash.

Key math concepts (ELI5):
  - Advantage   : "Was this action better or worse than average?" (Actual reward - Expected reward)
  - GAE         : A smoother way to estimate advantage using multiple future steps
  - Clipped Ratio: If new policy is too different from old, clip the update to prevent instability
  - Entropy Bonus: Encourage exploration — don't let the agent become boring and repetitive

SUPPORTS:
  ✅ Discrete action spaces    (e.g. CartPole, Atari, Gymnasium)
  ✅ Continuous action spaces  (e.g. MuJoCo, robotics arms, locomotion)
  ✅ Pluggable backbone networks (MLP default, Transformer, CNN, custom)
  ✅ Vectorized parallel environments
  ✅ Observation & reward normalization
  ✅ Gradient clipping
  ✅ Learning-rate scheduling (linear decay / cosine annealing / warmup)
  ✅ Value-function clipping (extra PPO stabilization trick)
  ✅ KL-divergence early stopping
  ✅ TensorBoard + W&B + CSV logging
  ✅ Full checkpoint save/resume
  ✅ Mixed precision (AMP) training
  ✅ Robot training (continuous, recurrent-ready)
  ✅ Transformer policy backbone

USAGE QUICK-START:
------------------
    from PPO.RL import PPOConfig, PPOAgent, make_gymnasium_env

    cfg   = PPOConfig(env_id="CartPole-v1", total_timesteps=500_000)
    agent = PPOAgent(cfg)
    agent.learn()

    # Continuous (robotic) env:
    cfg = PPOConfig(env_id="HalfCheetah-v4", action_space_type="continuous",
                   total_timesteps=1_000_000)
    agent = PPOAgent(cfg)
    agent.learn()

    # Custom environment:
    agent = PPOAgent(cfg, env_factory=my_env_factory)
    agent.learn()

    # Custom policy backbone (e.g. Transformer):
    from PPO.RL import TransformerPolicyBackbone
    cfg = PPOConfig(env_id="MyEnv-v0", backbone_cls=TransformerPolicyBackbone)
    agent = PPOAgent(cfg)

AUTHORS:  Built from PyTorch PPO Tutorial + tomasspangelo/proximal-policy-optimization
LICENSE:  MIT
================================================================================
"""

from __future__ import annotations

# ── Standard Library ─────────────────────────────────────────────────────────
import abc
import csv
import logging
import math
import os
import random
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, List, Optional,
    Sequence, Tuple, Type, Union,
)

# ── Third-Party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, Beta
#from torch.utils.tensorboard import SummaryWriter  # pip install tensorboard

try:
    import gymnasium as gym                         # pip install gymnasium
    _GYM_AVAILABLE = True
except ImportError:
    try:
        import gym                                  # fallback: classic gym
        _GYM_AVAILABLE = True
    except ImportError:
        _GYM_AVAILABLE = False
        warnings.warn("Neither 'gymnasium' nor 'gym' found. Install with: pip install gymnasium")

try:
    import wandb                                    # pip install wandb  (optional)
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PPO.RL")


# ══════════════════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
#
# ELI5: Think of PPOConfig as a recipe card.
#       Before you cook, you write down all the ingredients and cooking times.
#       Here we write down all the settings PPO needs before training starts.

@dataclass
class PPOConfig:
    """
    All hyper-parameters and training settings for PPO in one place.

    Key parameters explained simply:
      total_timesteps  : How many total "game steps" to train for.
      rollout_steps    : How many steps to collect before updating networks.
      num_envs         : How many parallel game copies to run simultaneously (faster!).
      gamma            : "How much do I care about future rewards?" 0=ignore, 1=care forever.
      gae_lambda       : Smoothing factor for advantage estimation. 0.95 is a safe default.
      clip_epsilon     : The PPO "safety leash". Max 20% policy change per update (0.2).
      n_epochs         : How many passes over the collected data during each update.
      lr_actor         : How fast the Actor brain learns. Too high → unstable, too low → slow.
      lr_critic        : How fast the Critic brain learns.
      entropy_coef     : Bonus reward for "thinking differently" → encourages exploration.
      value_coef       : How much to weight the Critic's loss vs. the Actor's loss.
    """

    # ── Environment ──────────────────────────────────────────────────────────
    env_id: str = "CartPole-v1"
    """Gymnasium environment ID (e.g. 'CartPole-v1', 'HalfCheetah-v4')."""

    action_space_type: str = "discrete"
    """'discrete' for integer actions (games), 'continuous' for real-valued (robots)."""

    num_envs: int = 4
    """
    Number of parallel environments.
    ELI5: Like running 4 copies of the game simultaneously to collect data faster.
    More envs → faster data collection, but more RAM/CPU needed.
    """

    seed: int = 42
    """Random seed for reproducibility. Same seed → same training trajectory."""

    # ── Training Schedule ────────────────────────────────────────────────────
    total_timesteps: int = 1_000_000
    """Total environment steps to train for."""

    rollout_steps: int = 2048
    """
    Steps collected per environment before each policy update.
    ELI5: Collect 2048 game steps per env, then teach the agent from that data.
    Total data per update = rollout_steps × num_envs.
    """

    n_epochs: int = 10
    """
    How many times to re-use collected data for gradient updates.
    ELI5: Re-read the same textbook chapter 10 times to really memorise it.
    More epochs → more efficient data use, but risk of overfitting to stale data.
    """

    minibatch_size: int = 64
    """
    Size of each mini-batch during gradient updates.
    ELI5: Instead of studying all 2048 examples at once, study 64 at a time.
    Smaller batches → noisier gradients but sometimes better generalisation.
    """

    # ── Discount & GAE ───────────────────────────────────────────────────────
    gamma: float = 0.99
    """
    Discount factor for future rewards.
    ELI5: A reward in 100 steps is worth gamma^100 of a reward right now.
    gamma=0.99 means future rewards are almost as valuable as immediate ones.
    gamma=0    means the agent is completely short-sighted (only NOW matters).
    """

    gae_lambda: float = 0.95
    """
    GAE (Generalized Advantage Estimation) smoothing factor.
    ELI5: Instead of estimating "how good was that action?" from one future step,
    we blend estimates from many future steps. lambda controls the blend:
      lambda=0 → only use the immediate next step (low variance, high bias)
      lambda=1 → use ALL future steps (high variance, unbiased)
    0.95 is a sweet spot used in most PPO papers.
    """

    # ── PPO Clipping & Losses ────────────────────────────────────────────────
    clip_epsilon: float = 0.2
    """
    The PPO clipping parameter — the "safety leash".
    ELI5: If the new policy's probability ratio vs old is outside [0.8, 1.2],
    we CLIP it. This prevents catastrophic policy changes in one update.
    Standard value: 0.2. Larger = more aggressive updates.
    """

    clip_epsilon_schedule: str = "linear"
    """
    How clip_epsilon changes over training: 'constant', 'linear' (decay to 0), or 'none'.
    Linear decay makes training more conservative as it matures.
    """

    value_clip_epsilon: float = 0.2
    """
    Clip value function loss too (same as clip_epsilon by default).
    This is an extra stabilisation trick: don't let the Critic's value estimates
    change too drastically in a single update either.
    Set to None to disable value clipping.
    """

    value_coef: float = 0.5
    """
    Weight of the Critic (value) loss in the total loss.
    Total loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy.
    """

    entropy_coef: float = 0.01
    """
    Weight of the entropy bonus in the total loss.
    ELI5: We ADD a bonus to encourage the agent to be "uncertain" (exploratory).
    If the agent becomes too confident too fast, it stops exploring better strategies.
    entropy_coef=0.01 adds a tiny exploration bonus.
    """

    entropy_coef_schedule: str = "linear"
    """
    How entropy_coef changes: 'constant', 'linear' (decay to 0), or 'none'.
    Decaying entropy → more exploration early, more exploitation later.
    """

    max_grad_norm: float = 0.5
    """
    Maximum gradient norm for gradient clipping.
    ELI5: If the gradient (the learning signal) is too large, it's scaled down.
    This prevents the network from making huge, destabilizing leaps in one step.
    Think of it as a speed limiter on learning.
    """

    target_kl: Optional[float] = 0.02
    """
    If KL-divergence between old and new policy exceeds this, stop epoch early.
    ELI5: KL measures "how different are the old and new policies?"
    If the policies diverge too much, we abort the update to prevent instability.
    Set to None to disable early stopping.
    """

    # ── Network Architecture ──────────────────────────────────────────────────
    hidden_sizes: Tuple[int, ...] = (64, 64)
    """
    Hidden layer sizes for MLP backbone.
    ELI5: The Actor and Critic each have a stack of layers. (64, 64) means
    two hidden layers each with 64 neurons. Bigger → more capacity, slower.
    """

    activation: str = "tanh"
    """
    Activation function: 'tanh', 'relu', 'elu', 'gelu'.
    ELI5: The "squishing function" applied after each layer.
    'tanh' works well for bounded state spaces. 'relu' is popular for large nets.
    """

    shared_backbone: bool = False
    """
    If True, Actor and Critic share the same feature extractor (backbone).
    ELI5: One pair of eyes (shared) sees the world, then two separate heads decide
    (action head) and judge (value head). Often faster to train.
    If False, Actor and Critic each have their own independent networks.
    """

    backbone_cls: Optional[Type[nn.Module]] = None
    """
    Optional custom backbone class. Must accept (obs_dim, hidden_sizes, activation)
    and output a feature tensor. Overrides the default MLP if provided.
    Use this to plug in a Transformer, CNN, LSTM, etc.
    """

    # ── Continuous Action Space Settings ─────────────────────────────────────
    continuous_dist: str = "gaussian"
    """
    For continuous actions: 'gaussian' (Normal) or 'beta' (Beta distribution).
    Gaussian: unbounded actions, common for MuJoCo, robotics.
    Beta: bounded [0,1] actions, good for normalized action spaces.
    """

    log_std_init: float = -0.5
    """
    Initial log-standard-deviation for Gaussian policy.
    ELI5: How "spread out" the action distribution is initially.
    More spread = more exploration early on. Becomes learnable during training.
    """

    squash_actions: bool = False
    """
    Apply tanh squashing to Gaussian actions to bound them to [-1, 1].
    ELI5: Forces continuous actions to stay within a safe range.
    Required for environments with bounded action spaces (e.g. [-1, 1]).
    """

    # ── Optimiser & LR ───────────────────────────────────────────────────────
    lr_actor: float = 3e-4
    """Actor learning rate. 3e-4 is the "magic number" from the PPO paper."""

    lr_critic: float = 1e-3
    """Critic learning rate. Usually slightly higher than actor."""

    lr_schedule: str = "linear"
    """
    LR schedule: 'constant', 'linear' (decay to 0), 'cosine', 'warmup_cosine'.
    Linear decay is the most common PPO choice (matches the paper).
    """

    optimizer: str = "adam"
    """Optimizer: 'adam', 'adamw', 'sgd'."""

    adam_eps: float = 1e-5
    """
    Adam epsilon for numerical stability.
    ELI5: Tiny number to avoid division by zero in the Adam optimizer.
    Slightly larger (1e-5 vs 1e-8) can improve stability for RL.
    """

    weight_decay: float = 0.0
    """L2 regularisation weight. 0 = no regularisation (standard for PPO)."""

    # ── Normalisation ─────────────────────────────────────────────────────────
    normalise_obs: bool = True
    """
    Normalise observations using a running mean/std.
    ELI5: If observations are wildly different scales (e.g. position=0.01, velocity=100),
    learning is unstable. Normalising puts everything in the same "currency".
    """

    normalise_rewards: bool = True
    """
    Normalise rewards using a running std (NOT mean — preserves sign).
    ELI5: If some rewards are +1000 and some are -0.001, the gradient signals are noisy.
    Normalising makes learning more stable across different reward scales.
    """

    reward_clip: float = 10.0
    """Clip normalised rewards to [-reward_clip, reward_clip]. Prevents outlier reward explosions."""

    # ── Mixed Precision ───────────────────────────────────────────────────────
    use_amp: bool = False
    """
    Use Automatic Mixed Precision (AMP / float16) for faster GPU training.
    ELI5: Use 16-bit numbers instead of 32-bit where safe → up to 2× speedup on modern GPUs.
    Requires CUDA. Automatically disabled if CPU is used.
    """

    # ── Logging & Checkpointing ───────────────────────────────────────────────
    log_dir: str = "./ppo_logs"
    """Directory for TensorBoard logs, CSV logs, and checkpoints."""

    experiment_name: str = "ppo_run"
    """Name for this experiment run (used in log filenames and W&B)."""

    log_interval: int = 1
    """Log metrics every N policy updates."""

    save_interval: int = 10
    """Save checkpoint every N policy updates."""

    use_wandb: bool = False
    """Enable Weights & Biases logging (requires: pip install wandb; wandb login)."""

    wandb_project: str = "ppo_rl"
    """W&B project name."""

    resume_from: Optional[str] = None
    """Path to checkpoint file to resume training from."""

    # ── Advanced ──────────────────────────────────────────────────────────────
    ortho_init: bool = True
    """
    Use orthogonal initialisation for network weights.
    ELI5: A special way to initialise weights that helps RL training be more stable.
    Empirically shown to improve PPO performance (used in Stable-Baselines3).
    """

    norm_adv: bool = True
    """
    Normalise advantages (zero mean, unit variance) within each mini-batch.
    ELI5: Makes sure advantage estimates don't have wildly different scales,
    keeping gradient updates consistent across different parts of training.
    """

    recurrent: bool = False
    """
    [Future flag] Enable recurrent (LSTM/GRU) policy for partially observable envs.
    Not yet implemented — use a Transformer backbone for sequence modelling instead.
    """

    device: str = "auto"
    """
    Compute device: 'auto' (GPU if available), 'cpu', 'cuda', 'cuda:0', 'mps'.
    'auto' is recommended — it automatically picks the best available device.
    """


# ══════════════════════════════════════════════════════════════════════════════
# §2  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_device(cfg_device: str) -> torch.device:
    """
    Resolve the compute device.

    ELI5: Figures out WHERE to run the calculations.
    GPU (CUDA) is like a sports car — fast at parallel math.
    CPU is like a reliable sedan — always available, but slower for big networks.
    MPS is Apple Silicon's GPU backend.
    """
    if cfg_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-selected device: CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-selected device: Apple MPS")
        else:
            device = torch.device("cpu")
            logger.info("Auto-selected device: CPU")
    else:
        device = torch.device(cfg_device)
        logger.info(f"Using device: {device}")
    return device


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    ELI5: Like shuffling a deck of cards in a specific way so you always get
    the same "random" order. Needed to reproduce results exactly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CUDA deterministic (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def discount_cumsum(x: np.ndarray, discount: float) -> np.ndarray:
    """
    Compute discounted cumulative sum of a 1D array.

    ELI5: Given rewards [r0, r1, r2, ...], compute:
      G0 = r0 + γ*r1 + γ²*r2 + ...
      G1 = r1 + γ*r2 + ...
      G2 = r2 + ...

    This is the "return" — how much total future reward from each step.
    scipy.signal.lfilter computes this efficiently as a digital filter.

    Args:
        x       : 1D array of rewards or deltas
        discount: discount factor γ (gamma)
    Returns:
        Discounted cumulative sums, same shape as x
    """
    # scipy's lfilter applies an IIR filter — this is a math trick to compute
    # the cumulative sum from right to left efficiently in O(n).

    #return scipy.signal.lfilter([1.0], [1.0, -discount], x[::-1], axis=0)[::-1]
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=np.float32)

    running_sum = 0.0
    for t in reversed(range(len(x))):
        running_sum = x[t] + discount * running_sum
        result[t] = running_sum

    return result

def get_activation(name: str) -> nn.Module:
    """
    Return activation function by name.

    ELI5: The "squishing function" that goes between neural network layers.
    Without activations, stacking linear layers is pointless (they collapse into one).
    Activations introduce non-linearity — the ability to learn complex patterns.
    """
    activations = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(0.01),
        "silu": nn.SiLU(),          # Smooth version of ReLU, used in transformers
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Choose from: {list(activations)}")
    return activations[name]


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """
    Orthogonal initialisation for a linear layer.

    ELI5: Normally neural network weights start random. Orthogonal init starts them
    in a special configuration that avoids vanishing/exploding gradients early on.
    This is the standard init used in PPO implementations (CleanRL, Stable-Baselines3).

    Args:
        layer    : nn.Linear layer to initialise
        std      : Standard deviation scale for the init (smaller = smaller initial outputs)
        bias_const: Initial value for bias terms
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# ══════════════════════════════════════════════════════════════════════════════
# §3  RUNNING STATISTICS (Observation & Reward Normalization)
# ══════════════════════════════════════════════════════════════════════════════

class RunningMeanStd:
    """
    Track running mean and standard deviation of a stream of data.

    ELI5: Imagine you're measuring heights of students one by one.
    Instead of storing all heights and computing average at the end,
    you track a "running average" that updates with each new student.
    This is memory-efficient and works for infinite data streams.

    Used to normalise observations so the network always sees well-scaled inputs.

    Algorithm: Welford's online algorithm (numerically stable).
    """

    def __init__(self, shape: Tuple[int, ...] = (), epsilon: float = 1e-8):
        """
        Args:
            shape  : Shape of each data point (e.g., (obs_dim,) for observations)
            epsilon: Tiny number to prevent division by zero
        """
        self.mean    = np.zeros(shape, dtype=np.float64)
        self.var     = np.ones(shape, dtype=np.float64)
        self.count   = epsilon          # Start at epsilon to avoid /0

    def update(self, x: np.ndarray) -> None:
        """
        Update running statistics with a batch of new data.

        ELI5: Given a batch of new observations, update our "running average"
        and "running variance" estimates efficiently.

        Args:
            x: Array of shape (batch, *shape) or (*shape,) for single sample
        """
        batch_mean = np.mean(x, axis=0)
        batch_var  = np.var(x, axis=0)
        batch_count = x.shape[0] if x.ndim > len(self.mean.shape) else 1

        # Parallel/Welford update formula — combines old stats with new batch stats
        total_count  = self.count + batch_count
        delta        = batch_mean - self.mean
        new_mean     = self.mean + delta * batch_count / total_count
        m_a          = self.var * self.count
        m_b          = batch_var * batch_count
        m2           = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var      = m2 / total_count

        self.mean, self.var, self.count = new_mean, new_var, total_count

    @property
    def std(self) -> np.ndarray:
        """Standard deviation (sqrt of variance)."""
        return np.sqrt(self.var + 1e-8)     # epsilon for numerical safety

    def normalise(self, x: np.ndarray) -> np.ndarray:
        """Subtract mean, divide by std → zero-mean, unit-variance."""
        return (x - self.mean) / self.std

    def state_dict(self) -> Dict:
        """Serialise for checkpointing."""
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: Dict) -> None:
        """Restore from checkpoint."""
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


# ══════════════════════════════════════════════════════════════════════════════
# §4  ROLLOUT BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Stores trajectories collected from the environment during rollout.

    ELI5: This is the "notebook" where we write down everything that happened
    during the game:
      - What state were we in?
      - What action did we take?
      - What reward did we get?
      - How likely was that action (old policy)?
      - What did the Critic think the state was worth?
      - Was this a terminal (game-over) state?

    After the episode, we compute advantages and returns from these notes,
    then use them to update the Actor and Critic.

    Supports MULTIPLE parallel environments (vectorised).
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        device: torch.device,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        action_space_type: str = "discrete",
    ):
        """
        Args:
            rollout_steps     : Steps per environment per rollout
            num_envs          : Number of parallel environments
            obs_shape         : Shape of a single observation
            act_shape         : Shape of a single action
            device            : Torch device to store tensors on
            gae_lambda        : GAE lambda parameter (λ)
            gamma             : Discount factor (γ)
            action_space_type : 'discrete' or 'continuous'
        """
        self.rollout_steps    = rollout_steps
        self.num_envs         = num_envs
        self.obs_shape        = obs_shape
        self.act_shape        = act_shape
        self.device           = device
        self.gae_lambda       = gae_lambda
        self.gamma            = gamma
        self.action_space_type = action_space_type

        # Pre-allocate tensors for efficiency
        # Shape: (rollout_steps, num_envs, *feature_shape)
        self.observations  = torch.zeros((rollout_steps, num_envs) + obs_shape, dtype=torch.float32)
        self.actions       = torch.zeros((rollout_steps, num_envs) + act_shape, dtype=torch.float32)
        self.rewards       = torch.zeros((rollout_steps, num_envs),             dtype=torch.float32)
        self.dones         = torch.zeros((rollout_steps, num_envs),             dtype=torch.float32)
        self.log_probs     = torch.zeros((rollout_steps, num_envs),             dtype=torch.float32)
        self.values        = torch.zeros((rollout_steps, num_envs),             dtype=torch.float32)

        # Computed after rollout (GAE)
        self.advantages    = torch.zeros((rollout_steps, num_envs),             dtype=torch.float32)
        self.returns       = torch.zeros((rollout_steps, num_envs),             dtype=torch.float32)

        self.ptr = 0            # Current write position in the buffer
        self.full = False       # Whether the buffer has been filled

    def reset(self) -> None:
        """Clear the buffer — reset pointer to start."""
        self.ptr  = 0
        self.full = False

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Store one timestep of experience across all parallel environments.

        ELI5: Write one row into our notebook. One row = one game step.
        We store the observation, action, reward, and the "metadata" (log prob, value).

        Args:
            obs     : Current observation  (num_envs, *obs_shape)
            action  : Action taken         (num_envs, *act_shape)
            reward  : Reward received      (num_envs,)
            done    : Terminal flag        (num_envs,)
            log_prob: Log prob of action   (num_envs,)
            value   : Critic value estimate(num_envs,)
        """
        assert self.ptr < self.rollout_steps, (
            f"Buffer overflow at ptr={self.ptr}. Call compute_returns_and_advantages() then reset()."
        )
        self.observations[self.ptr] = obs.cpu()
        self.actions[self.ptr]      = action.cpu()
        self.rewards[self.ptr]      = reward.cpu()
        self.dones[self.ptr]        = done.cpu()
        self.log_probs[self.ptr]    = log_prob.cpu()
        self.values[self.ptr]       = value.cpu()
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values: torch.Tensor, last_dones: torch.Tensor) -> None:
        """
        Compute Generalised Advantage Estimates (GAE) and returns.

        ELI5 of GAE:
          Say you're at step t. How GOOD was the action you took?
          Naively: actual_return - critic_estimate = advantage
          But this is noisy! GAE smooths this by blending multiple future steps.

          δ_t  = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
                 ↑ "TD error" = how much better/worse than expected (one step)

          A_t  = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
                 ↑ exponentially-weighted sum of future TD errors

          λ=0: A_t = δ_t (pure TD, low variance, biased)
          λ=1: A_t = G_t - V(s_t) (pure MC, unbiased, high variance)
          λ=0.95: blend that balances bias/variance

          Returns: R_t = A_t + V(s_t)  (advantage + baseline = estimated return)

        Args:
            last_values: Critic's value estimate for the state AFTER the last step
                         (needed to bootstrap if episode wasn't done) (num_envs,)
            last_dones : Whether the last step was terminal (num_envs,)
        """
        last_gae_lam = torch.zeros(self.num_envs, dtype=torch.float32)

        # Walk BACKWARD through time — this is how GAE is computed efficiently
        for step in reversed(range(self.rollout_steps)):
            if step == self.rollout_steps - 1:
                # After the last step, the "next" values come from last_values
                next_non_terminal = 1.0 - last_dones.float().cpu()
                next_values        = last_values.cpu()
            else:
                # "next" step is just step+1
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values        = self.values[step + 1]

            # TD Error (delta): how much better/worse than the critic predicted
            # ELI5: "You predicted 10 treats. You got 3 + predicted 8 next step.
            #        Delta = 3 + 0.99*8 - 10 = 0.92 (slightly better than expected)"
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )

            # GAE: accumulate TD errors backward with exponential decay
            # ELI5: "Stack up all future deltas, but discount them by (γλ) each step"
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        # Returns = advantages + baseline (critic's value estimate)
        # ELI5: "How much total reward do we expect from this state?"
        self.returns = self.advantages + self.values

    def get_batches(self, minibatch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield random mini-batches of experiences for gradient updates.

        ELI5: We have 2048 × 4 = 8192 experiences in the buffer.
        Instead of updating the network on all 8192 at once (memory hog),
        we shuffle them and take random groups of 64 to update from.
        This is standard mini-batch stochastic gradient descent.

        Args:
            minibatch_size: Number of samples per mini-batch

        Yields:
            Dictionary of tensors for one mini-batch
        """
        total_samples = self.rollout_steps * self.num_envs

        # Flatten (rollout_steps, num_envs) → (total_samples,) for each tensor
        # ELI5: Merge all parallel environments into one big list
        b_obs      = self.observations.reshape(total_samples, *self.obs_shape)
        b_actions  = self.actions.reshape(total_samples, *self.act_shape)
        b_log_probs= self.log_probs.reshape(total_samples)
        b_advantages = self.advantages.reshape(total_samples)
        b_returns  = self.returns.reshape(total_samples)
        b_values   = self.values.reshape(total_samples)

        # Random permutation of indices — shuffle the data before mini-batching
        indices = torch.randperm(total_samples)

        # Yield mini-batches of shuffled data
        for start in range(0, total_samples, minibatch_size):
            end   = start + minibatch_size
            mb_idx = indices[start:end]

            yield {
                "obs"       : b_obs[mb_idx].to(self.device),
                "actions"   : b_actions[mb_idx].to(self.device),
                "old_log_probs": b_log_probs[mb_idx].to(self.device),
                "advantages": b_advantages[mb_idx].to(self.device),
                "returns"   : b_returns[mb_idx].to(self.device),
                "old_values": b_values[mb_idx].to(self.device),
            }


# ══════════════════════════════════════════════════════════════════════════════
# §5  POLICY BACKBONES (Feature Extractors)
# ══════════════════════════════════════════════════════════════════════════════
#
# ELI5: The backbone is the "eyes and ears" of the agent.
#       It processes raw observations (pixels, joint angles, sensor readings)
#       into a compact feature vector that the Actor/Critic heads can use.

class MLPBackbone(nn.Module):
    """
    Multi-Layer Perceptron backbone — the default feature extractor.

    ELI5: A stack of fully-connected layers with non-linear activations between them.
    Input: observation vector (e.g. [position, velocity, angle] = 4 numbers for CartPole)
    Output: feature vector (e.g. 64 numbers)

    Suitable for: tabular/vector observations, physics simulations, robotics joint angles.
    Not suitable for: raw pixels (use CNN instead), sequences (use Transformer/LSTM).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        ortho_init: bool = True,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_size = obs_dim

        # Build hidden layers
        for h_size in hidden_sizes:
            lin = nn.Linear(in_size, h_size)
            if ortho_init:
                # Orthogonal init with scale=√2 is the PPO standard
                layer_init(lin, std=np.sqrt(2))
            layers.extend([lin, get_activation(activation)])
            in_size = h_size

        self.net = nn.Sequential(*layers)
        self.output_dim = in_size   # Used by Actor/Critic heads to know input size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Observation tensor (..., obs_dim)
        Returns:
            Feature tensor (..., hidden_sizes[-1])
        """
        return self.net(x)


class TransformerPolicyBackbone(nn.Module):
    """
    Transformer-based backbone for sequence / token observations.

    ELI5: Transformers are great at understanding "what relates to what" in a sequence.
    If your observation is a sequence of tokens (e.g., text, time-series sensor data,
    or a sequence of robot joint states over time), this backbone uses self-attention
    to extract rich relational features.

    Architecture:
      obs → Linear embedding → Positional Encoding → N×TransformerEncoderLayer → mean pool → features

    Args:
        obs_dim   : Dimension of each input token (or flat obs projected to d_model)
        seq_len   : Length of the input sequence (set to 1 for non-sequential use)
        d_model   : Internal transformer dimension (embedding size)
        nhead     : Number of attention heads
        n_layers  : Number of transformer encoder layers
        ffn_dim   : Feed-forward network dimension inside transformer
        dropout   : Dropout probability (0 = disabled for eval, useful for training)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Tuple[int, ...] = (64, 64),   # ignored — kept for interface compatibility
        activation: str = "gelu",
        ortho_init: bool = True,
        seq_len: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        n_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # Project input observations into transformer's d_model dimension
        # ELI5: Like translating your observation into the transformer's "language"
        self.input_proj = nn.Linear(obs_dim, d_model)
        if ortho_init:
            layer_init(self.input_proj, std=np.sqrt(2))

        # Positional encoding — tells the transformer WHERE in the sequence each token is
        # ELI5: Like numbering the pages in a book so the transformer knows order
        self.pos_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos_encoding, std=0.02)

        # Transformer encoder — the self-attention magic
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,       # (batch, seq, features) — easier to work with
            norm_first=True,        # Pre-norm for training stability (GPT-2 style)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Observation tensor.
               If seq_len=1: shape (..., obs_dim) — treated as single-step observation.
               If seq_len>1: shape (..., seq_len, obs_dim) — sequential observation.
        Returns:
            Feature tensor (..., d_model)
        """
        if x.dim() == 2:
            # (batch, obs_dim) → (batch, 1, obs_dim) — treat as length-1 sequence
            x = x.unsqueeze(1)

        # Project to d_model
        x = self.input_proj(x)          # (batch, seq_len, d_model)

        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Run through transformer
        x = self.transformer(x)         # (batch, seq_len, d_model)

        # Pool over the sequence dimension → single feature vector per sample
        # ELI5: "Summarise the whole sequence into one compact representation"
        x = x.mean(dim=1)              # (batch, d_model)
        return x


class CNNBackbone(nn.Module):
    """
    Convolutional Neural Network backbone for image/pixel observations.

    ELI5: CNNs are great at processing images. They use sliding "filters" that
    detect local patterns (edges, textures, shapes) and compose them into
    higher-level features (faces, objects, game elements).

    Suitable for: Atari games, visual robotics, any pixel-based observation.

    Architecture (Nature DQN style):
      (C, H, W) image → Conv(32,8,4) → Conv(64,4,2) → Conv(64,3,1) → Flatten → Linear(512)
    """

    def __init__(
        self,
        obs_dim: int,               # Actually treated as (C, H, W) tuple for images
        hidden_sizes: Tuple[int, ...] = (512,),
        activation: str = "relu",
        ortho_init: bool = True,
        channels: Tuple[int, ...] = (32, 64, 64),
        kernels:  Tuple[int, ...] = (8, 4, 3),
        strides:  Tuple[int, ...] = (4, 2, 1),
        in_channels: int = 4,       # Number of stacked frames (e.g., 4 for Atari)
    ):
        super().__init__()
        conv_layers: List[nn.Module] = []
        ch_in = in_channels

        for ch_out, k, s in zip(channels, kernels, strides):
            conv = nn.Conv2d(ch_in, ch_out, kernel_size=k, stride=s)
            if ortho_init:
                nn.init.orthogonal_(conv.weight, gain=np.sqrt(2))
                nn.init.constant_(conv.bias, 0)
            conv_layers.extend([conv, get_activation(activation)])
            ch_in = ch_out

        self.conv = nn.Sequential(*conv_layers)

        # Dummy forward to compute flattened CNN output size
        # ELI5: Run a fake image through the convolutions to see how big the output is
        with torch.no_grad():
            # Assume obs_dim is flattened image size; try to infer spatial dims
            dummy = torch.zeros(1, in_channels, 84, 84)  # Standard Atari size
            flat_size = int(np.prod(self.conv(dummy).shape[1:]))

        # Fully-connected head after convolutions
        fc_layers: List[nn.Module] = []
        in_size = flat_size
        for h in hidden_sizes:
            lin = nn.Linear(in_size, h)
            if ortho_init:
                layer_init(lin, std=np.sqrt(2))
            fc_layers.extend([lin, get_activation(activation)])
            in_size = h

        self.fc = nn.Sequential(*fc_layers)
        self.output_dim = in_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (batch, C, H, W) — values in [0, 255] or [0, 1]
        Returns:
            Feature tensor (batch, output_dim)
        """
        # Scale pixel values to [0, 1] if in [0, 255]
        if x.max() > 1.0:
            x = x / 255.0
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


# ══════════════════════════════════════════════════════════════════════════════
# §6  ACTOR-CRITIC NETWORKS
# ══════════════════════════════════════════════════════════════════════════════
#
# ELI5: Two "brains":
#   Actor  → the DECISION-MAKER → "given this situation, what should I do?"
#   Critic → the EVALUATOR     → "given this situation, how good is it?"
#
# During rollout: use Actor to pick actions, Critic to estimate value.
# During update:  optimise both using PPO loss.

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic module supporting:
      - Discrete actions  (Categorical distribution)
      - Continuous actions (Diagonal Gaussian / Beta distribution)
      - Shared or separate backbones for Actor and Critic
      - Pluggable feature extractor (MLP, Transformer, CNN, or custom)

    ELI5: This is the whole "brain" of the agent in one class.
    The backbone processes observations → features.
    The actor_head  turns features → action probabilities / mean.
    The critic_head turns features → a single number (value estimate).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        cfg: PPOConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.act_dim = act_dim
        self.action_space_type = cfg.action_space_type
        self.continuous_dist   = cfg.continuous_dist

        # ── Select backbone class ─────────────────────────────────────────
        # ELI5: Pick WHICH type of feature extractor to use.
        # Default: MLP. User can override with Transformer, CNN, or custom.
        BackboneCls = cfg.backbone_cls if cfg.backbone_cls is not None else MLPBackbone
        backbone_kwargs = dict(
            obs_dim      = obs_dim,
            hidden_sizes = cfg.hidden_sizes,
            activation   = cfg.activation,
            ortho_init   = cfg.ortho_init,
        )

        if cfg.shared_backbone:
            # ONE backbone shared by both actor and critic heads
            # Saves parameters; actor and critic "see the world" the same way
            self.shared_net = BackboneCls(**backbone_kwargs)
            feat_dim = self.shared_net.output_dim
            self.actor_backbone  = None
            self.critic_backbone = None
        else:
            # SEPARATE backbones — actor and critic learn independently
            # More parameters but often better for complex tasks
            self.actor_backbone  = BackboneCls(**backbone_kwargs)
            self.critic_backbone = BackboneCls(**backbone_kwargs)
            feat_dim = self.actor_backbone.output_dim
            self.shared_net = None

        # ── Critic head ────────────────────────────────────────────────────
        # ELI5: One output number = "this state is worth X future reward"
        self.critic_head = nn.Linear(feat_dim, 1)
        if cfg.ortho_init:
            # Small std for critic head → stable value estimates at start
            layer_init(self.critic_head, std=1.0)

        # ── Actor head(s) ──────────────────────────────────────────────────
        if cfg.action_space_type == "discrete":
            # Output: logits for each action (will be softmaxed into probabilities)
            # ELI5: "I have 5% chance of going left, 90% right, 5% jump"
            self.actor_head = nn.Linear(feat_dim, act_dim)
            if cfg.ortho_init:
                # Very small std for action logits → near-uniform distribution initially
                # ELI5: Start by choosing actions almost randomly, then learn preferences
                layer_init(self.actor_head, std=0.01)

        elif cfg.action_space_type == "continuous":
            if cfg.continuous_dist == "gaussian":
                # Mean network: maps features → action mean vector
                # ELI5: "Turn the steering wheel by about 0.3 radians (the mean)"
                self.actor_mean = nn.Linear(feat_dim, act_dim)
                if cfg.ortho_init:
                    layer_init(self.actor_mean, std=0.01)

                # Log-std: LEARNABLE per-action exploration spread
                # ELI5: "How uncertain am I about this action?"
                # Initialised to log_std_init (not a function of observations)
                self.log_std = nn.Parameter(
                    torch.full((act_dim,), cfg.log_std_init)
                )

            elif cfg.continuous_dist == "beta":
                # Beta distribution lives on [0, 1] — good for bounded actions
                # Parameterised by α (alpha) and β (beta) concentration params
                self.actor_alpha = nn.Linear(feat_dim, act_dim)
                self.actor_beta  = nn.Linear(feat_dim, act_dim)
                if cfg.ortho_init:
                    layer_init(self.actor_alpha, std=0.01)
                    layer_init(self.actor_beta, std=0.01)
            else:
                raise ValueError(f"Unknown continuous_dist: {cfg.continuous_dist}")
        else:
            raise ValueError(f"Unknown action_space_type: {cfg.action_space_type}")

    def _get_features(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from observation using the backbone(s).

        Returns:
            actor_features : Features for the actor head
            critic_features: Features for the critic head
        """
        if self.shared_net is not None:
            # Single shared backbone
            feats = self.shared_net(obs)
            return feats, feats
        else:
            # Separate backbones
            return self.actor_backbone(obs), self.critic_backbone(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get critic's value estimate for an observation.
        ELI5: "How many future treats do you expect from this game state?"

        Args:
            obs: Observation tensor (batch, obs_dim)
        Returns:
            Value tensor (batch, 1)
        """
        _, critic_feats = self._get_features(obs)
        return self.critic_head(critic_feats)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Main method used during rollout (sampling) AND during update (evaluating).

        During ROLLOUT: pass no action → sample a new action from the policy.
        During UPDATE:  pass the OLD action → evaluate it under the NEW policy.

        ELI5:
          Rollout: "What should I do?" → sample action from distribution.
          Update:  "How would the NEW me have evaluated that OLD action?" → compute log-prob.

        Args:
            obs   : Observation tensor (batch, obs_dim)
            action: (Optional) action tensor. If None, sample from distribution.

        Returns:
            action    : The (sampled or provided) action
            log_prob  : Log probability of the action under current policy
            entropy   : Entropy of the action distribution (for exploration bonus)
            value     : Critic's value estimate
        """
        actor_feats, critic_feats = self._get_features(obs)
        value = self.critic_head(critic_feats)

        # ── Build action distribution ──────────────────────────────────────
        if self.action_space_type == "discrete":
            # Categorical: "which bin (action) to sample from?"
            # ELI5: Like rolling a loaded die — each face has a probability
            logits = self.actor_head(actor_feats)
            dist   = Categorical(logits=logits)

        elif self.action_space_type == "continuous":
            if self.continuous_dist == "gaussian":
                mean   = self.actor_mean(actor_feats)
                # Clamp log_std to prevent numerical instability
                # ELI5: Don't let the "uncertainty" become infinitely large or zero
                log_std = torch.clamp(self.log_std, min=-20.0, max=2.0)
                std     = log_std.exp()
                dist    = Normal(mean, std)

                if self.cfg.squash_actions:
                    # Use tanh-squashed Gaussian (like SAC)
                    # ELI5: Sample from Gaussian, then squash into [-1, 1] range
                    # Requires correction to the log probability (Jacobian adjustment)
                    pass  # Handled below when computing log_prob

            elif self.continuous_dist == "beta":
                # Beta distribution: outputs in [0, 1]
                # α = softplus(alpha_net), β = softplus(beta_net) — ensure positivity
                alpha = F.softplus(self.actor_alpha(actor_feats)) + 1.0
                beta  = F.softplus(self.actor_beta(actor_feats)) + 1.0
                dist  = Beta(alpha, beta)

        # ── Sample or evaluate action ──────────────────────────────────────
        if action is None:
            # ROLLOUT: sample a new action
            action = dist.sample()

            if (self.action_space_type == "continuous"
                    and self.continuous_dist == "gaussian"
                    and self.cfg.squash_actions):
                # Apply tanh squashing AFTER sampling
                action = torch.tanh(action)

        # Compute log probability of the action
        if (self.action_space_type == "continuous"
                and self.continuous_dist == "gaussian"
                and self.cfg.squash_actions):
            # For tanh-squashed Gaussian, use the pre-squash action for log-prob
            # and apply Jacobian correction:  log π(a) = log π(u) - log(1 - tanh²(u))
            pre_tanh = torch.atanh(action.clamp(-0.9999, 0.9999))
            log_prob = dist.log_prob(pre_tanh).sum(-1)
            log_prob -= (2.0 * (np.log(2) - pre_tanh - F.softplus(-2.0 * pre_tanh))).sum(-1)
        else:
            if self.action_space_type == "discrete":
                log_prob = dist.log_prob(action)
            else:
                # For continuous: sum log probs across action dimensions
                # ELI5: If action has 3 dimensions, the joint probability is the product
                #        → in log space, that's the sum
                log_prob = dist.log_prob(action).sum(-1)

        # Entropy: measure of "how random/exploratory" the policy is
        # ELI5: High entropy = agent is uncertain → exploring.
        #       Low entropy  = agent is confident → exploiting.
        if self.action_space_type == "discrete":
            entropy = dist.entropy()
        else:
            entropy = dist.entropy().sum(-1)

        return action, log_prob, entropy, value.squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# §7  ENVIRONMENT WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════
#
# ELI5: Environments are the "game world". Wrappers are like controllers
#       that add extra features (e.g., normalisation, stacking frames, parallelism).

class VecEnv:
    """
    Abstract base class for vectorised (parallel) environments.

    ELI5: Instead of one game copy, run N copies simultaneously.
    This multiplies data collection speed by N with minimal overhead.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """Reset all environments. Returns initial observations (num_envs, obs_dim)."""
        ...

    @abc.abstractmethod
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments.
        Returns: obs, rewards, dones, infos
        """
        ...

    @property
    @abc.abstractmethod
    def observation_space(self): ...

    @property
    @abc.abstractmethod
    def action_space(self): ...


class SyncVecEnv(VecEnv):
    """
    Synchronous vectorised environment — runs N envs sequentially.

    ELI5: Like running 4 games one after the other in the same process.
    Simple, reliable, no multiprocessing headaches.
    For async (true parallel), use AsyncVecEnv (not implemented here — use stable-baselines3).

    Args:
        env_fns: List of callables, each returning a new gym.Env instance.
                 One callable per parallel environment.
    """

    def __init__(self, env_fns: List[Callable[[], Any]]):
        super().__init__(num_envs=len(env_fns))
        self.envs = [fn() for fn in env_fns]
        self._obs_space = self.envs[0].observation_space
        self._act_space = self.envs[0].action_space

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    def reset(self) -> np.ndarray:
        """
        Reset all environments and return initial observations.

        ELI5: Start all games from scratch. Returns the initial "screenshot"
        of each game copy.
        """
        obs_list = []
        for env in self.envs:
            result = env.reset()
            # gymnasium returns (obs, info); gym returns obs
            obs = result[0] if isinstance(result, tuple) else result
            obs_list.append(obs)
        return np.stack(obs_list, axis=0)     # (num_envs, obs_dim)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Apply actions to all environments, collect results.

        ELI5: Take one step in EACH game copy. Collect observations, rewards,
        and done-flags from all copies. If a game ends (done=True), auto-reset it.

        Args:
            actions: (num_envs, *act_shape) or (num_envs,) for discrete
        Returns:
            obs    : Next observations  (num_envs, obs_dim)
            rewards: Reward per env     (num_envs,)
            dones  : Terminal flags     (num_envs,)
            infos  : Extra info dicts   [num_envs]
        """
        obs_list, rews, dones, infos = [], [], [], []
        for i, (env, act) in enumerate(zip(self.envs, actions)):
            result = env.step(act)

            if len(result) == 5:
                # gymnasium: (obs, reward, terminated, truncated, info)
                o, r, terminated, truncated, info = result
                done = terminated or truncated
            else:
                # gym: (obs, reward, done, info)
                o, r, done, info = result

            if done:
                # Auto-reset: immediately restart the environment after it ends
                # ELI5: When the game is over, start a new one automatically.
                # We still return the TERMINAL observation in info['terminal_obs'].
                if isinstance(info, dict):
                    info["terminal_obs"] = o
                reset_result = env.reset()
                o = reset_result[0] if isinstance(reset_result, tuple) else reset_result

            obs_list.append(o)
            rews.append(r)
            dones.append(done)
            infos.append(info if info else {})

        return (
            np.stack(obs_list, axis=0),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            infos,
        )

    def close(self) -> None:
        """Close all environments cleanly."""
        for env in self.envs:
            env.close()


def make_gymnasium_env(env_id: str, seed: int, idx: int, **kwargs) -> Callable:
    """
    Factory function: returns a callable that creates one gym environment.

    ELI5: Like a cookie cutter — call this to get a function that makes
    one specific game copy with the right settings.

    Args:
        env_id: Gymnasium environment ID (e.g., 'CartPole-v1')
        seed  : Base random seed
        idx   : Environment index (seed offset, so each env gets a unique seed)
        kwargs: Extra kwargs passed to gym.make()

    Returns:
        Callable → gym.Env (called later by SyncVecEnv)
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return _init


# ══════════════════════════════════════════════════════════════════════════════
# §8  LR SCHEDULE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

class ScheduledValue:
    """
    A value that changes according to a schedule over training.

    ELI5: Imagine turning down the volume gradually on a song.
    'linear' decay goes from initial_value → 0 linearly.
    'cosine' decay follows a cosine curve (smooth, slower end).
    'constant' stays at the same value forever.

    Used for: learning rate, clip_epsilon, entropy_coef.
    """

    def __init__(self, initial: float, schedule: str, total_steps: int):
        """
        Args:
            initial    : Starting value
            schedule   : 'constant', 'linear', 'cosine', 'warmup_cosine'
            total_steps: Total number of update steps for scheduling
        """
        self.initial     = initial
        self.schedule    = schedule
        self.total_steps = max(total_steps, 1)

    def __call__(self, step: int) -> float:
        """
        Get the scheduled value at a given step.

        Args:
            step: Current update step (0-indexed)
        Returns:
            Scheduled value
        """
        progress = step / self.total_steps  # 0.0 at start, 1.0 at end

        if self.schedule == "constant" or self.schedule == "none":
            return self.initial

        elif self.schedule == "linear":
            # Linearly decay from initial to 0
            # ELI5: Like a candle burning down at a steady rate
            return self.initial * max(0.0, 1.0 - progress)

        elif self.schedule == "cosine":
            # Cosine annealing: smooth decay following a cosine curve
            # ELI5: Like a ball rolling down a smooth hill — slows at the bottom
            return self.initial * 0.5 * (1.0 + math.cos(math.pi * progress))

        elif self.schedule == "warmup_cosine":
            # Linear warmup for first 10%, then cosine decay
            # ELI5: Warm up the engine slowly, then let it run down smoothly
            warmup_steps = 0.1 * self.total_steps
            if step < warmup_steps:
                return self.initial * (step / warmup_steps)
            else:
                progress_after_warmup = (step - warmup_steps) / (self.total_steps - warmup_steps)
                return self.initial * 0.5 * (1.0 + math.cos(math.pi * progress_after_warmup))

        else:
            raise ValueError(f"Unknown schedule: '{self.schedule}'")


# ══════════════════════════════════════════════════════════════════════════════
# §9  LOGGER
# ══════════════════════════════════════════════════════════════════════════════

class TrainingLogger:
    """
    Handles all logging: TensorBoard, CSV, Weights & Biases, and console.

    ELI5: Keeps a diary of training. Every few updates, it writes down
    reward, loss, learning rate, etc. so you can see how training is going
    and debug problems early.
    """

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.log_dir = Path(cfg.log_dir) / cfg.experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        #self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
        #logger.info(f"TensorBoard logs: {self.log_dir / 'tensorboard'}")

        # CSV log file
        self.csv_path = self.log_dir / "training_log.csv"
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = None     # Initialised on first log call (to get headers)
        logger.info(f"CSV log: {self.csv_path}")

        # Weights & Biases
        if cfg.use_wandb:
            if not _WANDB_AVAILABLE:
                warnings.warn("wandb not installed. Install with: pip install wandb")
            else:
                wandb.init(
                    project=cfg.wandb_project,
                    name=cfg.experiment_name,
                    config=vars(cfg),
                    sync_tensorboard=True,  # Auto-sync TensorBoard to W&B
                )
                logger.info("W&B run initialised.")

        # Episode reward tracking (deque = sliding window)
        # ELI5: Keep the last 100 episode rewards to compute a rolling average
        self.ep_rewards: deque = deque(maxlen=100)
        self.ep_lengths: deque = deque(maxlen=100)

    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """
        Log a dictionary of metrics at a given training step.

        Args:
            step   : Current update step
            metrics: Dict of metric_name → value
        """
        # TensorBoard
        #for k, v in metrics.items():
        #    if v is not None:
        #        self.tb_writer.add_scalar(k, v, step)

        # CSV (write header on first call)
        if self._csv_writer is None:
            fieldnames = ["step"] + sorted(metrics.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            self._csv_writer.writeheader()
        row = {"step": step}
        row.update({k: (f"{v:.6f}" if v is not None else "") for k, v in metrics.items()})
        self._csv_writer.writerow(row)
        self._csv_file.flush()

        # W&B
        if self.cfg.use_wandb and _WANDB_AVAILABLE:
            wandb.log({"step": step, **metrics})

    def log_episode(self, reward: float, length: int) -> None:
        """Track completed episode stats."""
        self.ep_rewards.append(reward)
        self.ep_lengths.append(length)

    @property
    def mean_reward(self) -> float:
        """Rolling mean reward over last 100 episodes."""
        return float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0

    @property
    def mean_length(self) -> float:
        """Rolling mean episode length over last 100 episodes."""
        return float(np.mean(self.ep_lengths)) if self.ep_lengths else 0.0

    def close(self) -> None:
        """Clean up all logging resources."""
        #self.tb_writer.close()
        self._csv_file.close()
        if self.cfg.use_wandb and _WANDB_AVAILABLE:
            wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# §10  PPO AGENT — THE MAIN CLASS
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    Production-grade Proximal Policy Optimisation (PPO) agent.

    Implements the full PPO training loop:
      1. ROLLOUT   : Run current policy in environment, collect experiences.
      2. GAE       : Compute advantages and returns using GAE.
      3. UPDATE    : Update Actor and Critic using PPO loss for N epochs.
      4. REPEAT    : Repeat until total_timesteps reached.

    Supports:
      - Discrete and continuous action spaces
      - Observation and reward normalisation
      - LR, clip_epsilon, entropy_coef scheduling
      - Gradient clipping
      - Value function clipping
      - KL-divergence early stopping
      - Mixed precision (AMP)
      - Full checkpointing (save/resume)
      - TensorBoard + W&B + CSV logging
      - Pluggable policy backbones (MLP, Transformer, CNN, custom)
      - Vectorised parallel environments

    Usage:
        cfg   = PPOConfig(env_id="CartPole-v1", total_timesteps=500_000)
        agent = PPOAgent(cfg)
        agent.learn()
    """

    def __init__(
        self,
        cfg: PPOConfig,
        env_factory: Optional[Callable[[int], Any]] = None,
    ):
        """
        Args:
            cfg        : PPOConfig dataclass with all hyper-parameters.
            env_factory: Optional callable(idx: int) → gym.Env.
                         If None, uses cfg.env_id with gymnasium.
                         Provide this to use custom environments.
        """
        self.cfg = cfg
        self.device = get_device(cfg.device)
        set_seed(cfg.seed)

        # ── Build environments ──────────────────────────────────────────────
        if env_factory is not None:
            # User-provided environment factory
            # ELI5: User knows best what game to play — use their setup
            env_fns = [lambda i=i: env_factory(i) for i in range(cfg.num_envs)]
        elif _GYM_AVAILABLE:
            # Standard Gymnasium environment
            env_fns = [
                make_gymnasium_env(cfg.env_id, cfg.seed, i)
                for i in range(cfg.num_envs)
            ]
        else:
            raise RuntimeError(
                "No environment available. Install gymnasium: pip install gymnasium, "
                "or provide env_factory."
            )

        self.envs = SyncVecEnv(env_fns)

        # ── Extract space dimensions ────────────────────────────────────────
        obs_space = self.envs.observation_space
        act_space = self.envs.action_space

        # Observation shape (supports Box observations)
        # ELI5: "How many numbers describe the game state?"
        self.obs_shape = obs_space.shape
        self.obs_dim   = int(np.prod(self.obs_shape))

        # Action dimensions
        if cfg.action_space_type == "discrete":
            self.act_dim   = act_space.n           # Number of discrete choices
            self.act_shape = ()
        else:
            self.act_dim   = act_space.shape[0]    # Dimension of continuous action
            self.act_shape = (self.act_dim,)

        logger.info(
            f"Environment: {cfg.env_id} | obs_shape={self.obs_shape} | "
            f"act_dim={self.act_dim} | action_type={cfg.action_space_type}"
        )

        # ── Build Actor-Critic ──────────────────────────────────────────────
        self.policy = ActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            cfg=cfg,
        ).to(self.device)

        total_params = sum(p.numel() for p in self.policy.parameters())
        logger.info(f"Policy network: {total_params:,} parameters")

        # ── Build Optimizer ─────────────────────────────────────────────────
        # ELI5: The optimizer is the "learning mechanism" that adjusts weights.
        # Adam is most popular: it adapts the learning rate for each weight individually.
        optim_cls = {
            "adam":  torch.optim.Adam,
            "adamw": torch.optim.AdamW,
            "sgd":   torch.optim.SGD,
        }.get(cfg.optimizer.lower(), torch.optim.Adam)

        optim_kwargs = dict(lr=cfg.lr_actor, eps=cfg.adam_eps, weight_decay=cfg.weight_decay)
        if cfg.optimizer.lower() == "sgd":
            optim_kwargs = dict(lr=cfg.lr_actor, weight_decay=cfg.weight_decay, momentum=0.9)

        self.optimizer = optim_cls(self.policy.parameters(), **optim_kwargs)

        # ── Running Statistics ──────────────────────────────────────────────
        # ELI5: Track the "average" and "spread" of observations and rewards
        #       so we can normalise them during training.
        self.obs_rms    = RunningMeanStd(shape=self.obs_shape) if cfg.normalise_obs    else None
        self.reward_rms = RunningMeanStd(shape=())             if cfg.normalise_rewards else None

        # ── Rollout Buffer ──────────────────────────────────────────────────
        self.buffer = RolloutBuffer(
            rollout_steps     = cfg.rollout_steps,
            num_envs          = cfg.num_envs,
            obs_shape         = self.obs_shape,
            act_shape         = self.act_shape,
            device            = self.device,
            gae_lambda        = cfg.gae_lambda,
            gamma             = cfg.gamma,
            action_space_type = cfg.action_space_type,
        )

        # ── Compute total update steps ──────────────────────────────────────
        # ELI5: How many times will we update the policy in total?
        # One update = collect rollout_steps × num_envs steps, then run N epochs
        self.steps_per_update = cfg.rollout_steps * cfg.num_envs
        self.total_updates    = int(cfg.total_timesteps // self.steps_per_update)
        logger.info(
            f"Total timesteps: {cfg.total_timesteps:,} | "
            f"Updates: {self.total_updates} | "
            f"Steps/update: {self.steps_per_update:,}"
        )

        # ── Scheduled Values ────────────────────────────────────────────────
        self.lr_schedule      = ScheduledValue(cfg.lr_actor,       cfg.lr_schedule,             self.total_updates)
        self.clip_schedule    = ScheduledValue(cfg.clip_epsilon,    cfg.clip_epsilon_schedule,   self.total_updates)
        self.entropy_schedule = ScheduledValue(cfg.entropy_coef,    cfg.entropy_coef_schedule,   self.total_updates)

        # ── Mixed Precision ─────────────────────────────────────────────────
        # ELI5: AMP uses 16-bit floats where safe → up to 2× faster on GPUs
        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            logger.info("Mixed precision (AMP) enabled.")

        # ── Logger ──────────────────────────────────────────────────────────
        self.training_logger = TrainingLogger(cfg)

        # ── State variables ──────────────────────────────────────────────────
        self.global_step  = 0       # Total env steps taken
        self.update_count = 0       # Number of policy updates performed
        self._ep_rewards  = np.zeros(cfg.num_envs, dtype=np.float32)   # Accumulate per-env episode reward
        self._ep_lengths  = np.zeros(cfg.num_envs, dtype=np.int32)

        # ── Resume from checkpoint ──────────────────────────────────────────
        if cfg.resume_from:
            self.load_checkpoint(cfg.resume_from)

    # ──────────────────────────────────────────────────────────────────────────
    # §10.1  OBSERVATION NORMALISATION
    # ──────────────────────────────────────────────────────────────────────────

    def _normalise_obs(self, obs: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """
        Normalise observations using running mean/std.

        ELI5: If the ant robot's leg angles are in degrees (0-360) and velocities
        are in m/s (0-5), the network has a hard time because they're different scales.
        Normalising makes all inputs roughly zero-mean, unit-variance → easier to learn.

        Args:
            obs         : Raw observation (num_envs, *obs_shape)
            update_stats: Whether to update the running stats (True during rollout, False during eval)
        Returns:
            Normalised observation array
        """
        if self.obs_rms is None:
            return obs
        if update_stats:
            self.obs_rms.update(obs)
        return self.obs_rms.normalise(obs).astype(np.float32)

    def _normalise_rewards(self, rewards: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """
        Normalise rewards by running std (preserves sign).

        ELI5: If we clip and normalise rewards, sparse-reward environments
        (where most rewards are 0 with occasional +1) become easier to learn from.

        Note: We only divide by std (NOT subtract mean) to preserve reward sign.
        """
        if self.reward_rms is None:
            return rewards
        if update_stats:
            self.reward_rms.update(rewards)
        normalised = rewards / (self.reward_rms.std + 1e-8)
        return np.clip(normalised, -self.cfg.reward_clip, self.cfg.reward_clip).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # §10.2  ROLLOUT COLLECTION
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def collect_rollout(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Collect rollout_steps × num_envs environment transitions.

        ELI5: Play the game for rollout_steps steps using the CURRENT policy,
        writing down everything that happened into the buffer.
        We use torch.no_grad() because we're just collecting data — not training yet.

        Args:
            obs: Current observations (num_envs, *obs_shape)

        Returns:
            obs    : Updated observations at the end of rollout
            ep_time: Time taken for the rollout (seconds)
        """
        self.buffer.reset()
        t0 = time.time()

        for step in range(self.cfg.rollout_steps):
            self.global_step += self.cfg.num_envs

            # Normalise observations
            obs_norm = self._normalise_obs(obs, update_stats=True)
            obs_tensor = torch.FloatTensor(obs_norm).to(self.device)

            # Get action from policy (no gradient needed)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                action, log_prob, _, value = self.policy.get_action_and_value(obs_tensor)

            # Step ALL environments simultaneously
            action_np = action.cpu().numpy()
            if self.cfg.action_space_type == "discrete":
                action_np = action_np.astype(int)

            next_obs, reward, done, infos = self.envs.step(action_np)

            # Normalise rewards
            reward_norm = self._normalise_rewards(reward, update_stats=True)

            # Track episode statistics
            self._ep_rewards += reward
            self._ep_lengths += 1
            for i, (d, info) in enumerate(zip(done, infos)):
                if d:
                    self.training_logger.log_episode(self._ep_rewards[i], self._ep_lengths[i])
                    self._ep_rewards[i] = 0.0
                    self._ep_lengths[i] = 0

            # Store in buffer
            self.buffer.add(
                obs=obs_tensor.cpu(),
                action=action.cpu(),
                reward=torch.FloatTensor(reward_norm),
                done=torch.FloatTensor(done),
                log_prob=log_prob.cpu(),
                value=value.cpu(),
            )

            obs = next_obs

        # Bootstrap value for the last state (needed if episode isn't done)
        # ELI5: "What does the Critic think the state AFTER the last step is worth?"
        #        This is used to compute advantages for the last few steps.
        obs_norm = self._normalise_obs(obs, update_stats=False)
        obs_tensor = torch.FloatTensor(obs_norm).to(self.device)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            last_value = self.policy.get_value(obs_tensor).squeeze(-1).cpu()

        last_done = torch.FloatTensor(done)
        self.buffer.compute_returns_and_advantages(last_value, last_done)

        return obs, time.time() - t0

    # ──────────────────────────────────────────────────────────────────────────
    # §10.3  PPO LOSS & UPDATE
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_ppo_loss(
        self,
        mb: Dict[str, torch.Tensor],
        clip_epsilon: float,
        entropy_coef: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the full PPO loss for one mini-batch.

        ELI5 of PPO loss:
          1. ACTOR LOSS (Clipped Surrogate Objective):
             - Ratio r_t(θ) = π_new(a|s) / π_old(a|s)  → in log space: exp(new_logprob - old_logprob)
             - "The ratio of NEW to OLD probability of taking action a in state s."
             - Unclipped objective: r_t * A_t  (if better action, use it more)
             - Clipped objective:   clip(r_t, 1-ε, 1+ε) * A_t  (don't change policy too much)
             - Take the MIN of both → pessimistic bound → prevents over-optimistic updates
             - Negate it (because PyTorch minimises, but we want to maximise reward)

          2. CRITIC LOSS (Value Function MSE):
             - Predict state value V(s), minimise (V(s) - actual_return)²
             - Optional: also clip value updates (like PPO clips policy updates)

          3. ENTROPY BONUS:
             - Add entropy H[π] to encourage exploration
             - Entropy is "how random/exploratory is the policy right now?"
             - High entropy → exploring more options. We want to maintain some.

          TOTAL LOSS = -actor_loss + value_coef * critic_loss - entropy_coef * entropy
          (minus signs because we MAXIMISE actor obj and entropy, but MINIMISE losses)

        Args:
            mb          : Mini-batch dictionary from RolloutBuffer.get_batches()
            clip_epsilon: Current clipping parameter value (may be scheduled)
            entropy_coef: Current entropy coefficient (may be scheduled)

        Returns:
            loss  : Combined PPO loss scalar
            stats : Dictionary of loss components for logging
        """
        obs         = mb["obs"]
        actions     = mb["actions"]
        old_lp      = mb["old_log_probs"]
        advantages  = mb["advantages"]
        returns     = mb["returns"]
        old_values  = mb["old_values"]

        # Normalise advantages within this mini-batch
        # ELI5: Make advantages zero-mean and unit-variance so gradient magnitudes
        #       are consistent regardless of the reward scale.
        if self.cfg.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # For discrete, actions must be integer type for log_prob
        if self.cfg.action_space_type == "discrete":
            actions = actions.long()

        # Get new log-probs, entropy, and values from the CURRENT policy
        # ELI5: "Under the CURRENT (updated) policy, how likely would we have taken
        #        that action in that state?"
        _, new_log_probs, entropy, new_values = self.policy.get_action_and_value(obs, actions)

        # ── Clipped Surrogate (Actor) Loss ─────────────────────────────────
        # Probability ratio: how much has the policy changed for this action?
        # ELI5: If old π gave 10% chance and new π gives 15%, ratio = 1.5
        log_ratio = new_log_probs - old_lp
        ratio     = log_ratio.exp()

        # For monitoring: approximate KL divergence between old and new policy
        # KL ≈ (ratio - 1) - log(ratio)  [first-order approximation]
        # ELI5: "How different is the new policy from the old one?"
        with torch.no_grad():
            approx_kl = ((ratio - 1.0) - log_ratio).mean().item()

        # Two versions of the surrogate objective:
        surr1 = ratio * advantages                                              # Unclipped
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages  # Clipped

        # PPO actor loss: take the pessimistic (minimum) of both
        # Negate because we MAXIMISE this (but torch minimises → negate)
        # ELI5: "Take whichever update is more conservative to prevent going too far"
        actor_loss = -torch.min(surr1, surr2).mean()

        # Proportion of updates where clipping was active (diagnostic)
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()

        # ── Value Function (Critic) Loss ───────────────────────────────────
        if self.cfg.value_clip_epsilon is not None:
            # Clipped value loss: don't let the value estimate jump too far from old estimate
            # ELI5: Same idea as PPO's clipped policy — don't let Critic change too aggressively
            value_clipped = old_values + torch.clamp(
                new_values - old_values,
                -self.cfg.value_clip_epsilon,
                +self.cfg.value_clip_epsilon,
            )
            v_loss_unclipped = (new_values   - returns) ** 2
            v_loss_clipped   = (value_clipped - returns) ** 2
            critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            # Standard MSE loss (no value clipping)
            critic_loss = 0.5 * F.mse_loss(new_values, returns)

        # ── Entropy Bonus ───────────────────────────────────────────────────
        # ELI5: Reward the agent for being "uncertain" → keeps it exploring
        entropy_loss = -entropy.mean()  # Negate because we want to MAXIMISE entropy

        # ── Combined Loss ───────────────────────────────────────────────────
        loss = (
            actor_loss
            + self.cfg.value_coef  * critic_loss
            + entropy_coef         * entropy_loss
        )

        stats = {
            "train/actor_loss"   : actor_loss.item(),
            "train/critic_loss"  : critic_loss.item(),
            "train/entropy"      : -entropy_loss.item(),
            "train/approx_kl"   : approx_kl,
            "train/clip_fraction": clip_fraction,
            "train/total_loss"   : loss.item(),
        }
        return loss, stats

    def update_policy(self) -> Dict[str, float]:
        """
        Perform N epochs of PPO gradient updates on collected rollout data.

        ELI5: We collected a big batch of experiences. Now we re-read them
        multiple times (epochs), each time updating the Actor and Critic
        a little bit. We stop early if the policy changed too much (KL check).

        Returns:
            Dictionary of averaged metrics across all mini-batch updates
        """
        cfg = self.cfg

        # Get current scheduled values
        clip_epsilon = self.clip_schedule(self.update_count)
        entropy_coef = self.entropy_schedule(self.update_count)

        # Update learning rate
        lr = self.lr_schedule(self.update_count)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        all_stats: List[Dict[str, float]] = []
        early_stopped = False

        for epoch in range(cfg.n_epochs):
            for mb in self.buffer.get_batches(cfg.minibatch_size):
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    loss, stats = self._compute_ppo_loss(mb, clip_epsilon, entropy_coef)

                all_stats.append(stats)

                # KL early stopping: stop training if policy changed too much
                # ELI5: "The gap between old and new policy is getting too wide — stop now!"
                if cfg.target_kl is not None and stats["train/approx_kl"] > 1.5 * cfg.target_kl:
                    logger.debug(
                        f"Early stopping at epoch {epoch} due to KL {stats['train/approx_kl']:.4f} "
                        f"> target {1.5 * cfg.target_kl:.4f}"
                    )
                    early_stopped = True
                    break

                # Gradient update
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                # AMP backward pass: scales loss to prevent underflow in float16
                self.scaler.scale(loss).backward()

                # Gradient clipping: prevent huge gradient steps from destabilising training
                # ELI5: "If the learning signal is shouting too loudly, turn it down"
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)

                # Apply gradients (using scaler for AMP)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if early_stopped:
                break

        # Average all logged stats across mini-batches
        avg_stats: Dict[str, float] = {}
        for k in all_stats[0]:
            avg_stats[k] = float(np.mean([s[k] for s in all_stats]))

        avg_stats["train/learning_rate"]  = lr
        avg_stats["train/clip_epsilon"]   = clip_epsilon
        avg_stats["train/entropy_coef"]   = entropy_coef
        avg_stats["train/early_stopped"]  = float(early_stopped)

        return avg_stats

    # ──────────────────────────────────────────────────────────────────────────
    # §10.4  MAIN TRAINING LOOP
    # ──────────────────────────────────────────────────────────────────────────

    def learn(self) -> "PPOAgent":
        """
        Main PPO training loop.

        ELI5 of the full loop:
          1. Reset all game copies.
          2. Play the game for rollout_steps steps (collect experience).
          3. Compute how good each action was (GAE advantages + returns).
          4. Update Actor and Critic using PPO loss for n_epochs.
          5. Log metrics, save checkpoint if needed.
          6. Repeat from step 2 until total_timesteps reached.

        Returns:
            self (for method chaining)
        """
        logger.info(
            f"Starting PPO training: {self.cfg.env_id} | "
            f"{self.cfg.total_timesteps:,} steps | device={self.device}"
        )

        # ── Initial reset ───────────────────────────────────────────────────
        obs = self.envs.reset()     # (num_envs, *obs_shape)
        start_time = time.time()

        for update in range(self.update_count, self.total_updates):
            self.update_count = update

            # ── 1. Collect rollout ──────────────────────────────────────────
            obs, rollout_time = self.collect_rollout(obs)

            # ── 2. Update policy ────────────────────────────────────────────
            update_t0 = time.time()
            self.policy.train()     # Switch to training mode (enables dropout, etc.)
            stats = self.update_policy()
            self.policy.eval()      # Switch back to eval mode for rollout
            update_time = time.time() - update_t0

            # ── 3. Compute throughput ───────────────────────────────────────
            elapsed     = time.time() - start_time
            sps         = self.global_step / max(elapsed, 1e-6)   # Steps per second
            progress    = self.global_step / self.cfg.total_timesteps * 100

            # ── 4. Log metrics ──────────────────────────────────────────────
            if update % self.cfg.log_interval == 0:
                stats.update({
                    "charts/global_step"  : self.global_step,
                    "charts/mean_reward"  : self.training_logger.mean_reward,
                    "charts/mean_ep_length": self.training_logger.mean_length,
                    "charts/steps_per_sec": sps,
                    "charts/rollout_time" : rollout_time,
                    "charts/update_time"  : update_time,
                    "charts/progress_pct" : progress,
                })
                self.training_logger.log_step(self.global_step, stats)

                logger.info(
                    f"[{progress:5.1f}%] step={self.global_step:,} | "
                    f"reward={self.training_logger.mean_reward:7.2f} | "
                    f"actor_loss={stats['train/actor_loss']:+.4f} | "
                    f"critic_loss={stats['train/critic_loss']:.4f} | "
                    f"entropy={stats['train/entropy']:.4f} | "
                    f"kl={stats['train/approx_kl']:.4f} | "
                    f"sps={sps:.0f}"
                )

            # ── 5. Save checkpoint ──────────────────────────────────────────
            if update % self.cfg.save_interval == 0:
                ckpt_path = (
                    Path(self.cfg.log_dir)
                    / self.cfg.experiment_name
                    / f"checkpoint_step{self.global_step}.pt"
                )
                self.save_checkpoint(str(ckpt_path))

        # ── Final save ──────────────────────────────────────────────────────
        final_path = (
            Path(self.cfg.log_dir)
            / self.cfg.experiment_name
            / "checkpoint_final.pt"
        )
        self.save_checkpoint(str(final_path))
        self.training_logger.close()
        self.envs.close()

        logger.info(
            f"Training complete! Final mean reward: {self.training_logger.mean_reward:.2f} | "
            f"Total time: {(time.time() - start_time) / 60:.1f} min"
        )
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # §10.5  EVALUATION
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        n_episodes: int = 10,
        max_steps: int = 1000,
        env_factory: Optional[Callable] = None,
        render: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate the current policy for n_episodes.

        ELI5: After training, test how well the agent performs.
        No randomness from PPO — just the best action (deterministic / greedy).

        Args:
            n_episodes : Number of test episodes
            max_steps  : Max steps per episode
            env_factory: Optional custom env factory for evaluation
            render     : Whether to render the environment visually

        Returns:
            Dictionary with mean/std of rewards and episode lengths
        """
        self.policy.eval()

        if env_factory is not None:
            eval_env = env_factory(999)
        elif _GYM_AVAILABLE:
            kwargs = {"render_mode": "human"} if render else {}
            eval_env = gym.make(self.cfg.env_id, **kwargs)
        else:
            raise RuntimeError("No environment available for evaluation.")

        ep_rewards, ep_lengths = [], []

        for ep in range(n_episodes):
            result = eval_env.reset()
            obs    = result[0] if isinstance(result, tuple) else result
            ep_r, ep_l = 0.0, 0
            done = False

            while not done and ep_l < max_steps:
                obs_norm   = self._normalise_obs(np.array([obs]), update_stats=False)[0]
                obs_tensor = torch.FloatTensor(obs_norm).unsqueeze(0).to(self.device)

                # Deterministic action: sample from policy (which is near-deterministic after training)
                # For truly greedy: use argmax for discrete, mean for continuous
                action, _, _, _ = self.policy.get_action_and_value(obs_tensor)
                action_val = action.cpu().numpy()[0]

                if self.cfg.action_space_type == "discrete":
                    action_val = int(action_val)

                step_result = eval_env.step(action_val)
                if len(step_result) == 5:
                    obs, r, term, trunc, _ = step_result
                    done = term or trunc
                else:
                    obs, r, done, _ = step_result

                ep_r += r
                ep_l += 1

            ep_rewards.append(ep_r)
            ep_lengths.append(ep_l)
            logger.info(f"  Eval episode {ep+1}/{n_episodes}: reward={ep_r:.1f}, length={ep_l}")

        eval_env.close()

        results = {
            "eval/mean_reward"  : float(np.mean(ep_rewards)),
            "eval/std_reward"   : float(np.std(ep_rewards)),
            "eval/min_reward"   : float(np.min(ep_rewards)),
            "eval/max_reward"   : float(np.max(ep_rewards)),
            "eval/mean_length"  : float(np.mean(ep_lengths)),
        }
        logger.info(
            f"Evaluation ({n_episodes} episodes): "
            f"mean={results['eval/mean_reward']:.2f} ± {results['eval/std_reward']:.2f} | "
            f"range=[{results['eval/min_reward']:.1f}, {results['eval/max_reward']:.1f}]"
        )
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # §10.6  CHECKPOINT SAVE / LOAD
    # ──────────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """
        Save complete training state to disk.

        ELI5: Like saving a game mid-play. Stores EVERYTHING:
        - Network weights (actor + critic)
        - Optimizer state (so Adam's momentum is preserved)
        - Running stats (obs and reward normalisers)
        - Training counters (where we were in training)

        This allows FULL resumption of training later.

        Args:
            path: Full path to save the checkpoint file (.pt)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "version"           : "PPO.RL.v1",
            "global_step"       : self.global_step,
            "update_count"      : self.update_count,
            "policy_state_dict" : self.policy.state_dict(),
            "optimizer_state"   : self.optimizer.state_dict(),
            "scaler_state"      : self.scaler.state_dict(),
            "obs_rms"           : self.obs_rms.state_dict()    if self.obs_rms    else None,
            "reward_rms"        : self.reward_rms.state_dict() if self.reward_rms else None,
            "config"            : self.cfg.__dict__,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load a previously saved checkpoint to resume training.

        ELI5: Load a saved game. Restores the network weights, optimizer state,
        and training counters so training continues exactly where it left off.

        Args:
            path: Path to checkpoint file (.pt)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])

        self.global_step  = checkpoint["global_step"]
        self.update_count = checkpoint["update_count"]

        if self.obs_rms and checkpoint["obs_rms"]:
            self.obs_rms.load_state_dict(checkpoint["obs_rms"])
        if self.reward_rms and checkpoint["reward_rms"]:
            self.reward_rms.load_state_dict(checkpoint["reward_rms"])

        logger.info(
            f"Checkpoint loaded from {path} "
            f"(step={self.global_step}, updates={self.update_count})"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # §10.7  CONVENIENCE — SAVE / LOAD POLICY ONLY
    # ──────────────────────────────────────────────────────────────────────────

    def save_policy(self, path: str) -> None:
        """
        Save only the policy network weights (for deployment / inference).

        ELI5: If you just want to USE the trained agent (not continue training),
        you only need the network weights — not the optimizer state.
        This makes a much smaller file.
        """
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "obs_rms"          : self.obs_rms.state_dict() if self.obs_rms else None,
            "config"           : self.cfg.__dict__,
        }, path)
        logger.info(f"Policy saved → {path}")

    def load_policy(self, path: str) -> None:
        """Load only policy weights (for inference/evaluation after training)."""
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data["policy_state_dict"])
        if self.obs_rms and data.get("obs_rms"):
            self.obs_rms.load_state_dict(data["obs_rms"])
        logger.info(f"Policy weights loaded from {path}")

    # ──────────────────────────────────────────────────────────────────────────
    # §10.8  PREDICT (INFERENCE) — Deploy the trained policy
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> np.ndarray:
        """
        Predict an action for a given observation (inference mode).

        ELI5: After training is done, use this to let the agent "play" in a new situation.
        deterministic=True → always pick the most likely action (greedy, no randomness).
        deterministic=False → sample from the distribution (stochastic, like during training).

        Args:
            obs          : Single or batched observation
            deterministic: If True, argmax for discrete / mean for continuous

        Returns:
            action: Predicted action as numpy array
        """
        self.policy.eval()

        if isinstance(obs, np.ndarray):
            obs = self._normalise_obs(obs, update_stats=False)
            obs = torch.FloatTensor(obs)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)      # Add batch dimension

        obs = obs.to(self.device)

        if deterministic:
            if self.cfg.action_space_type == "discrete":
                # Deterministic: pick action with highest logit
                actor_feats, _ = self.policy._get_features(obs)
                logits = self.policy.actor_head(actor_feats)
                action = logits.argmax(dim=-1)
            else:
                # Deterministic: use the mean of the Gaussian (no sampling)
                actor_feats, _ = self.policy._get_features(obs)
                action = self.policy.actor_mean(actor_feats)
                if self.cfg.squash_actions:
                    action = torch.tanh(action)
        else:
            # Stochastic: sample from the distribution
            action, _, _, _ = self.policy.get_action_and_value(obs)

        return action.cpu().numpy()

    def __repr__(self) -> str:
        return (
            f"PPOAgent(env={self.cfg.env_id}, "
            f"device={self.device}, "
            f"action_type={self.cfg.action_space_type}, "
            f"total_timesteps={self.cfg.total_timesteps:,})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# §11  CONVENIENCE HELPERS — QUICK-START FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def train_discrete(
    env_id: str = "CartPole-v1",
    total_timesteps: int = 500_000,
    **kwargs,
) -> PPOAgent:
    """
    Quick-start PPO training for DISCRETE action space environments.

    ELI5: One function call to train a PPO agent on any discrete gym environment.
    Good for: CartPole, LunarLander, Atari (with appropriate backbone).

    Args:
        env_id          : Gymnasium environment ID
        total_timesteps : Training budget
        **kwargs        : Any PPOConfig fields to override

    Returns:
        Trained PPOAgent
    """
    cfg = PPOConfig(
        env_id             = env_id,
        action_space_type  = "discrete",
        total_timesteps    = total_timesteps,
        **kwargs,
    )
    agent = PPOAgent(cfg)
    return agent.learn()


def train_continuous(
    env_id: str = "HalfCheetah-v4",
    total_timesteps: int = 1_000_000,
    **kwargs,
) -> PPOAgent:
    """
    Quick-start PPO training for CONTINUOUS action space environments.

    ELI5: One function call to train on any robot/physics sim environment.
    Good for: MuJoCo (HalfCheetah, Hopper, Ant), robotics, real-world control.

    Args:
        env_id          : Gymnasium environment ID
        total_timesteps : Training budget
        **kwargs        : Any PPOConfig fields to override

    Returns:
        Trained PPOAgent
    """
    cfg = PPOConfig(
        env_id             = env_id,
        action_space_type  = "continuous",
        total_timesteps    = total_timesteps,
        hidden_sizes       = (256, 256),
        lr_actor           = 3e-4,
        lr_critic          = 1e-3,
        rollout_steps      = 2048,
        n_epochs           = 10,
        minibatch_size     = 64,
        **kwargs,
    )
    agent = PPOAgent(cfg)
    return agent.learn()


def train_with_transformer_backbone(
    env_id: str = "CartPole-v1",
    total_timesteps: int = 500_000,
    **kwargs,
) -> PPOAgent:
    """
    Quick-start PPO with Transformer policy backbone.

    ELI5: Use a Transformer (the same architecture as GPT/BERT) as the policy
    backbone. Good when observations have sequential or relational structure.

    Args:
        env_id          : Gymnasium environment ID
        total_timesteps : Training budget
        **kwargs        : Any PPOConfig fields to override

    Returns:
        Trained PPOAgent
    """
    cfg = PPOConfig(
        env_id             = env_id,
        backbone_cls       = TransformerPolicyBackbone,
        total_timesteps    = total_timesteps,
        **kwargs,
    )
    agent = PPOAgent(cfg)
    return agent.learn()


# ══════════════════════════════════════════════════════════════════════════════
# §12  ENTRY POINT — Demo / Smoke Test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Demo: Train PPO on CartPole-v1 (should solve in ~200k steps).

    ELI5: CartPole is a "balance the stick on the cart" game.
    The agent learns to push the cart left/right to keep the pole upright.
    A perfect agent can balance it indefinitely (score = 500 = max).
    """

    # ── Example 1: Discrete (CartPole) ──────────────────────────────────────
    print("\n" + "="*70)
    print("  PPO.RL.py — Production PPO Demo")
    print("="*70)
    print("\nExample 1: Discrete action space — CartPole-v1")

    cfg_discrete = PPOConfig(
        env_id            = "CartPole-v1",
        action_space_type = "discrete",
        total_timesteps   = 200_000,
        num_envs          = 4,
        rollout_steps     = 512,
        n_epochs          = 10,
        minibatch_size    = 64,
        hidden_sizes      = (64, 64),
        lr_actor          = 2.5e-4,
        lr_schedule       = "linear",
        gamma             = 0.99,
        gae_lambda        = 0.95,
        clip_epsilon      = 0.2,
        entropy_coef      = 0.01,
        normalise_obs     = True,
        normalise_rewards = False,  # CartPole rewards are already {0, 1}
        log_dir           = "./ppo_logs",
        experiment_name   = "cartpole_demo",
        log_interval      = 5,
        save_interval     = 20,
        seed              = 42,
    )

    agent_discrete = PPOAgent(cfg_discrete)
    agent_discrete.learn()

    print("\nEvaluating trained CartPole agent...")
    results = agent_discrete.evaluate(n_episodes=10, max_steps=500)
    print(f"  Mean reward: {results['eval/mean_reward']:.1f} / 500.0")

    # ── Example 2: Continuous (simple env if available) ──────────────────────
    print("\n" + "-"*70)
    print("Example 2: Continuous action space — Pendulum-v1")

    cfg_cont = PPOConfig(
        env_id            = "Pendulum-v1",
        action_space_type = "continuous",
        total_timesteps   = 100_000,
        num_envs          = 4,
        rollout_steps     = 512,
        n_epochs          = 10,
        minibatch_size    = 64,
        hidden_sizes      = (64, 64),
        lr_actor          = 3e-4,
        lr_critic         = 1e-3,
        gamma             = 0.99,
        gae_lambda        = 0.95,
        clip_epsilon      = 0.2,
        entropy_coef      = 0.0,
        normalise_obs     = True,
        normalise_rewards = True,
        continuous_dist   = "gaussian",
        log_dir           = "./ppo_logs",
        experiment_name   = "pendulum_demo",
        log_interval      = 5,
        save_interval     = 20,
        seed              = 42,
    )

    try:
        agent_cont = PPOAgent(cfg_cont)
        agent_cont.learn()
        results_cont = agent_cont.evaluate(n_episodes=5)
        print(f"  Pendulum mean reward: {results_cont['eval/mean_reward']:.1f}")
    except Exception as e:
        print(f"  Skipped (env may not be installed): {e}")

    # ── Example 3: Transformer Backbone ─────────────────────────────────────
    print("\n" + "-"*70)
    print("Example 3: Transformer backbone — CartPole-v1")

    cfg_tf = PPOConfig(
        env_id            = "CartPole-v1",
        action_space_type = "discrete",
        total_timesteps   = 100_000,
        num_envs          = 4,
        rollout_steps     = 512,
        backbone_cls      = TransformerPolicyBackbone,
        log_dir           = "./ppo_logs",
        experiment_name   = "cartpole_transformer",
        log_interval      = 5,
        save_interval     = 20,
        seed              = 0,
    )

    try:
        agent_tf = PPOAgent(cfg_tf)
        agent_tf.learn()
        results_tf = agent_tf.evaluate(n_episodes=5)
        print(f"  Transformer backbone mean reward: {results_tf['eval/mean_reward']:.1f}")
    except Exception as e:
        print(f"  Skipped: {e}")

    print("\n" + "="*70)
    print("  All demos complete. Check ./ppo_logs for TensorBoard & CSV logs.")
    print("  Run: tensorboard --logdir ./ppo_logs")
    print("="*70)