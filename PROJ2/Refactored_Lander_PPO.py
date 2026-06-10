import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import gymnasium as gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from lunar_lander_ppo import train_one_seed

# "step" : one time step, one environment action/update
# "episode" : one complete environment run, env.reset() -> {game over}
# "rollout [batch]" : collection of steps for training
# "mini-batch" : rollout chunk for optimizer update
# "epoch" : training pass through rollout [batch]
# "training cycle" : full rollout collect/update loop
# "rollout length" : number of steps for a training cycle
# "episode length" : number of steps in one environment run (episode)

### GLOBAL VARIABLES
## NUM_ENVIRONMENTS_PER_RUN - [number of] environments for each python execution
## MAX_EPISODES_PER_RUN - max [number of] episodes for each python execution
## MAX_STEPS_PER_EPISODE - "episode horizon" : max [number of] steps before one episode is  force-ended
## NUM_STEPS_PER_ENVIRONMENT_PER_ROLLOUT
## MAX_STEPS_PER_TRAINING - "rollout batch size" : max [number of] environment steps collected before each training cycle
## NUM_TRAININGS_PER_ROLLOUT - max [number of] collect/update iterations for each rollout
## ROLLOUTS_PER_RUN
## NUM_WORKERS_PER_RUN - [number of] rollout workers for each python execution
## LEARNING_RATE_ACTOR -
## LEARNING_RATE_CRITIC -
## PPO_CLIP - proximal policy objective clipping, limiting new and old policy probability ratio difference to a fixed range
## OBS_CLIP - observation clipping, limiting normalized state values to a fixed range
## GRAD_CLIP - gradient clipping, limiting nn gradients (during backpropogation) to a fixed range
## VF_COEF - value function coefficient, weighting critic/value loss
## ENT_COEF - entropy coefficient, weighting entropy bonus
## GAE_COEF - generalized advantage estimation coefficient, smoothing advantages by averaaging TD errors
## REWARDS_DECAY - discount rate for prior rewards, decreasing with time (steps)
## LEARNING_RATE_DECAY - discount rate for learning rates, decreasing with time (steps)
## SEEDS -
## RENDER_INTERVAL - how often we render one episode during python execution
NUM_ENVIRONMENTS_PER_RUN = 4
MAX_EPISODES_PER_RUN = 2000
MAX_STEPS_PER_EPISODE = 500
#NUM_STEPS_PER_ENVIRONMENT_PER_ROLLOUT = # 2048 batch size // envs
#MAX_STEPS_PER_TRAINING = NUM_ENVIRONMENTS_PER_RUN * NUM_STEPS_PER_ENVIRONMENT_PER_ROLLOUT
NUM_TRAININGS_PER_ROLLOUT = 4
#ROLLOUTS_PER_RUN = MAX_EPISODES_PER_RUN * MAX_STEPS_PER_EPISODE / MAX_STEPS_PER_TRAINING
NUM_WORKERS_PER_RUN = 4
LEARNING_RATE_ACTOR = 0.0003
LEARNING_RATE_CRITIC = 0.001
PPO_CLIP = 0.20
OBS_CLIP = 10.0
GRAD_CLIP = 0.50
VF_COEF = 0.50
ENT_COEF = 0.01
GAE_COEF = 0.95
REWARDS_DECAY = 0.99
LEARNING_RATE_DECAY = 1
SEEDS = [0,1,2,3,4]
LOG_DIR = "runs/ppo_lunar_lander"
MODEL_PATH = "ppo_lunar_lander.pt"
RENDER_INTERVAL = 100

# LOCAL VARIABLES
# num_steps - running total [number] of steps per episode
# num_rollouts - running total [number] of rollouts
# num_training_cycles - running total [number] of passes per rollout
# num_updates - running total [number] of PPO update rounds

### SEEDS HELPER FUNCTION
def parse_seeds(txt):
    return [int(seed.strip()) for seed in txt.split(",") if seed.strip()]

### MULTI LAYER PERCEPTRON
def make_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim): super().__init__(); self.net = make_mlp(state_dim, action_dim)
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim): super().__init__(); self.net = make_mlp(state_dim, 1)
    def forward(self, x): return self.net(x)

class ActorCriticAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

    def act(self, state):
        dist = Categorical(logits=self.actor(state))
        action = dist.sample()
        return action, dist.log_prob(action), self.critic(state)

    def evaluate(self, states, actions):
        dist = Categorical(logits=self.actor(states))
        return dist.log_prob(actions), self.critic(states), dist.entropy()

class PPOAgent:
    def __init__(
        self,
        state_size,
        action_size,
        device,
        ppo_clip=PPO_CLIP,
        obs_clip=OBS_CLIP,
        grad_clip=GRAD_CLIP,
        gae_coef=GAE_COEF,
        rewards_decay=REWARDS_DECAY,
        learning_rate_actor=LEARNING_RATE_ACTOR,
        learning_rate_critic=LEARNING_RATE_CRITIC,
    ):
        self.device = device
        self.ppo_clip = ppo_clip
        self.obs_clip = obs_clip
        self.grad_clip = grad_clip
        self.gae_coef = gae_coef
        self.rewards_decay = rewards_decay
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.obs_mean = np.zeros(state_size, dtype=np.float32)
        self.obs_var = np.ones(state_size, dtype=np.float32)
        self.obs_count = 1e-4
        self.policy = ActorCriticAgent(state_size, action_size).to(device)
        self.policy_old = ActorCriticAgent(state_size, action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": learning_rate_actor},
                {"params": self.policy.critic.parameters(), "lr": learning_rate_critic},
            ]
        )
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.state_values = []
        self.next_state_values = []
        self.rewards = []
        self.dones = []
        self.env_ids = []

    ### Update observation mean, variance, and count
    def update_obs_stats(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1: obs = obs[None, :]
        batch_mean = obs.mean(axis=0)
        batch_var = obs.var(axis=0)
        batch_count = obs.shape[0]
        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        new_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.obs_count * batch_count / total_count
        self.obs_mean = new_mean.astype(np.float32)
        self.obs_var = (m_2 / total_count).astype(np.float32)
        self.obs_count = total_count

    ### Normalize observation and clip value
    def normalize_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        norm_obs = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        return np.clip(norm_obs, -self.obs_clip, self.obs_clip).astype(np.float32)

    ### Update observation mean, variance, and count; convert to tensor
    def obs_tensor(self, obs, update_stats=False):
        if update_stats: self.update_obs_stats(obs)
        norm_obs = self.normalize_obs(obs)
        return torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device)

    ### Append state, action, log_prob, value, and env_id to lists
    def remember_action(self, s, a, lp, v, e):
        self.states.append(s)
        self.actions.append(a)
        self.log_probs.append(lp)
        self.state_values.append(v)
        self.env_ids.append(e)

    ### Append reward, next_state, and done to lists
    def remember_transition(self, r, n, d):
        self.rewards.append(r)
        self.next_state_values.append(n)
        self.dones.append(d)

    ### Return old policy actor value from state (pass through actor nn weights)
    def act(self, state, env_id=0):
        state_tensor = self.obs_tensor(state, update_stats=True)
        with torch.no_grad():
            action, log_prob, value = self.policy_old.act(state_tensor)
        self.remember_action(state_tensor, action, log_prob, value, env_id)
        return action.item()

    ### Return old policy actor values from states (pass through actor nn weights)
    def act_batch(self, states):
        states = np.asarray(states, dtype=np.float32)
        state_tensor = self.obs_tensor(states, update_stats=True)
        with torch.no_grad():
            actions, log_probs, values = self.policy_old.act(state_tensor)
        for i in range(len(states)):
            self.remember_action(state_tensor[i], actions[i], log_probs[i], values[i], i)
        return actions.detach().cpu().numpy().astype(int)

    ### Return old policy critic value (pass through critic nn weights)
    def crit(self, state):
        with torch.no_grad():
            return self.policy_old.critic(self.obs_tensor(state)).reshape(-1)[0].detach()

    ### Decay learning rate based on steps remaining (current step # vs max step #) for episode
    def decay_learning_rate(self, percent_incomplete, writer=None, step=None):
        percent_incomplete = max(0.0, min(1.0, percent_incomplete))
        a = self.learning_rate_actor * percent_incomplete
        c = self.learning_rate_critic * percent_incomplete
        self.optimizer.param_groups[0]["lr"] = a
        self.optimizer.param_groups[1]["lr"] = c
        if writer and step is not None:
            writer.add_scalar("LR/Actor", a, step)
            writer.add_scalar("LR/Critic", c, step)
        
    ### Add episode data to SummaryWriter        
    def log_episode(self, writer, total, avg, episode_length, time_step, episode):
        print(f"\rEpisode {episode}\tAverage Score: {avg:.2f}", end="", flush=True)
        if episode % 100 == 0: print(f"\rEpisode {episode}\tAverage Score: {avg:.2f}", flush=True)
        if not writer: return
        writer.add_scalar("Train/Episode_Return", total, episode)
        writer.add_scalar("Train/Average_Return_100", avg, episode)
        writer.add_scalar("Train/Episode_Length", episode_length, episode)
        writer.add_scalar("Train/Total_Timesteps", time_step, episode)
        writer.add_scalar("Obs/Mean_Abs", float(np.mean(np.abs(self.obs_mean))), episode)
        writer.add_scalar("Obs/Var_Mean", float(np.mean(self.obs_var)), episode)
    
    ### 
    def update(self, returns, advantages, writer=None, time_step=None):
        states = torch.stack(self.states).detach().to(self.device)
        actions = torch.stack(self.actions).detach().to(self.device).reshape(-1)
        old_log_probs = torch.stack(self.log_probs).detach().to(self.device).reshape(-1)
        old_values = torch.stack(self.state_values).detach().to(self.device).reshape(-1)
        returns = returns.detach().to(self.device).reshape(-1)
        advantages = advantages.detach().to(self.device).reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
        assert returns.shape == old_values.shape, f"returns {returns.shape}, old_values {old_values.shape}"
        assert advantages.shape == old_values.shape, f"advantages {advantages.shape}, old_values {old_values.shape}"
        batch_size = states.shape[0]
        for _ in range(self.epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for mb_idx in torch.chunk(indices, self.minibatches):
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]
                log_probs, values, entropy = self.policy.evaluate(mb_states, mb_actions)
                log_probs = log_probs.reshape(-1)
                values = values.reshape(-1)
                entropy = entropy.reshape(-1)
                ratios = torch.exp(log_probs - mb_old_log_probs)
                clipped_ratios = torch.clamp(ratios, 1 - self.ppo_clip, 1 + self.ppo_clip)
                actor_loss = -torch.min(ratios * mb_advantages, clipped_ratios * mb_advantages)
                critic_loss = F.smooth_l1_loss(values, mb_returns)
                loss = actor_loss + VF_COEF * critic_loss - ENT_COEF * entropy
                self.optimizer.zero_grad()
                loss.mean().backward()
                grad_norm = None
                if self.grad_clip is not None and self.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
                self.optimizer.step()
                if writer and time_step:
                    writer.add_scalar("Loss/Total", loss.mean().item(), time_step)
                    writer.add_scalar("Loss/Actor", actor_loss.mean().item(), time_step)
                    writer.add_scalar("Loss/Critic", critic_loss.item(), time_step)
                    writer.add_scalar("Policy/Entropy", entropy.mean().item(), time_step)
                    if grad_norm is not None:
                        writer.add_scalar("Grad/Global_Norm_Pre_Clip", float(grad_norm), time_step)
        self.policy_old.load_state_dict(self.policy.state_dict())

    ###
    def learn(self, writer=None, time_step=None):
        rewards = torch.stack(self.rewards).detach().reshape(-1).to(self.device)
        values = torch.stack(self.state_values).detach().reshape(-1).to(self.device)
        next_values = torch.stack(self.next_state_values).detach().reshape(-1).to(self.device)
        dones = torch.stack(self.dones).detach().reshape(-1).to(self.device)
        advantages = torch.zeros_like(rewards)
        gae_by_env = {}
        for idx in range(len(rewards) - 1, -1, -1):
            env_id = self.env_ids[idx]
            if self.dones[idx]: gae_by_env[env_id] = 0.0
            mask = 1.0 - dones[idx]
            delta = rewards[idx] + self.rewards_decay * next_values[idx] * mask - values[idx]
            gae_by_env[env_id] = delta + self.rewards_decay * self.gae_coef * mask * gae_by_env.get(env_id, 0.0)
            advantages[idx] = gae_by_env[env_id]
        returns = advantages + values
        self.update(returns, advantages, writer, time_step)
        self.clear()

    ###
    def train_agent(
        self,
        make_env,
        writer=None,
        seed=None,
        args
    ):
        envs = [make_env() for _ in range(args.envs)]
        executor = ThreadPoolExecutor(max_workers=max(1, args.wks)) if args.wks > 1 else None
        states = [env.reset(seed=seed + i if seed is not None else None)[0] for i, env in enumerate(envs)]
        totals = [0.0] * args.env
        lengths = [0] * args.env
        scores = []
        time_step = 0
        completed = 0
        print(f"Collecting rollouts with parallel_envs={args.env}, rollout_workers={args.wks}", flush=True)
        try:
            while completed < args.eps:
                actions = self.act_batch(states)
                if executor:
                    results = list(executor.map(lambda pair: pair[0].step(int(pair[1])), zip(envs, actions)))
                else:
                    results = [envs[i].step(int(actions[i])) for i in range(args.env)]
                for i, (next_state, reward, terminated, truncated, _) in enumerate(results):
                    lengths[i]+=1
                    done = terminated or truncated
                    if lengths[i] >= args.stp and not done:
                        done=True
                    next_value = torch.tensor(0.0,device=self.device) if done else self.crit(next_state)
                    self.remember_transition(reward,next_value,done)
                    totals[i] += reward
                    time_step += 1
                    if done:
                        completed += 1
                        scores.append(totals[i])
                        avg = np.mean(scores[-100:])
                        self.decay_learning_rate(1.0 - (completed - 1) / args.eps, writer, completed)
                        self.log_episode(writer, totals[i], avg, lengths[i], time_step, completed)
                        self.render(completed, seed, args.int)
                        totals[i] = 0.0
                        lengths[i] = 0
                        states[i] = envs[i].reset(seed=seed + completed + i if seed is not None else None)[0]
                        if avg >= 200 and completed >= 100:
                            print(f"\nEnvironment solved in {completed:d} episodes!\tAverage Score: {avg:.2f}", flush=True)
                            completed = args.eps
                            break
                    else:
                        states[i] = next_state
                if len(self.rewards) >= self.update_timestep:
                    self.learn(writer, time_step)
        finally:
            if executor:
                executor.shutdown(wait=True)
            for env in envs:
                env.close()
        if self.rewards:
            self.learn(writer, time_step)
        return scores
    
    ### Render one episode in gymnasium
    def render(self, episode, seed, render_interval):
        if not render_interval or render_interval <= 0 or episode % render_interval != 0: return
        env = gym.make("LunarLander-v2", render_mode="human")
        state, _ = env.reset(seed=seed)
        done = False
        total = 0.0
        self.policy.eval()
        while not done:
            state_tensor = self.obs_tensor(state)
            with torch.no_grad():
                action = torch.argmax(self.policy.actor(state_tensor)).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward
        env.close()
        self.policy.train()
        print(f"Rendered eval episode at train episode {episode}: reward = {total:.2f}", flush=True)
    
    ### Save checkpoint from path    
    def save(self, path):
        torch.save(
            {
                "Actor_state_dict": self.policy.actor.state_dict(),
                "Critic_state_dict": self.policy.critic.state_dict(),
                "obs_mean": self.obs_mean,
                "obs_var": self.obs_var,
                "obs_count": self.obs_count,
                "obs_clip": self.obs_clip,
                "grad_clip": self.grad_clip,
            },
            path,
        )

    ### Load checkpoint from path
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.actor.load_state_dict(checkpoint["Actor_state_dict"])
        self.policy.critic.load_state_dict(checkpoint["Critic_state_dict"])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.obs_mean = np.asarray(checkpoint.get("obs_mean", self.obs_mean), dtype=np.float32)
        self.obs_var = np.asarray(checkpoint.get("obs_var", self.obs_var), dtype=np.float32)
        self.obs_count = checkpoint.get("obs_count", self.obs_count)
        self.obs_clip = checkpoint.get("obs_clip", self.obs_clip)
        self.grad_clip = checkpoint.get("grad_clip", self.grad_clip)
        
### Make an agent
def make_agent(args, device, env):
    return PPOAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        device,
        timestep=args.timestep,
        epochs=args.epochs,
        ppo_clip=args.ppo_clip,
        obs_clip=args.obs_clip,
        grad_clip=args.grad_clip,
        minibatches=args.minibatches,
        rewards_decay=args.rewards_decay,
        gae_coef=args.gae_coef,
        learning_rate_actor=args.learning_rate_actor,
        learning_rate_critic=args.learning_rate_critic,
    )

def train(args):
    all_scores=[]
    seeds=args.seeds
    print(f"Training seeds: {seeds}", flush=True)
    for seed in seeds:
        print(f"Training seed {seed}", flush=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_dir = args.log_dir if len(seeds) == 1 else os.path.join(args.log_dir, f"seed_{seed}")
        writer = SummaryWriter(log_dir)
        probe_env = gym.make("LunarLander-v2")
        probe_env.reset(seed=seed)
        probe_env.action_space.seed(seed)
        agent = make_agent(args, device, probe_env)
        probe_env.close()
        for key, value in vars(args).items(): print(f"  {key}: {value}", flush=True)
        if args.num_envs > 1:
            scores = agent.train_agent(lambda: gym.make("LunarLander-v2"), writer, seed, args)
        else:
            env = gym.make("LunarLander-v2")
            env.reset(seed=seed)
            env.action_space.seed(seed)
            scores = agent.train_agent(env, writer, seed, args)
            env.close()
        model_path = args.model if len(seeds) == 1 else f"{os.path.splitext(args.model)[0]}_seed_{seed}{os.path.splitext(args.model)[1]}"
        agent.save(model_path)
        writer.close()
        all_scores.append(scores)
    return all_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "render", "play", "export"], nargs="?", default="train")
    parser.add_argument("--env", type=int, default=NUM_ENVIRONMENTS_PER_RUN)
    parser.add_argument("--eps", type=int, default=MAX_EPISODES_PER_RUN)
    parser.add_argument("--stp", type=int, default=MAX_STEPS_PER_EPISODE)
    parser.add_argument("--cyc", type=int, default=NUM_TRAININGS_PER_ROLLOUT)
    parser.add_argument("--wks", type=int, default=NUM_WORKERS_PER_RUN)
    parser.add_argument("--lra", type=float, default=LEARNING_RATE_ACTOR)
    parser.add_argument("--lrc", type=float, default=LEARNING_RATE_CRITIC)
    parser.add_argument("--ppo", type=float, default=PPO_CLIP)
    parser.add_argument("--obc", type=float, default=OBS_CLIP)
    parser.add_argument("--grc", type=float, default=GRAD_CLIP)
    parser.add_argument("--vfc", type=float, default=VF_COEF)
    parser.add_argument("--enc", type=float, default=ENT_COEF)
    parser.add_argument("--gae", type=float, default=GAE_COEF)
    parser.add_argument("--rdy", type=float, default=REWARDS_DECAY)
    parser.add_argument("--ldy", type=float, default=LEARNING_RATE_DECAY)
    parser.add_argument("--sed", type=parse_seeds, default=SEEDS)
    parser.add_argument("--log", default=LOG_DIR)
    parser.add_argument("--pth", default=MODEL_PATH)
    parser.add_argument("--int", type=int, default=RENDER_INTERVAL)

    args = parser.parse_args()

    {
        "train": lambda: train(args),
        #"render": lambda: render(args.episodes, args.model, args.obs_clip, args.grad_clip, args.seed),
        #"play": play,
        #"export": lambda: export_pngs(args.log_dir),
    }[args.mode]()