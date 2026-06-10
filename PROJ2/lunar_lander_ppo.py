import argparse
import os
from concurrent.futures import ThreadPoolExecutor
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

TIMESTEP, EPOCHS, MINIBATCHES, MAX_EPISODES, EP_MAX_STEPS = 1000, 10, 8, 2000, 500
LR_ACTOR, LR_CRITIC, LR_DECAY = 0.0003, 0.001, True
PPO_CLIP, OBS_CLIP, GRAD_CLIP, VF_COEF, ENT_COEF, DISC_COEF, GAE_COEF = 0.20, 10.0, 0.50, 0.50, 0.01, 0.99, 0.95
EP_REWARD_PENALTY, EP_TIMEOUT_PENALTY = 0.0, 0.0
PARALLEL_ENVS, ROLLOUT_WORKERS = 4, 4
MODEL_PATH = "ppo_lunar_lander.pt"
SEED = 43
RENDER_INTERVAL = 0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
    )


def print_hyperparameters(args=None, agent=None):
    print("\nHyperparameters:", flush=True)

    if args:
        for key, value in vars(args).items():
            print(f"  {key}: {value}", flush=True)

    if agent:
        keys = [
            "update_timestep",
            "epochs",
            "minibatches",
            "ppo_clip",
            "obs_clip",
            "grad_clip",
            "disc_coef",
            "gae_coef",
        ]
        for key in keys:
            print(f"  {key}: {getattr(agent, key)}", flush=True)


def save_plot(scores, log_dir, seed):
    os.makedirs(log_dir, exist_ok=True)
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"PPO LunarLander Training Scores Seed {seed}")
    plt.savefig(os.path.join(log_dir, "training_scores.png"), dpi=200)
    plt.close()


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = make_mlp(state_dim, action_dim)

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = make_mlp(state_dim, 1)

    def forward(self, x):
        return self.net(x)


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
        timestep=TIMESTEP,
        epochs=EPOCHS,
        ppo_clip=PPO_CLIP,
        obs_clip=OBS_CLIP,
        grad_clip=GRAD_CLIP,
        minibatches=MINIBATCHES,
        disc_coef=DISC_COEF,
        gae_coef=GAE_COEF,
        lr_actor=LR_ACTOR,
        lr_critic=LR_CRITIC,
    ):
        self.update_timestep = timestep
        self.epochs = epochs
        self.ppo_clip = ppo_clip
        self.obs_clip = obs_clip
        self.grad_clip = grad_clip
        self.minibatches = max(1, minibatches)
        self.disc_coef = disc_coef
        self.gae_coef = gae_coef
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device

        self.obs_mean = np.zeros(state_size, dtype=np.float32)
        self.obs_var = np.ones(state_size, dtype=np.float32)
        self.obs_count = 1e-4

        self.policy = ActorCriticAgent(state_size, action_size).to(device)
        self.policy_old = ActorCriticAgent(state_size, action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": lr_actor},
                {"params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.next_state_values = []
        self.rewards = []
        self.dones = []
        self.env_ids = []

    def update_obs_stats(self, obs):
        obs = np.asarray(obs, dtype=np.float32)

        if obs.ndim == 1:
            obs = obs[None, :]

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

    def normalize_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        norm_obs = (obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        return np.clip(norm_obs, -self.obs_clip, self.obs_clip).astype(np.float32)

    def obs_tensor(self, obs, update_stats=False):
        if update_stats:
            self.update_obs_stats(obs)

        norm_obs = self.normalize_obs(obs)
        return torch.as_tensor(norm_obs, dtype=torch.float32, device=self.device)

    def value_of_state(self, state):
        with torch.no_grad():
            return self.policy_old.critic(self.obs_tensor(state)).reshape(-1)[0].detach()

    def remember_action(self, state_tensor, action, logprob, value, env_id):
        self.states.append(state_tensor)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.state_values.append(value)
        self.env_ids.append(env_id)

    def remember_transition(self, reward, next_value, done):
        self.rewards.append(reward)
        self.next_state_values.append(next_value)
        self.dones.append(done)

    def act(self, state, env_id=0):
        state_tensor = self.obs_tensor(state, update_stats=True)

        with torch.no_grad():
            action, logprob, value = self.policy_old.act(state_tensor)

        self.remember_action(state_tensor, action, logprob, value, env_id)
        return action.item()

    def act_batch(self, states):
        states = np.asarray(states, dtype=np.float32)
        state_tensor = self.obs_tensor(states, update_stats=True)

        with torch.no_grad():
            actions, logprobs, values = self.policy_old.act(state_tensor)

        for i in range(len(states)):
            self.remember_action(state_tensor[i], actions[i], logprobs[i], values[i], i)

        return actions.detach().cpu().numpy().astype(int)

    def decay_learning_rate(self, progress_remaining, writer=None, step=None):
        progress_remaining = max(0.0, min(1.0, progress_remaining))
        actor_lr = self.lr_actor * progress_remaining
        critic_lr = self.lr_critic * progress_remaining

        self.optimizer.param_groups[0]["lr"] = actor_lr
        self.optimizer.param_groups[1]["lr"] = critic_lr

        if writer and step is not None:
            writer.add_scalar("LR/Actor", actor_lr, step)
            writer.add_scalar("LR/Critic", critic_lr, step)

    def log_episode(self, writer, total, avg, episode_length, time_step, episode):
        if not writer:
            return

        writer.add_scalar("Train/Episode_Return", total, episode)
        writer.add_scalar("Train/Average_Return_100", avg, episode)
        writer.add_scalar("Train/Episode_Length", episode_length, episode)
        writer.add_scalar("Train/Total_Timesteps", time_step, episode)
        writer.add_scalar("Obs/Mean_Abs", float(np.mean(np.abs(self.obs_mean))), episode)
        writer.add_scalar("Obs/Var_Mean", float(np.mean(self.obs_var)), episode)

    def maybe_render(self, writer, episode, seed, render_interval):
        if not render_interval or render_interval <= 0 or episode % render_interval != 0:
            return

        render_score = self.render_eval_episode(seed + episode if seed is not None else SEED)
        print(f"Rendered eval episode at train episode {episode}: reward = {render_score:.2f}", flush=True)

        if writer:
            writer.add_scalar("Eval/Rendered_Return", render_score, episode)

    def update(self, returns, advantages, writer=None, time_step=None):
        states = torch.stack(self.states).detach().to(self.device)
        actions = torch.stack(self.actions).detach().to(self.device).reshape(-1)
        old_logprobs = torch.stack(self.logprobs).detach().to(self.device).reshape(-1)
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
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                logprobs, values, entropy = self.policy.evaluate(mb_states, mb_actions)
                logprobs = logprobs.reshape(-1)
                values = values.reshape(-1)
                entropy = entropy.reshape(-1)

                ratios = torch.exp(logprobs - mb_old_logprobs)
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

    def learn(self, writer=None, time_step=None):
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=self.device)
        values = torch.stack(self.state_values).detach().reshape(-1).to(self.device)
        next_values = torch.stack(self.next_state_values).detach().reshape(-1).to(self.device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=self.device)

        advantages = torch.zeros_like(rewards)
        gae_by_env = {}

        for idx in range(len(rewards) - 1, -1, -1):
            env_id = self.env_ids[idx]

            if self.dones[idx]:
                gae_by_env[env_id] = 0.0

            mask = 1.0 - dones[idx]
            delta = rewards[idx] + self.disc_coef * next_values[idx] * mask - values[idx]
            gae_by_env[env_id] = delta + self.disc_coef * self.gae_coef * mask * gae_by_env.get(env_id, 0.0)
            advantages[idx] = gae_by_env[env_id]

        returns = advantages + values
        self.update(returns, advantages, writer, time_step)
        self.clear()

    def handle_step(self, next_state, reward, terminated, truncated, length, ep_max_steps, ep_reward_penalty, ep_timeout_penalty):
        length += 1
        reward += ep_reward_penalty
        done = terminated or truncated

        if length >= ep_max_steps and not done:
            reward += ep_timeout_penalty
            done = True

        next_value = torch.tensor(0.0, device=self.device) if done else self.value_of_state(next_state)
        self.remember_transition(reward, next_value, done)
        return reward, done, length

    def train_agent(
        self,
        env,
        episodes=MAX_EPISODES,
        ep_max_steps=EP_MAX_STEPS,
        ep_reward_penalty=EP_REWARD_PENALTY,
        ep_timeout_penalty=EP_TIMEOUT_PENALTY,
        writer=None,
        seed=SEED,
        lr_decay=LR_DECAY,
        render_interval=RENDER_INTERVAL,
    ):
        time_step = 0
        scores = []

        print("Collecting rollouts with parallel_envs=1, rollout_workers=1", flush=True)

        for ep in range(1, episodes + 1):
            if lr_decay:
                self.decay_learning_rate(1.0 - (ep - 1) / episodes, writer, ep)

            state, _ = env.reset(seed=seed + ep - 1 if seed is not None else None)
            total = 0.0
            episode_length = ep_max_steps

            for _ in range(ep_max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                reward, done, episode_length = self.handle_step(
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    episode_length if total else 0,
                    ep_max_steps,
                    ep_reward_penalty,
                    ep_timeout_penalty,
                )

                time_step += 1
                total += reward

                if len(self.rewards) >= self.update_timestep:
                    self.learn(writer, time_step)

                state = next_state

                if done:
                    break

            scores.append(total)
            avg = np.mean(scores[-100:])

            self.log_episode(writer, total, avg, episode_length, time_step, ep)
            print(f"\rEpisode {ep}\tAverage Score: {avg:.2f}", end="", flush=True)

            if ep % 100 == 0:
                print(f"\rEpisode {ep}\tAverage Score: {avg:.2f}", flush=True)

            self.maybe_render(writer, ep, seed, render_interval)

            if avg >= 200:
                print(f"\nEnvironment solved in {ep:d} episodes!\tAverage Score: {avg:.2f}", flush=True)
                break

        if self.rewards:
            self.learn(writer, time_step)

        return scores

    def train_agent_parallel(
        self,
        make_env,
        episodes=MAX_EPISODES,
        ep_max_steps=EP_MAX_STEPS,
        ep_reward_penalty=EP_REWARD_PENALTY,
        ep_timeout_penalty=EP_TIMEOUT_PENALTY,
        parallel_envs=PARALLEL_ENVS,
        rollout_workers=ROLLOUT_WORKERS,
        writer=None,
        seed=SEED,
        lr_decay=LR_DECAY,
        render_interval=RENDER_INTERVAL,
    ):
        envs = [make_env() for _ in range(parallel_envs)]
        executor = ThreadPoolExecutor(max_workers=max(1, rollout_workers)) if rollout_workers > 1 else None
        states = [env.reset(seed=seed + i if seed is not None else None)[0] for i, env in enumerate(envs)]
        totals = [0.0] * parallel_envs
        lengths = [0] * parallel_envs
        scores = []
        time_step = 0
        completed = 0

        print(f"Collecting rollouts with parallel_envs={parallel_envs}, rollout_workers={rollout_workers}", flush=True)

        try:
            while completed < episodes:
                actions = self.act_batch(states)

                if executor:
                    results = list(executor.map(lambda pair: pair[0].step(int(pair[1])), zip(envs, actions)))
                else:
                    results = [envs[i].step(int(actions[i])) for i in range(parallel_envs)]

                for i, (next_state, reward, terminated, truncated, _) in enumerate(results):
                    reward, done, lengths[i] = self.handle_step(
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        lengths[i],
                        ep_max_steps,
                        ep_reward_penalty,
                        ep_timeout_penalty,
                    )

                    totals[i] += reward
                    time_step += 1

                    if done:
                        completed += 1
                        scores.append(totals[i])
                        avg = np.mean(scores[-100:])

                        if lr_decay:
                            self.decay_learning_rate(1.0 - (completed - 1) / episodes, writer, completed)

                        self.log_episode(writer, totals[i], avg, lengths[i], time_step, completed)
                        print(f"\rEpisode {completed}\tAverage Score: {avg:.2f}", end="", flush=True)

                        if completed % 100 == 0:
                            print(f"\rEpisode {completed}\tAverage Score: {avg:.2f}", flush=True)

                        self.maybe_render(writer, completed, seed, render_interval)

                        totals[i] = 0.0
                        lengths[i] = 0
                        states[i] = envs[i].reset(seed=seed + completed + i if seed is not None else None)[0]

                        if avg >= 200 and completed >= 100:
                            print(f"\nEnvironment solved in {completed:d} episodes!\tAverage Score: {avg:.2f}", flush=True)
                            completed = episodes
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

    def render_eval_episode(self, seed=SEED):
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
        return total

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
        disc_coef=args.disc_coef,
        gae_coef=args.gae_coef,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
    )


def train_one_seed(args, seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = args.log_dir if args.num_seeds == 1 else os.path.join(args.log_dir, f"seed_{seed}")
    writer = SummaryWriter(log_dir)
    probe_env = gym.make("LunarLander-v2")

    probe_env.reset(seed=seed)
    probe_env.action_space.seed(seed)

    agent = make_agent(args, device, probe_env)
    probe_env.close()

    print_hyperparameters(args, agent)

    if args.parallel_envs > 1:
        scores = agent.train_agent_parallel(
            lambda: gym.make("LunarLander-v2"),
            args.episodes,
            args.ep_max_steps,
            args.ep_reward_penalty,
            args.ep_timeout_penalty,
            args.parallel_envs,
            args.rollout_workers,
            writer,
            seed,
            args.lr_decay,
            args.render_interval,
        )
    else:
        env = gym.make("LunarLander-v2")
        env.reset(seed=seed)
        env.action_space.seed(seed)

        scores = agent.train_agent(
            env,
            args.episodes,
            args.ep_max_steps,
            args.ep_reward_penalty,
            args.ep_timeout_penalty,
            writer,
            seed,
            args.lr_decay,
            args.render_interval,
        )

        env.close()

    model_path = args.model if args.num_seeds == 1 else f"{os.path.splitext(args.model)[0]}_seed_{seed}{os.path.splitext(args.model)[1]}"
    agent.save(model_path)
    writer.close()
    save_plot(scores, log_dir, seed)
    return scores


def train(args):
    all_scores = []

    for seed in range(args.seed, args.seed + args.num_seeds):
        print(f"\nTraining seed {seed}", flush=True)
        all_scores.append(train_one_seed(args, seed))

    return all_scores


def render(episodes=5, model_path=MODEL_PATH, obs_clip=OBS_CLIP, grad_clip=GRAD_CLIP, seed=SEED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLander-v2", render_mode="human")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, device, obs_clip=obs_clip, grad_clip=grad_clip)

    agent.load(model_path)
    agent.policy.eval()

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        done = False
        total = 0.0

        while not done:
            state_tensor = agent.obs_tensor(state)

            with torch.no_grad():
                action = torch.argmax(agent.policy.actor(state_tensor)).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

        print(f"Episode {ep + 1}: reward = {total:.2f}")

    env.close()


def play():
    from gymnasium.utils.play import play

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    keys_to_action = {
        (): 0,
        ("s",): 0,
        ("a",): 1,
        ("w",): 2,
        ("d",): 3,
        ("w", "a"): 2,
        ("w", "d"): 2,
    }

    play(env, keys_to_action=keys_to_action, noop=2, fps=60)
    env.close()


def export_pngs(log_dir="runs/ppo_lunar_lander", out_dir="tensorboard_pngs"):
    from pathlib import Path
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    Path(out_dir).mkdir(exist_ok=True)
    event_accumulator = EventAccumulator(log_dir)
    event_accumulator.Reload()

    for tag in event_accumulator.Tags().get("scalars", []):
        events = event_accumulator.Scalars(tag)
        x_values = [event.step for event in events]
        y_values = [event.value for event in events]

        plt.figure()
        plt.plot(x_values, y_values)
        plt.xlabel("Step")
        plt.ylabel(tag)
        plt.title(tag)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, tag.replace("/", "_") + ".png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "render", "play", "export"], nargs="?", default="train")
    parser.add_argument("--timestep", type=int, default=TIMESTEP)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--minibatches", type=int, default=MINIBATCHES)
    parser.add_argument("--ppo-clip", type=float, default=PPO_CLIP)
    parser.add_argument("--obs-clip", type=float, default=OBS_CLIP)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--disc-coef", type=float, default=DISC_COEF)
    parser.add_argument("--gae-coef", type=float, default=GAE_COEF)
    parser.add_argument("--lr-actor", type=float, default=LR_ACTOR)
    parser.add_argument("--lr-critic", type=float, default=LR_CRITIC)
    parser.add_argument("--episodes", type=int, default=MAX_EPISODES)
    parser.add_argument("--ep-max-steps", type=int, default=EP_MAX_STEPS)
    parser.add_argument("--ep-reward-penalty", type=float, default=EP_REWARD_PENALTY)
    parser.add_argument("--ep-timeout-penalty", type=float, default=EP_TIMEOUT_PENALTY)
    parser.add_argument("--parallel-envs", type=int, default=PARALLEL_ENVS)
    parser.add_argument("--rollout-workers", type=int, default=ROLLOUT_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--log-dir", default="runs/ppo_lunar_lander")
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--lr-decay", action=argparse.BooleanOptionalAction, default=LR_DECAY)
    parser.add_argument("--render-interval", type=int, default=RENDER_INTERVAL)

    args = parser.parse_args()

    {
        "train": lambda: train(args),
        "render": lambda: render(args.episodes, args.model, args.obs_clip, args.grad_clip, args.seed),
        "play": play,
        "export": lambda: export_pngs(args.log_dir),
    }[args.mode]()