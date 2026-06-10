import argparse, os
from concurrent.futures import ThreadPoolExecutor
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

TIMESTEP, EPOCHS, MINIBATCHES, EPISODES, EP_MAX_STEPS = 1000, 10, 8, 2000, 500
LR_ACTOR, LR_CRITIC, LR_DECAY = 0.0003, 0.001, True
PPO_CLIP, OBS_CLIP, GRAD_CLIP, VF_COEF, ENT_COEF, DISC_COEF = 0.20, 10.0, 0.50, 0.50, 0.01, 0.99
# TO COMPLETELY NEGATE: 1e9,1e9,0,1,0,1 #
EP_REWARD_PENALTY, EP_TIMEOUT_PENALTY = -0.1, -10.0
PARALLEL_ENVS, ROLLOUT_WORKERS,  = 4, 4
MODEL_PATH = "ppo_lunar_lander.pt"
SEED = 43
RENDER_INTERVAL = 0

def print_hyperparameters(args=None, agent=None):
    print("\nHyperparameters:", flush=True)
    if args:
        for k, v in vars(args).items():
            print(f"  {k}: {v}", flush=True)
    if agent:
        print(f"  update_timestep: {agent.update_timestep}", flush=True)
        print(f"  epochs: {agent.epochs}", flush=True)
        print(f"  minibatches: {agent.minibatches}", flush=True)
        print(f"  ppo_clip: {agent.ppo_clip}", flush=True)
        print(f"  obs_clip: {agent.obs_clip}", flush=True)
        print(f"  grad_clip: {agent.grad_clip}", flush=True)
        print(f"  disc_coef: {agent.disc_coef}", flush=True)

def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class Actor(nn.Module):
    def __init__(self, s, a):
        super().__init__(); self.net = nn.Sequential(nn.Linear(s, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, a))
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, s):
        super().__init__(); self.net = nn.Sequential(nn.Linear(s, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x)

class ActorCriticAgent(nn.Module):
    def __init__(self, s, a):
        super().__init__(); self.actor = Actor(s, a); self.critic = Critic(s)
    def act(self, s):
        d = Categorical(logits=self.actor(s)); a = d.sample(); return a, d.log_prob(a), self.critic(s)
    def evaluate(self, s, a):
        d = Categorical(logits=self.actor(s)); return d.log_prob(a), self.critic(s), d.entropy()

class PPOAgent:
    def __init__(
            self, state_size, action_size, device, timestep=TIMESTEP, epochs=EPOCHS, ppo_clip=PPO_CLIP,
            obs_clip=OBS_CLIP,
            grad_clip=GRAD_CLIP, minibatches=MINIBATCHES, disc_coef=DISC_COEF, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC
    ):
        self.update_timestep, self.epochs, self.ppo_clip, self.disc_coef = timestep, epochs, ppo_clip, disc_coef
        self.lr_actor,self.lr_critic, self.minibatches = lr_actor,lr_critic,max(1, minibatches)
        self.device, self.state_dim, self.action_dim, self.grad_clip = device, state_size, action_size, grad_clip
        self.actions, self.states, self.logprobs, self.rewards, self.state_values, self.dones, self.env_ids = [], [], [], [], [], [], []
        self.obs_mean, self.obs_var, self.obs_count, self.obs_clip = np.zeros(state_size, np.float32), np.ones(state_size, np.float32), 1e-4, obs_clip
        self.policy = ActorCriticAgent(state_size, action_size).to(device)
        self.policy_old = ActorCriticAgent(state_size, action_size).to(device); self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam([{"params": self.policy.actor.parameters(), "lr": lr_actor}, {"params": self.policy.critic.parameters(), "lr": lr_critic}])

    def update_obs_stats(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1: obs = obs[None, :]
        batch_mean, batch_var, batch_count = obs.mean(axis=0), obs.var(axis=0), obs.shape[0]
        delta = batch_mean - self.obs_mean; total_count = self.obs_count + batch_count
        new_mean = self.obs_mean + delta * batch_count / total_count
        m_a, m_b = self.obs_var * self.obs_count, batch_var * batch_count
        self.obs_mean = new_mean.astype(np.float32)
        self.obs_var = ((m_a + m_b + delta**2 * self.obs_count * batch_count / total_count) / total_count).astype(np.float32)
        self.obs_count = total_count

    def normalize_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return np.clip((obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8), -self.obs_clip, self.obs_clip).astype(np.float32)

    def clear(self):
        self.actions.clear(); self.states.clear(); self.logprobs.clear(); self.rewards.clear(); self.state_values.clear(); self.dones.clear(); self.env_ids.clear()

    def act(self, state, env_id=0):
        self.update_obs_stats(state); norm_state = self.normalize_obs(state)
        with torch.no_grad():
            s = torch.as_tensor(norm_state, device=self.device, dtype=torch.float32); a, lp, v = self.policy_old.act(s)
        self.states.append(s); self.actions.append(a); self.logprobs.append(lp); self.state_values.append(v); self.env_ids.append(env_id)
        return a.item()

    def act_batch(self, states):
        states = np.asarray(states, dtype=np.float32); self.update_obs_stats(states); norm_states = self.normalize_obs(states)
        with torch.no_grad():
            s = torch.as_tensor(norm_states, device=self.device, dtype=torch.float32); a, lp, v = self.policy_old.act(s)
        for i in range(len(states)):
            self.states.append(s[i]); self.actions.append(a[i]); self.logprobs.append(lp[i]); self.state_values.append(v[i]); self.env_ids.append(i)
        return a.detach().cpu().numpy().astype(int)

    def decay_learning_rate(self, progress_remaining, writer=None, step=None):
        progress_remaining = max(0.0, min(1.0, progress_remaining))
        actor_lr = self.lr_actor * progress_remaining
        critic_lr = self.lr_critic * progress_remaining
        self.optimizer.param_groups[0]["lr"] = actor_lr
        self.optimizer.param_groups[1]["lr"] = critic_lr
        if writer and step is not None:
            writer.add_scalar("LR/Actor", actor_lr, step)
            writer.add_scalar("LR/Critic", critic_lr, step)

    def update(self, returns, writer=None, time_step=None):
        s = torch.stack(self.states).detach().to(self.device)
        a = torch.stack(self.actions).detach().to(self.device).reshape(-1)
        old_lp = torch.stack(self.logprobs).detach().to(self.device).reshape(-1)
        old_v = torch.stack(self.state_values).detach().to(self.device).reshape(-1)
        returns = returns.detach().to(self.device).reshape(-1)
        assert returns.shape == old_v.shape, f"returns {returns.shape}, old_v {old_v.shape}"

        adv = returns - old_v
        adv = (adv - adv.mean()) / (adv.std() + 1e-7)

        batch_size = s.shape[0]
        minibatch_size = max(1, batch_size // self.minibatches)

        for _ in range(self.epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                mb_s, mb_a, mb_old_lp = s[mb_idx], a[mb_idx], old_lp[mb_idx]
                mb_returns, mb_adv = returns[mb_idx], adv[mb_idx]

                lp, v, ent = self.policy.evaluate(mb_s, mb_a)
                lp, v, ent = lp.reshape(-1), v.reshape(-1), ent.reshape(-1)

                r = torch.exp(lp - mb_old_lp)
                actor_loss = -torch.min(r * mb_adv, torch.clamp(r, 1 - self.ppo_clip, 1 + self.ppo_clip) * mb_adv)
                critic_loss = F.smooth_l1_loss(v, mb_returns)
                loss = actor_loss + VF_COEF * critic_loss - ENT_COEF * ent

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
                    writer.add_scalar("Policy/Entropy", ent.mean().item(), time_step)
                    if grad_norm is not None:
                        writer.add_scalar("Grad/Global_Norm_Pre_Clip", float(grad_norm), time_step)

        self.policy_old.load_state_dict(self.policy.state_dict())

    def learn(self, writer=None, time_step=None):
        returns, r_gamma = [0.0] * len(self.rewards), {}
        for idx in range(len(self.rewards) - 1, -1, -1):
            eid = self.env_ids[idx]
            if self.dones[idx]: r_gamma[eid] = 0.0
            r_gamma[eid] = self.rewards[idx] + self.disc_coef * r_gamma.get(eid, 0.0)
            returns[idx] = r_gamma[eid]
        self.update(torch.tensor(returns, dtype=torch.float32, device=self.device), writer, time_step); self.clear()

    def train_agent(
            self, env, episodes=EPISODES, ep_max_steps=EP_MAX_STEPS, ep_reward_penalty=EP_REWARD_PENALTY,
            ep_timeout_penalty=EP_TIMEOUT_PENALTY, writer=None, seed=SEED, lr_decay=LR_DECAY, render_interval=RENDER_INTERVAL
    ):
        time_step, scores = 0, []
        for ep in range(1, episodes + 1):
            if lr_decay: self.decay_learning_rate(1.0 - (ep - 1) / episodes, writer, ep)
            state, _ = env.reset(seed=seed + ep - 1 if seed is not None else None)
            total, episode_length = 0, ep_max_steps
            for t in range(1, ep_max_steps + 1):
                action = self.act(state); state, reward, terminated, truncated, _ = env.step(action)
                reward += ep_reward_penalty; done = terminated or truncated
                if t == ep_max_steps and not done: reward += ep_timeout_penalty; done = True
                self.rewards.append(reward); self.dones.append(done); time_step += 1; total += reward
                if len(self.rewards) >= self.update_timestep: self.learn(writer, time_step)
                if done: episode_length = t; break
            scores.append(total); avg = np.mean(scores[-100:])
            if writer:
                writer.add_scalar("Train/Episode_Return", total, ep)
                writer.add_scalar("Train/Average_Return_100", avg, ep)
                writer.add_scalar("Train/Episode_Length", episode_length, ep)
                writer.add_scalar("Train/Total_Timesteps", time_step, ep)
                writer.add_scalar("Obs/Mean_Abs", float(np.mean(np.abs(self.obs_mean))), ep)
                writer.add_scalar("Obs/Var_Mean", float(np.mean(self.obs_var)), ep)
            print(f"\rEpisode {ep}\tAverage Score: {avg:.2f}", end="", flush=True)
            if ep % 100 == 0:
                print(f"\rEpisode {ep}\tAverage Score: {avg:.2f}", flush=True)

            if render_interval and render_interval > 0 and ep % render_interval == 0:
                render_score = self.render_eval_episode(seed + ep if seed is not None else SEED)
                print(f"\nRendered eval episode at train episode {ep}: reward = {render_score:.2f}", flush=True)
                if writer:
                    writer.add_scalar("Eval/Rendered_Return", render_score, ep)
            if avg >= 200: print(f"\nEnvironment solved in {ep:d} episodes!\tAverage Score: {avg:.2f}", flush=True); break
        if self.rewards: self.learn(writer, time_step)
        return scores

    def train_agent_parallel(
                self, make_env, episodes=EPISODES, ep_max_steps=EP_MAX_STEPS, ep_reward_penalty=EP_REWARD_PENALTY,
                ep_timeout_penalty=EP_TIMEOUT_PENALTY,
                parallel_envs=PARALLEL_ENVS, rollout_workers=ROLLOUT_WORKERS, writer=None, seed=SEED, lr_decay=LR_DECAY, render_interval=RENDER_INTERVAL
        ):
        envs = [make_env() for _ in range(parallel_envs)]
        executor = ThreadPoolExecutor(max_workers=max(1, rollout_workers)) if rollout_workers > 1 else None
        states = [env.reset(seed=seed + i if seed is not None else None)[0] for i, env in enumerate(envs)]
        totals, lengths, scores, time_step, completed = [0.0] * parallel_envs, [0] * parallel_envs, [], 0, 0
        print(f"Collecting rollouts with parallel_envs={parallel_envs}, rollout_workers={rollout_workers}", flush=True)
        print(f"\n")
        try:
            while completed < episodes:
                actions = self.act_batch(states)
                results = list(executor.map(lambda pair: pair[0].step(int(pair[1])), zip(envs, actions))) if executor else [envs[i].step(int(actions[i])) for i in range(parallel_envs)]
                for i, (next_state, reward, terminated, truncated, _) in enumerate(results):
                    lengths[i] += 1; reward += ep_reward_penalty; done = terminated or truncated
                    if lengths[i] >= ep_max_steps and not done: reward += ep_timeout_penalty; done = True
                    self.rewards.append(reward); self.dones.append(done); totals[i] += reward; time_step += 1
                    if done:
                        completed += 1; scores.append(totals[i]); avg = np.mean(scores[-100:])
                        if lr_decay: self.decay_learning_rate(1.0 - (completed - 1) / episodes, writer, completed)
                        if writer:
                            writer.add_scalar("Train/Episode_Return", totals[i], completed)
                            writer.add_scalar("Train/Average_Return_100", avg, completed)
                            writer.add_scalar("Train/Episode_Length", lengths[i], completed)
                            writer.add_scalar("Train/Total_Timesteps", time_step, completed)
                            writer.add_scalar("Obs/Mean_Abs", float(np.mean(np.abs(self.obs_mean))), completed)
                            writer.add_scalar("Obs/Var_Mean", float(np.mean(self.obs_var)), completed)
                        print(f"\rEpisode {completed}\tAverage Score: {avg:.2f}", end="", flush=True)
                        if completed % 100 == 0:
                            print(f"\rEpisode {completed}\tAverage Score: {avg:.2f}", flush=True)

                        if render_interval and render_interval > 0 and completed % render_interval == 0:
                            render_score = self.render_eval_episode(seed + completed if seed is not None else SEED)
                            print(f"\nRendered eval episode at train episode {completed}: reward = {render_score:.2f}",
                                  flush=True)
                            if writer:
                                writer.add_scalar("Eval/Rendered_Return", render_score, completed)

                        totals[i], lengths[i] = 0.0, 0
                        states[i] = envs[i].reset(seed=seed + completed + i if seed is not None else None)[0]
                        if avg >= 200 and completed >= 100:
                            print(f"\nEnvironment solved in {completed:d} episodes!\tAverage Score: {avg:.2f}", flush=True); completed = episodes; break
                    else: states[i] = next_state
                if len(self.rewards) >= self.update_timestep: self.learn(writer, time_step)
        finally:
            if executor: executor.shutdown(wait=True)
            for env in envs: env.close()
        if self.rewards: self.learn(writer, time_step)
        return scores

    def render_eval_episode(self, seed=SEED):
        env = gym.make("LunarLander-v2", render_mode="human")
        state, _ = env.reset(seed=seed)
        done, total = False, 0.0
        self.policy.eval()

        while not done:
            norm_state = self.normalize_obs(state)
            s = torch.as_tensor(norm_state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action = torch.argmax(self.policy.actor(s)).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += reward

        env.close()
        self.policy.train()
        return total

    def save(self, p):
        torch.save({"Actor_state_dict": self.policy.actor.state_dict(), "Critic_state_dict": self.policy.critic.state_dict(), "obs_mean": self.obs_mean,
                    "obs_var": self.obs_var, "obs_count": self.obs_count, "obs_clip": self.obs_clip, "grad_clip": self.grad_clip}, p)

    def load(self, p):
        c = torch.load(p, map_location=self.device)
        self.policy.actor.load_state_dict(c["Actor_state_dict"]); self.policy.critic.load_state_dict(c["Critic_state_dict"])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.obs_mean = np.asarray(c.get("obs_mean", self.obs_mean), dtype=np.float32)
        self.obs_var = np.asarray(c.get("obs_var", self.obs_var), dtype=np.float32)
        self.obs_count, self.obs_clip, self.grad_clip = c.get("obs_count", self.obs_count), c.get("obs_clip", self.obs_clip), c.get("grad_clip", self.grad_clip)

def make_agent(args, device, env):
    return PPOAgent(env.observation_space.shape[0], env.action_space.n, device, args.timestep, args.epochs, args.ppo_clip, args.obs_clip, args.grad_clip, args.disc_coef, args.lr_actor, args.lr_critic)
def train_one_seed(args, seed):
    set_seed(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = args.log_dir if args.num_seeds == 1 else os.path.join(args.log_dir, f"seed_{seed}")
    writer = SummaryWriter(log_dir); probe_env = gym.make("LunarLander-v2")
    probe_env.reset(seed=seed); probe_env.action_space.seed(seed); agent = make_agent(args, device, probe_env); probe_env.close()
    print_hyperparameters(args); print("\n")
    if args.parallel_envs > 1:
        scores = agent.train_agent_parallel(
            lambda: gym.make("LunarLander-v2"), args.episodes, args.ep_max_steps, args.ep_reward_penalty,
            args.ep_timeout_penalty, args.parallel_envs, args.rollout_workers, writer, seed, args.lr_decay, args.render_interval
        )
    else:
        env = gym.make("LunarLander-v2"); env.reset(seed=seed); env.action_space.seed(seed)
        scores = agent.train_agent(
            env, args.episodes, args.ep_max_steps, args.ep_reward_penalty,
            args.ep_timeout_penalty, writer, seed, args.lr_decay, args.render_interval
        ); env.close()
    model_path = args.model if args.num_seeds == 1 else f"{os.path.splitext(args.model)[0]}_seed_{seed}{os.path.splitext(args.model)[1]}"
    agent.save(model_path); writer.close(); os.makedirs(log_dir, exist_ok=True)
    plt.plot(scores); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title(f"PPO LunarLander Training Scores Seed {seed}")
    plt.savefig(os.path.join(log_dir, "training_scores.png"), dpi=200); plt.close()
    return scores

def train(args):
    all_scores = []
    for seed in range(args.seed, args.seed + args.num_seeds):
        print(f"\nTraining seed {seed}", flush=True); all_scores.append(train_one_seed(args, seed))
    return all_scores

def render(episodes=5, model_path=MODEL_PATH, obs_clip=OBS_CLIP, grad_clip=GRAD_CLIP, seed=SEED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("LunarLander-v2", render_mode="human")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.n, device, obs_clip=obs_clip, grad_clip=grad_clip)
    agent.load(model_path); agent.policy.eval()
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep); done, total = False, 0
        while not done:
            s = torch.as_tensor(agent.normalize_obs(state), dtype=torch.float32, device=device)
            with torch.no_grad(): action = torch.argmax(agent.policy.actor(s)).item()
            state, reward, terminated, truncated, _ = env.step(action); done = terminated or truncated; total += reward
        print(f"Episode {ep + 1}: reward = {total:.2f}")
    env.close()

def play():
    from gymnasium.utils.play import play
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    keys_to_action = {(): 0, ("s",): 0, ("a",): 1, ("w",): 2, ("d",): 3, ("w", "a"): 2, ("w", "d"): 2}
    play(env, keys_to_action=keys_to_action, noop=2, fps=60); env.close()

def export_pngs(log_dir="runs/ppo_lunar_lander", out_dir="tensorboard_pngs"):
    from pathlib import Path
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    Path(out_dir).mkdir(exist_ok=True); ea = EventAccumulator(log_dir); ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        e = ea.Scalars(tag); x = [v.step for v in e]; y = [v.value for v in e]
        plt.figure(); plt.plot(x, y); plt.xlabel("Step"); plt.ylabel(tag); plt.title(tag); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, tag.replace("/", "_") + ".png"), dpi=200); plt.close()

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
    parser.add_argument("--lr-actor", type=float, default=LR_ACTOR)
    parser.add_argument("--lr-critic", type=float, default=LR_CRITIC)
    parser.add_argument("--episodes", type=int, default=EPISODES)
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
    {"train": lambda: train(args), "render": lambda: render(args.episodes, args.model, args.obs_clip, args.grad_clip, args.seed),
     "play": play, "export": lambda: export_pngs(args.log_dir)}[args.mode]()