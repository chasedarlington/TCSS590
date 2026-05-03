"""Run BC + DAgger on Reacher + PointMaze, save loss plots and metrics."""
import os
import json
import argparse
import numpy as np
import torch # --index-url https://download.pytorch.org/whl/cpu
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import get_expert_data, PolicyGaussian, PolicyAutoRegressiveModel, rollout
from bc import simulate_policy_bc
from dagger import simulate_policy_dagger
import pytorch_utils as ptu
from reach_goal.envs.pointmaze_env import PointMazeEnv
from reach_goal.envs.pointmaze_expert import WaypointController

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using device", device)
import sys
import subprocess
import yaml

YAML_FILE = "environment.yml"

with open(YAML_FILE, "r") as f:
    env = yaml.safe_load(f)

pip_packages = []

for dep in env.get("dependencies", []):
    # Handles nested pip section:
    # - pip:
    #     - package==version
    if isinstance(dep, dict) and "pip" in dep:
        pip_packages.extend(dep["pip"])

    # Handles simple string dependencies:
    # - numpy=1.26.4
    # - matplotlib=3.8.4
    elif isinstance(dep, str):
        # Skip Python itself and pip
        if dep.startswith("python") or dep == "pip":
            continue

        # Convert conda format package=version to pip format package==version
        if "=" in dep and "==" not in dep:
            parts = dep.split("=")
            if len(parts) >= 2:
                package = parts[0]
                version = parts[1]
                pip_packages.append(f"{package}=={version}")
        else:
            pip_packages.append(dep)

print("Installing packages:")
for pkg in pip_packages:
    print("  ", pkg)

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--upgrade",
    "--force-reinstall",
    *pip_packages
])

torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)

OUT = "results"
os.makedirs(OUT, exist_ok=True)


def make_env(env_name):
    if env_name == "reacher":
        return gym.make("Reacher-v4")
    return PointMazeEnv(render_mode="rgb_array")


def make_policy(env, policy_name, flattened_expert):
    obs_size = env.observation_space.shape[0]
    ac_size = env.action_space.shape[0]
    hidden_dim, hidden_depth = 128, 2
    if policy_name == "gaussian":
        return PolicyGaussian(obs_size, ac_size, hidden_dim, hidden_depth)
    margin = 0.1
    return PolicyAutoRegressiveModel(
        obs_size, ac_size, hidden_dim, hidden_depth, num_buckets=10,
        ac_low=flattened_expert["actions"].min(axis=0) - margin,
        ac_high=flattened_expert["actions"].max(axis=0) + margin,
    )


def evaluate_policy(env, policy, agent_name, env_name, episode_length, n=100):
    success, rew_suc, rew_all = 0, 0.0, 0.0
    for _ in range(n):
        path = rollout(env, policy, agent_name=agent_name, episode_length=episode_length)
        if env_name == "reacher":
            inner = env.unwrapped
            ok = np.linalg.norm(inner.get_body_com("fingertip") - inner.get_body_com("target")) < 0.1
        else:
            ok = sum(path["dones"]) > 0
        r = float(np.sum(path["rewards"]))
        if ok:
            success += 1
            rew_suc += r
        rew_all += r
    return {
        "success_rate": success / n,
        "avg_reward_success": rew_suc / max(success, 1),
        "avg_reward_all": rew_all / n,
    }


def run_one(env_name, train, policy_name):
    tag = f"{policy_name}_{env_name}_{train}"
    print(f"\n===== {tag} =====")

    expert_path = f"data/{env_name}_expert_data.pkl"
    expert_data = get_expert_data(expert_path)

    flat = {"observations": [], "actions": []}
    for p in expert_data:
        for k in flat:
            flat[k].append(p[k])
    for k in flat:
        flat[k] = np.concatenate(flat[k])

    env = make_env(env_name)
    policy = make_policy(env, policy_name, flat).to(device)

    if env_name == "reacher":
        episode_length, num_epochs, batch_size = 50, 500, 32
    else:
        episode_length, num_epochs, batch_size = 300, 10, 128

    if train == "behavior_cloning":
        losses = simulate_policy_bc(env, policy, expert_data,
                                    num_epochs=num_epochs,
                                    episode_length=episode_length,
                                    batch_size=batch_size)
        flat_losses = list(losses)
    else:
        if env_name == "reacher":
            expert_policy = torch.load("data/reacher_expert_policy.pkl",
                                       map_location=device, weights_only=False)
            expert_policy.to(device)
            ptu.set_gpu_mode(torch.cuda.is_available())
        else:
            expert_policy = WaypointController(env.maze)
        num_dagger_iters = 10
        ne = num_epochs // num_dagger_iters if num_epochs >= num_dagger_iters else 1
        losses, returns = simulate_policy_dagger(
            env, policy, expert_data, expert_policy,
            num_epochs=ne, episode_length=episode_length,
            batch_size=batch_size, num_dagger_iters=num_dagger_iters,
            num_trajs_per_dagger=10,
        )
        flat_losses = [v for it in losses for v in it]

    torch.save(policy.state_dict(), f"{tag}_final.pth")
    np.save(os.path.join(OUT, f"{tag}_losses.npy"), np.array(flat_losses))

    metrics = evaluate_policy(env, policy, train, env_name, episode_length, n=100)
    metrics["env"] = env_name
    metrics["train"] = train
    metrics["policy"] = policy_name
    print(tag, "->", metrics)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(flat_losses)
    ax.set_xlabel("epoch (or dagger-epoch step)")
    ax.set_ylabel("loss = -mean log prob")
    ax.set_title(f"{tag} training loss")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{tag}_loss.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT, f"{tag}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="filter by tag substring")
    args = ap.parse_args()

    runs = [
        ("reacher", "behavior_cloning", "gaussian"),
        ("pointmaze", "behavior_cloning", "gaussian"),
        ("reacher", "dagger", "gaussian"),
        ("pointmaze", "dagger", "gaussian"),
    ]

    summary = []
    for env_name, train, policy_name in runs:
        tag = f"{policy_name}_{env_name}_{train}"
        if args.only and not any(s in tag for s in args.only):
            continue
        try:
            m = run_one(env_name, train, policy_name)
            summary.append(m)
        except Exception as e:
            print(f"FAILED {tag}: {e!r}")
            summary.append({"env": env_name, "train": train, "policy": policy_name, "error": str(e)})

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY ===")
    for m in summary:
        print(m)


if __name__ == "__main__":
    main()
