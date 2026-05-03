import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_eval_lines(stdout: str):
    """
    Extract evaluation results from lines like:

        test 99, success False, reward -13.7725877272614

    Returns:
        rewards: list[float]
        successes: list[bool]
    """
    pattern = re.compile(
        r"test\s+\d+,\s+success\s+(True|False),\s+reward\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    )

    rewards = []
    successes = []

    for match in pattern.finditer(stdout):
        success = match.group(1) == "True"
        reward = float(match.group(2))

        successes.append(success)
        rewards.append(reward)

    return rewards, successes


def summarize_numeric(values):
    arr = np.array(values, dtype=float)

    if len(arr) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "median": np.nan,
            "max": np.nan,
        }

    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def run_main(main_file, env, train, policy, seed, num_epochs, episode_length, batch_size):
    command = [
        sys.executable,
        main_file,
        "--env", env,
        "--train", train,
        "--policy", policy,
        "--seed", str(seed),
        "--num_epochs", str(num_epochs),
        "--episode_length", str(episode_length),
        "--batch_size", str(batch_size),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    return completed, command


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sweep_one_hyperparameter(
    sweep_name,
    sweep_values,
    fixed_num_epochs,
    fixed_episode_length,
    fixed_batch_size,
    seeds,
    args,
    output_dir,
    experiment_prefix,
    show_plots=True,
):
    raw_rows = []
    seed_summary_rows = []
    hyper_summary_rows = []

    print(f"\n{'=' * 70}")
    print(f"Starting sweep: {sweep_name}")
    print(f"Values: {sweep_values}")
    print(
        f"Fixed settings -> num_epochs={fixed_num_epochs}, "
        f"episode_length={fixed_episode_length}, batch_size={fixed_batch_size}"
    )
    print(f"{'=' * 70}")

    for sweep_value in sweep_values:
        print(f"\n--- {sweep_name} = {sweep_value} ---")

        if sweep_name == "num_epochs":
            num_epochs = sweep_value
            episode_length = fixed_episode_length
            batch_size = fixed_batch_size
        elif sweep_name == "episode_length":
            num_epochs = fixed_num_epochs
            episode_length = sweep_value
            batch_size = fixed_batch_size
        elif sweep_name == "batch_size":
            num_epochs = fixed_num_epochs
            episode_length = fixed_episode_length
            batch_size = sweep_value
        else:
            raise ValueError(f"Unknown sweep_name: {sweep_name}")

        for seed in seeds:
            print(
                f"  seed={seed} | num_epochs={num_epochs}, "
                f"episode_length={episode_length}, batch_size={batch_size}"
            )

            completed, command = run_main(
                main_file=args.main_file,
                env=args.env,
                train=args.train,
                policy=args.policy,
                seed=seed,
                num_epochs=num_epochs,
                episode_length=episode_length,
                batch_size=batch_size,
            )

            if completed.returncode != 0:
                print("    Run failed.")
                print("    Command:", " ".join(command))
                print("    STDERR:")
                print(completed.stderr)
                print("    STDOUT tail:")
                print(completed.stdout[-1500:])

                raw_rows.append({
                    "sweep_name": sweep_name,
                    "sweep_value": sweep_value,
                    "num_epochs": num_epochs,
                    "episode_length": episode_length,
                    "batch_size": batch_size,
                    "seed": seed,
                    "eval_episode": "",
                    "status": "failed",
                    "reward": np.nan,
                    "success": "",
                })
                continue

            rewards, successes = parse_eval_lines(completed.stdout)

            if len(rewards) == 0:
                print("    No evaluation rewards found in output.")
                print("    STDOUT tail:")
                print(completed.stdout[-1500:])

                raw_rows.append({
                    "sweep_name": sweep_name,
                    "sweep_value": sweep_value,
                    "num_epochs": num_epochs,
                    "episode_length": episode_length,
                    "batch_size": batch_size,
                    "seed": seed,
                    "eval_episode": "",
                    "status": "no_eval_lines_found",
                    "reward": np.nan,
                    "success": "",
                })
                continue

            reward_stats = summarize_numeric(rewards)
            success_rate = float(np.mean(successes))

            print(
                f"    eval episodes={reward_stats['count']}, "
                f"mean reward={reward_stats['mean']:.3f}, "
                f"std reward={reward_stats['std']:.3f}, "
                f"success rate={success_rate:.3f}"
            )

            for i, (reward, success) in enumerate(zip(rewards, successes)):
                raw_rows.append({
                    "sweep_name": sweep_name,
                    "sweep_value": sweep_value,
                    "num_epochs": num_epochs,
                    "episode_length": episode_length,
                    "batch_size": batch_size,
                    "seed": seed,
                    "eval_episode": i,
                    "status": "ok",
                    "reward": reward,
                    "success": int(success),
                })

            seed_summary_rows.append({
                "sweep_name": sweep_name,
                "sweep_value": sweep_value,
                "num_epochs": num_epochs,
                "episode_length": episode_length,
                "batch_size": batch_size,
                "seed": seed,
                "eval_count": reward_stats["count"],
                "mean_reward": reward_stats["mean"],
                "std_reward": reward_stats["std"],
                "min_reward": reward_stats["min"],
                "median_reward": reward_stats["median"],
                "max_reward": reward_stats["max"],
                "success_rate": success_rate,
            })

    for sweep_value in sweep_values:
        reward_means = [
            row["mean_reward"]
            for row in seed_summary_rows
            if row["sweep_value"] == sweep_value
        ]

        success_rates = [
            row["success_rate"]
            for row in seed_summary_rows
            if row["sweep_value"] == sweep_value
        ]

        reward_stats = summarize_numeric(reward_means)
        success_stats = summarize_numeric(success_rates)

        hyper_summary_rows.append({
            "sweep_name": sweep_name,
            "sweep_value": sweep_value,
            "seed_count": reward_stats["count"],
            "mean_reward": reward_stats["mean"],
            "std_reward": reward_stats["std"],
            "min_reward": reward_stats["min"],
            "median_reward": reward_stats["median"],
            "max_reward": reward_stats["max"],
            "mean_success_rate": success_stats["mean"],
            "std_success_rate": success_stats["std"],
            "min_success_rate": success_stats["min"],
            "median_success_rate": success_stats["median"],
            "max_success_rate": success_stats["max"],
        })

    raw_csv = output_dir / f"{experiment_prefix}_{sweep_name}_raw_eval_results.csv"
    seed_csv = output_dir / f"{experiment_prefix}_{sweep_name}_seed_summary.csv"
    hyper_csv = output_dir / f"{experiment_prefix}_{sweep_name}_summary.csv"

    save_csv(
        raw_csv,
        [
            "sweep_name",
            "sweep_value",
            "num_epochs",
            "episode_length",
            "batch_size",
            "seed",
            "eval_episode",
            "status",
            "reward",
            "success",
        ],
        raw_rows,
    )

    save_csv(
        seed_csv,
        [
            "sweep_name",
            "sweep_value",
            "num_epochs",
            "episode_length",
            "batch_size",
            "seed",
            "eval_count",
            "mean_reward",
            "std_reward",
            "min_reward",
            "median_reward",
            "max_reward",
            "success_rate",
        ],
        seed_summary_rows,
    )

    save_csv(
        hyper_csv,
        [
            "sweep_name",
            "sweep_value",
            "seed_count",
            "mean_reward",
            "std_reward",
            "min_reward",
            "median_reward",
            "max_reward",
            "mean_success_rate",
            "std_success_rate",
            "min_success_rate",
            "median_success_rate",
            "max_success_rate",
        ],
        hyper_summary_rows,
    )

    x_values = []
    mean_rewards = []
    std_rewards = []
    mean_success_rates = []
    std_success_rates = []

    for row in hyper_summary_rows:
        if row["seed_count"] > 0:
            x_values.append(row["sweep_value"])
            mean_rewards.append(row["mean_reward"])
            std_rewards.append(row["std_reward"])
            mean_success_rates.append(row["mean_success_rate"])
            std_success_rates.append(row["std_success_rate"])

    # ------------------------------------------------------------------
    # Combined two-segment plot:
    # Segment 1: average reward
    # Segment 2: success rate
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

    axes[0].errorbar(
        x_values,
        mean_rewards,
        yerr=std_rewards,
        marker="o",
        capsize=5,
    )
    axes[0].set_ylabel("Average Evaluation Reward")
    axes[0].set_title(
        f"{args.env} / {args.train} / {args.policy}\n"
        f"Sweep: {sweep_name}"
    )
    axes[0].grid(True)

    axes[1].errorbar(
        x_values,
        mean_success_rates,
        yerr=std_success_rates,
        marker="o",
        capsize=5,
    )
    axes[1].set_xlabel(sweep_name)
    axes[1].set_ylabel("Success Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True)

    combined_plot_path = output_dir / f"{experiment_prefix}_{sweep_name}_reward_and_success.png"
    plt.tight_layout()
    plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ------------------------------------------------------------------
    # Box-whisker reward distribution by seed
    # ------------------------------------------------------------------
    seed_labels = []
    seed_reward_lists = []

    for seed in seeds:
        rewards = [
            row["reward"]
            for row in raw_rows
            if row["status"] == "ok" and row["seed"] == seed
        ]

        if len(rewards) > 0:
            seed_labels.append(str(seed))
            seed_reward_lists.append(rewards)

    if len(seed_reward_lists) > 0:
        plt.figure(figsize=(8, 5))
        plt.boxplot(seed_reward_lists, tick_labels=seed_labels, showmeans=True)
        plt.xlabel("Seed")
        plt.ylabel("Evaluation Reward")
        plt.title(
            f"Reward Distribution by Seed\n"
            f"{args.env} / {args.train} / {args.policy} | Sweep: {sweep_name}"
        )
        plt.grid(True)

        seed_plot_path = output_dir / f"{experiment_prefix}_{sweep_name}_reward_distribution_by_seed.png"
        plt.savefig(seed_plot_path, dpi=300, bbox_inches="tight")

        if show_plots:
            plt.show()
        else:
            plt.close()
    else:
        seed_plot_path = None

    print(f"\nSummary for sweep: {sweep_name}")
    print(
        f"{'value':>12} | {'seeds':>5} | {'mean reward':>12} | "
        f"{'std reward':>11} | {'success':>8} | {'success std':>11}"
    )
    print("-" * 76)

    for row in hyper_summary_rows:
        print(
            f"{row['sweep_value']:>12} | "
            f"{row['seed_count']:>5} | "
            f"{row['mean_reward']:>12.3f} | "
            f"{row['std_reward']:>11.3f} | "
            f"{row['mean_success_rate']:>8.3f} | "
            f"{row['std_success_rate']:>11.3f}"
        )

    print("\nSaved files:")
    print(f"  Raw eval results:        {raw_csv}")
    print(f"  Seed summary:            {seed_csv}")
    print(f"  Sweep summary:           {hyper_csv}")
    print(f"  Reward/success plot:     {combined_plot_path}")
    if seed_plot_path is not None:
        print(f"  Seed reward boxplot:     {seed_plot_path}")

    return {
        "raw_rows": raw_rows,
        "seed_summary_rows": seed_summary_rows,
        "hyper_summary_rows": hyper_summary_rows,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="reacher", choices=["reacher", "pointmaze"])
    parser.add_argument("--train", type=str, default="behavior_cloning", choices=["behavior_cloning", "dagger"])
    parser.add_argument("--policy", type=str, default="gaussian", choices=["gaussian", "autoregressive"])

    parser.add_argument(
        "--num_epochs_values",
        type=int,
        nargs="+",
        default=[10, 50, 100, 250, 500],
        help="Values for num_epochs sweep.",
    )

    parser.add_argument(
        "--episode_length_values",
        type=int,
        nargs="+",
        default=[25, 50, 100],
        help="Values for episode_length sweep.",
    )

    parser.add_argument(
        "--batch_size_values",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128],
        help="Values for batch_size sweep.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
        help="Random seeds.",
    )

    parser.add_argument("--base_num_epochs", type=int, default=500)
    parser.add_argument("--base_episode_length", type=int, default=50)
    parser.add_argument("--base_batch_size", type=int, default=32)

    parser.add_argument("--main_file", type=str, default="main.py")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--no_show", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    experiment_prefix = (
        f"{args.env}_{args.train}_{args.policy}"
        f"_baseEpochs{args.base_num_epochs}"
        f"_baseEpLen{args.base_episode_length}"
        f"_baseBatch{args.base_batch_size}"
    )

    config_path = output_dir / f"{experiment_prefix}_config.json"

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Saved config: {config_path}")

    sweep_one_hyperparameter(
        sweep_name="num_epochs",
        sweep_values=args.num_epochs_values,
        fixed_num_epochs=args.base_num_epochs,
        fixed_episode_length=args.base_episode_length,
        fixed_batch_size=args.base_batch_size,
        seeds=args.seeds,
        args=args,
        output_dir=output_dir,
        experiment_prefix=experiment_prefix,
        show_plots=not args.no_show,
    )

    sweep_one_hyperparameter(
        sweep_name="episode_length",
        sweep_values=args.episode_length_values,
        fixed_num_epochs=args.base_num_epochs,
        fixed_episode_length=args.base_episode_length,
        fixed_batch_size=args.base_batch_size,
        seeds=args.seeds,
        args=args,
        output_dir=output_dir,
        experiment_prefix=experiment_prefix,
        show_plots=not args.no_show,
    )

    sweep_one_hyperparameter(
        sweep_name="batch_size",
        sweep_values=args.batch_size_values,
        fixed_num_epochs=args.base_num_epochs,
        fixed_episode_length=args.base_episode_length,
        fixed_batch_size=args.base_batch_size,
        seeds=args.seeds,
        args=args,
        output_dir=output_dir,
        experiment_prefix=experiment_prefix,
        show_plots=not args.no_show,
    )


if __name__ == "__main__":
    main()