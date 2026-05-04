import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULTS = {
    "num_epochs": 500,
    "episode_length": 50,
    "batch_size": 32,
    "num_validation_runs": 100,
}


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
        successes.append(match.group(1) == "True")
        rewards.append(float(match.group(2)))

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


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_or_default(values, default):
    """
    If user gave --num_epochs 300, values=[300].
    If omitted, values=None.
    """
    if values is None or len(values) == 0:
        return default
    return values[0]


def values_or_single(values, fallback):
    """
    If user gives a list, use the list.
    Otherwise use the fallback as a one-item list.
    """
    if values is None or len(values) == 0:
        return [fallback]
    return values


def run_main(
    main_file,
    env,
    train,
    policy,
    seed,
    num_epochs,
    episode_length,
    batch_size,
    num_validation_runs,
):
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
        "--num_validation_runs", str(num_validation_runs),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    return completed, command


def sweep_one_hyperparameter(
    sweep_name,
    sweep_values,
    baseline,
    seeds,
    args,
    output_dir,
    experiment_prefix,
):
    raw_rows = []
    seed_summary_rows = []
    hyper_summary_rows = []

    print("\n" + "=" * 80)
    print(f"Starting sweep: {sweep_name}")
    print(f"Values: {sweep_values}")
    print(
        "Fixed baseline -> "
        f"num_epochs={baseline['num_epochs']}, "
        f"episode_length={baseline['episode_length']}, "
        f"batch_size={baseline['batch_size']}, "
        f"num_validation_runs={baseline['num_validation_runs']}"
    )
    print("=" * 80)

    for sweep_value in sweep_values:
        run_params = dict(baseline)
        run_params[sweep_name] = sweep_value

        print(f"\n--- {sweep_name} = {sweep_value} ---")

        for seed in seeds:
            print(
                f"  seed={seed} | "
                f"num_epochs={run_params['num_epochs']}, "
                f"episode_length={run_params['episode_length']}, "
                f"batch_size={run_params['batch_size']}, "
                f"num_validation_runs={run_params['num_validation_runs']}"
            )

            completed, command = run_main(
                main_file=args.main_file,
                env=args.env,
                train=args.train,
                policy=args.policy,
                seed=seed,
                num_epochs=run_params["num_epochs"],
                episode_length=run_params["episode_length"],
                batch_size=run_params["batch_size"],
                num_validation_runs=run_params["num_validation_runs"],
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
                    "num_epochs": run_params["num_epochs"],
                    "episode_length": run_params["episode_length"],
                    "batch_size": run_params["batch_size"],
                    "num_validation_runs": run_params["num_validation_runs"],
                    "seed": seed,
                    "eval_episode": "",
                    "status": "failed",
                    "reward": np.nan,
                    "success": "",
                })
                continue

            rewards, successes = parse_eval_lines(completed.stdout)

            if len(rewards) == 0:
                print("    No evaluation lines found.")
                print("    STDOUT tail:")
                print(completed.stdout[-1500:])

                raw_rows.append({
                    "sweep_name": sweep_name,
                    "sweep_value": sweep_value,
                    "num_epochs": run_params["num_epochs"],
                    "episode_length": run_params["episode_length"],
                    "batch_size": run_params["batch_size"],
                    "num_validation_runs": run_params["num_validation_runs"],
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
                    "num_epochs": run_params["num_epochs"],
                    "episode_length": run_params["episode_length"],
                    "batch_size": run_params["batch_size"],
                    "num_validation_runs": run_params["num_validation_runs"],
                    "seed": seed,
                    "eval_episode": i,
                    "status": "ok",
                    "reward": reward,
                    "success": int(success),
                })

            seed_summary_rows.append({
                "sweep_name": sweep_name,
                "sweep_value": sweep_value,
                "num_epochs": run_params["num_epochs"],
                "episode_length": run_params["episode_length"],
                "batch_size": run_params["batch_size"],
                "num_validation_runs": run_params["num_validation_runs"],
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
            "num_validation_runs",
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
            "num_validation_runs",
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

    combined_plot_path = None

    if len(x_values) > 0:
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

        if args.no_show:
            plt.close()
        else:
            plt.show()

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

    seed_boxplot_path = None

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

        seed_boxplot_path = output_dir / f"{experiment_prefix}_{sweep_name}_reward_distribution_by_seed.png"
        plt.savefig(seed_boxplot_path, dpi=300, bbox_inches="tight")

        if args.no_show:
            plt.close()
        else:
            plt.show()

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
    print(f"  Raw eval results:    {raw_csv}")
    print(f"  Seed summary:        {seed_csv}")
    print(f"  Sweep summary:       {hyper_csv}")

    if combined_plot_path is not None:
        print(f"  Reward/success plot: {combined_plot_path}")

    if seed_boxplot_path is not None:
        print(f"  Seed boxplot:        {seed_boxplot_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="reacher", choices=["reacher", "pointmaze"])
    parser.add_argument("--train", type=str, default="behavior_cloning", choices=["behavior_cloning", "dagger"])
    parser.add_argument("--policy", type=str, default="gaussian", choices=["gaussian", "autoregressive"])

    # User-friendly singular flags.
    # These accept variable-length lists so they can be used both as baselines and sweep values.
    parser.add_argument("--num_epochs", type=int, nargs="+", default=None)
    parser.add_argument("--episode_length", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, nargs="+", default=None)
    parser.add_argument("--num_validation_runs", type=int, nargs="+", default=None)

    # Backward-compatible aliases.
    parser.add_argument("--num_epochs_values", type=int, nargs="+", default=None)
    parser.add_argument("--episode_length_values", type=int, nargs="+", default=None)
    parser.add_argument("--batch_size_values", type=int, nargs="+", default=None)
    parser.add_argument("--num_validation_runs_values", type=int, nargs="+", default=None)

    # Accept both --seed and --seeds.
    parser.add_argument("--seed", "--seeds", dest="seeds", type=int, nargs="+", default=[0])

    # Accept both --sweep and --sweeps.
    parser.add_argument(
        "--sweep",
        "--sweeps",
        dest="sweeps",
        type=str,
        nargs="+",
        default=["num_epochs", "episode_length", "batch_size", "num_validation_runs"],
        choices=["num_epochs", "episode_length", "batch_size", "num_validation_runs"],
    )

    parser.add_argument("--main_file", type=str, default="main.py")
    parser.add_argument("--output_dir", type=str, default="plots")
    parser.add_argument("--no_show", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Resolve baseline values.
    # If user passes --num_epochs 300, this becomes the baseline.
    baseline = {
        "num_epochs": first_or_default(args.num_epochs, DEFAULTS["num_epochs"]),
        "episode_length": first_or_default(args.episode_length, DEFAULTS["episode_length"]),
        "batch_size": first_or_default(args.batch_size, DEFAULTS["batch_size"]),
        "num_validation_runs": first_or_default(
            args.num_validation_runs,
            DEFAULTS["num_validation_runs"],
        ),
    }

    # Resolve sweep values.
    # Priority:
    #   1. explicit *_values flag
    #   2. singular flag list, e.g. --num_epochs 100 300 500
    #   3. one-item list using baseline
    sweep_values_by_name = {
        "num_epochs": values_or_single(
            args.num_epochs_values if args.num_epochs_values is not None else args.num_epochs,
            baseline["num_epochs"],
        ),
        "episode_length": values_or_single(
            args.episode_length_values if args.episode_length_values is not None else args.episode_length,
            baseline["episode_length"],
        ),
        "batch_size": values_or_single(
            args.batch_size_values if args.batch_size_values is not None else args.batch_size,
            baseline["batch_size"],
        ),
        "num_validation_runs": values_or_single(
            args.num_validation_runs_values
            if args.num_validation_runs_values is not None
            else args.num_validation_runs,
            baseline["num_validation_runs"],
        ),
    }

    experiment_prefix = (
        f"{args.env}_{args.train}_{args.policy}"
        f"_baseEpochs{baseline['num_epochs']}"
        f"_baseEpLen{baseline['episode_length']}"
        f"_baseBatch{baseline['batch_size']}"
        f"_baseValRuns{baseline['num_validation_runs']}"
    )

    config = {
        **vars(args),
        "baseline": baseline,
        "sweep_values_by_name": sweep_values_by_name,
    }

    config_path = output_dir / f"{experiment_prefix}_config.json"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved config: {config_path}")
    print(f"Resolved baseline: {baseline}")
    print(f"Resolved sweeps: {args.sweeps}")
    print(f"Resolved sweep values: {sweep_values_by_name}")

    for sweep_name in args.sweeps:
        sweep_one_hyperparameter(
            sweep_name=sweep_name,
            sweep_values=sweep_values_by_name[sweep_name],
            baseline=baseline,
            seeds=args.seeds,
            args=args,
            output_dir=output_dir,
            experiment_prefix=experiment_prefix,
        )


if __name__ == "__main__":
    main()