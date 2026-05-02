# TCSS590-SP26 Homework 1: Supervised Learning of Behaviors

Starter code for **TCSS590-SP26: Theory and Algorithms of Reinforcement Learning**.

This homework covers supervised learning of behaviors, including behavior cloning, DAgger, Gaussian policies, and autoregressive policies.

## Repository

Intended public repository URL:

```text
https://github.com/liuxt/tcss590-rl-coding-assignment-1
```

## Setup

The starter code is written in Python 3.10 and depends on NumPy, Matplotlib, PyTorch, Gymnasium, Gymnasium Robotics, and MuJoCo. It runs on Windows, macOS, and Linux without a MuJoCo binary download or compiler — everything ships as prebuilt wheels.

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate tcss590hw1
```

Always run the code from the **repository root** so that local modules (e.g. `policy.py`, `reach_goal/`) are importable.

### Platform notes

- **Windows:** use Anaconda Prompt, PowerShell, or Git Bash. `scripts/download_data.sh` requires Git Bash or WSL; on plain PowerShell, download the three `.pkl` files manually into `data/` using the URLs in that script.
- **macOS (Intel and Apple Silicon):** works out of the box with the wheels pinned in `environment.yml`.
- **Linux:** works out of the box. If you want on-screen rendering with `--render` and you hit a `GLEW` error, export `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so` before launching. `main.py` sets this automatically on Linux only.

## Data files

The expert datasets and expert policy are large binary `.pkl` files and are not included. Download them with:

```bash
bash scripts/download_data.sh
```

The script downloads into `data/`:

- `reacher_expert_data.pkl`
- `reacher_expert_policy.pkl`
- `pointmaze_expert_data.pkl`

## Google Colab

The repository includes a starter-only Colab notebook:

```text
TCSS590_SP26_HW1.ipynb
```

After the public GitHub repository is created, students can either open the notebook from GitHub in Colab or upload the notebook directly to Colab. The notebook installs dependencies, clones or uploads the starter project, downloads expert data if missing, and provides run cells for the required experiments. It does not contain solution code.

## Running

Behavior cloning with a Gaussian policy:

```bash
python main.py --env reacher --train behavior_cloning --policy gaussian
python main.py --env pointmaze --train behavior_cloning --policy gaussian
```

DAgger with a Gaussian policy:

```bash
python main.py --env reacher --train dagger --policy gaussian
python main.py --env pointmaze --train dagger --policy gaussian
```

Autoregressive policies:

```bash
python main.py --env pointmaze --train behavior_cloning --policy autoregressive
python main.py --env pointmaze --train dagger --policy autoregressive
```

## Student TODOs

- `bc.py`: implement behavior cloning.
- `dagger.py`: implement DAgger.
- `utils.py`: implement `PolicyAutoRegressiveModel`.
- Do not modify `evaluate.py`.
- Modify `main.py` only when tuning hyperparameters requested by the homework.

## Submission

Submit the written answers as a PDF and submit the completed working directory as a zip file through Canvas.

## Support files

- `run_experiments.py`: optional helper script for running default experiments and saving plots/metrics.
