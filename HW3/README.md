# TCSS590-SP26 Assignment 3 - Model-Based RL

Starter code for **TCSS590-SP26: Theory and Algorithms of Reinforcement Learning**.

In this assignment, you will implement model-based reinforcement learning methods for `Reacher-v4`:

1. Random MPC with shooting using a learned single dynamics model
2. Model Predictive Path Integral (MPPI) control using a learned single dynamics model
3. Ensemble MPPI using multiple learned dynamics models

## Setup and installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate tcss590-hw3
```

The code supports local CPU or GPU execution. CUDA will be used automatically when PyTorch detects an available GPU.

## Running the assignment

```bash
python main.py --model_type single --plan_mode random_mpc
python main.py --model_type single --plan_mode mppi
python main.py --model_type ensemble --plan_mode mppi
```

Append `--test` to evaluate saved checkpoints instead of training:

```bash
python main.py --model_type single --plan_mode random_mpc --test
```

## Files you need to touch

More details are in the assignment PDF.

- `planning.py` - random MPC, MPPI, and ensemble MPPI TODOs
- `train_model.py` - ensemble model training TODO

You may tune hyperparameters in `main.py`, but do not change the evaluation protocol.

## Colab notebook

The included notebook `TCSS590_SP26_HW3.ipynb` mirrors the starter files and can be uploaded to Google Colab or opened from GitHub after this repository is public:

```text
https://colab.research.google.com/github/liuxt/tcss590-rl-coding-assignment-3/blob/main/TCSS590_SP26_HW3.ipynb
```
