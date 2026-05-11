# TCSS590-SP26 Homework 2 - Policy Gradient and Actor Critic

This repository contains the starter code for **TCSS590-SP26: Theory and Algorithms of Reinforcement Learning**, Homework 2.

## Setup and Installation

The setup is the same as Homework 1, so you may reuse that environment if it already works for you.

### Install MuJoCo

1. Download the MuJoCo version 2.1 binaries for
   [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or
   [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
3. Add `resources/mjkey.txt` in the repo into `~/.mujoco/mujoco210` if your setup requires it.

### Setup environment

To set up the project environment, use the `environment.yml` file. It contains the necessary dependencies and installation instructions.

```bash
conda env create -f environment.yml
conda activate tcss590hw2
```

### Install LibGLEW

```bash
sudo apt-get install libglew-dev
sudo apt-get install patchelf
```

### Export path variables

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

### Compile `mujoco_py` only once

```bash
python -c "import mujoco_py"
```

## Training

```bash
python main.py --task policy_gradient
python main.py --task actor_critic
```

## Evaluation

```bash
python main.py --task policy_gradient --test --render
python main.py --task actor_critic --test --render
```

## Colab

A Colab notebook is included as `TCSS590_SP26_HW2.ipynb`. The notebook mirrors the starter-code structure and is intended for students who prefer to run the assignment in Google Colab.
