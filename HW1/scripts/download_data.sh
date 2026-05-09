#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
BASE="https://github.com/WEIRDLabUW/cse542_sp24_hw1/raw/refs/heads/master/data"

echo "Downloading reacher_expert_data.pkl..."
curl -L "$BASE/reacher_expert_data.pkl" -o data/reacher_expert_data.pkl

echo "Downloading reacher_expert_policy.pkl..."
curl -L "$BASE/reacher_expert_policy.pkl" -o data/reacher_expert_policy.pkl

echo "Downloading pointmaze_expert_data.pkl..."
curl -L "$BASE/pointmaze_expert_data.pkl" -o data/pointmaze_expert_data.pkl

echo "Done."
