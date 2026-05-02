# Create and publish the public GitHub repository

This directory is ready to push as:

```text
https://github.com/liuxt/tcss590-rl-coding-assignment-1
```

The package intentionally excludes the large expert `.pkl` files. Students can download them by running:

```bash
bash scripts/download_data.sh
```

## Option A: GitHub CLI

```bash
gh auth login
cd tcss590-rl-coding-assignment-1
git init
git branch -M main
git add .
git commit -m "Initial TCSS590-SP26 HW1 starter code"
gh repo create liuxt/tcss590-rl-coding-assignment-1 --public --source=. --remote=origin --push
```

## Option B: GitHub web UI plus Git

1. Create a new public repository named `tcss590-rl-coding-assignment-1` under the `liuxt` account.
2. Do not initialize it with a README, `.gitignore`, or license.
3. Push the local folder:

```bash
cd tcss590-rl-coding-assignment-1
git init
git branch -M main
git add .
git commit -m "Initial TCSS590-SP26 HW1 starter code"
git remote add origin https://github.com/liuxt/tcss590-rl-coding-assignment-1.git
git push -u origin main
```

## Colab notebook

The repository includes `TCSS590_SP26_HW1.ipynb`. After the repo is pushed, upload the notebook to Google Colab, save it in Drive, and share the resulting Colab link with students. You can also open it from GitHub using Colab's GitHub integration.

## Data files and git

The `.gitignore` file excludes `data/*.pkl`, so `git add .` will not commit expert data files even if they are present locally. This is intentional for a lighter public starter repository.
