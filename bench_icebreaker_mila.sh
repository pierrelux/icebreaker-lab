#!/bin/bash
#SBATCH --job-name=icebreaker-bench
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=bench-%j.out
# One short GPU job: installs JAX in the node's temporary folder (nothing lands in $HOME), then times batched rollouts.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UV_CACHE_DIR="$SLURM_TMPDIR/uv-cache" UV_PYTHON_INSTALL_DIR="$SLURM_TMPDIR/uv-python"
UV="$HOME/.local/bin/uv"
"$UV" venv --quiet --python 3.12 "$SLURM_TMPDIR/env"
"$UV" pip install --quiet --python "$SLURM_TMPDIR/env/bin/python" "jax[cuda12]"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$SLURM_TMPDIR/env/bin/python" bench_icebreaker_jax.py 64 128 256 512 1024 2048 4096
"$SLURM_TMPDIR/env/bin/python" bench_icebreaker_jax.py --fast 64 256 1024 4096
