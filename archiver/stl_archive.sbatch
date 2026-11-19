#!/bin/bash
#SBATCH --job-name=stl-archive
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/network/projects/plurl/datasets/st-lawrence/logs/slurm-%j.out
#
# Daily St. Lawrence data archive (see stl_archive.py). Mila has no cron for users (scrontab is disabled),
# so each run first queues the next one for tomorrow 09:00 local time, then archives yesterday (UTC).
# The chain survives logouts and the end of any Claude session. It breaks only if a submission fails;
# check it with:  squeue -u $USER -n stl-archive   and   cat /network/projects/plurl/datasets/st-lawrence/STATUS.json
# Restart it with:  sbatch /network/projects/plurl/datasets/st-lawrence/code/stl_archive.sbatch
# Stop it with:     scancel -n stl-archive
set -u
ROOT=/network/projects/plurl/datasets/st-lawrence
SELF=$ROOT/code/stl_archive.sbatch

if [ -z "${NO_CHAIN:-}" ] && [ -z "$(squeue -h -u "$USER" -n stl-archive -t PENDING -o %i)" ]; then
  sbatch --begin="$(date -d 'tomorrow 09:00' +%Y-%m-%dT%H:%M)" "$SELF" \
    || echo "WARNING: could not queue tomorrow's run" >&2
fi

export UV_CACHE_DIR=$SCRATCH/.cache/uv UV_PYTHON_INSTALL_DIR=$SCRATCH/.local/uv-python
# Catch up on up to 3 missed days each time; files already archived are skipped.
~/.local/bin/uv run --quiet "$ROOT/code/stl_archive.py" --root "$ROOT/data" --backfill "${BACKFILL:-3}" --workers 6
