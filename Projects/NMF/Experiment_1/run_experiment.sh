#!/usr/bin/env bash
set -euo pipefail

# --- Configuration ---
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_REPO="https://github.com/egerc/exp-runner.git"
RUNNER_COMMIT="de935f2996fdb3ee7e2098ccd17aa93c21d9eec0"
CONFIG_PATH="$EXPERIMENT_DIR/config.toml"
REQ_FILE="$EXPERIMENT_DIR/requirements.txt"
RESULTS_DIR="$EXPERIMENT_DIR/results"
LOG_FILE="$RESULTS_DIR/run.log"



uv venv "$EXPERIMENT_DIR/.venv"
source "$EXPERIMENT_DIR/.venv/bin/activate"


uv pip install --quiet git+$RUNNER_REPO@$RUNNER_COMMIT

if [ -f "$REQ_FILE" ]; then
  uv pip install --quiet -r "$REQ_FILE"
fi

mkdir -p "$RESULTS_DIR"
echo "{
  \"runner_repo\": \"$RUNNER_REPO\",
  \"runner_commit\": \"$RUNNER_COMMIT\",
  \"timestamp\": \"$(date -u +"%Y-%m-%dT%H:%M:%SZ")\",
  \"local_commit\": \"$(git -C "$EXPERIMENT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)\"
}" > "$RESULTS_DIR/metadata.json"

uv run python "$EXPERIMENT_DIR/experiment.py" --config "$CONFIG_PATH" | tee "$LOG_FILE"

echo "Experiment completed. Logs and results saved to:"
echo "   $RESULTS_DIR/"