#!/usr/bin/env bash

# Script helper to run an experiment and save output in experiments/<timestamp>/

set -e

if [ -z "$1" ]; then
  echo "Usage: ./run_experiment.sh <path_to_csv> [--save-graph <filename>] [--extra-args]"
  exit 1
fi

CSV_PATH="$1"
SAVE_GRAPH='pnl.png'
shift

# parse optional args
while (( "$#" )); do
  case "$1" in
    --save-graph)
      if [ -n "$2" ] && [[ ! "$2" =~ ^-- ]]; then
        SAVE_GRAPH="$2"
        shift
      fi
      ;;
    *)
      EXTRA_ARGS="$EXTRA_ARGS $1"
      ;;
  esac
  shift
done

TIMESTAMP=$(date +%Y%m%dT%H%M%S)
OUT_DIR="experiments/exp-${TIMESTAMP}"
mkdir -p "$OUT_DIR"

# Run
venv/bin/python main.py "$CSV_PATH" --save-graph "$OUT_DIR/$SAVE_GRAPH" $EXTRA_ARGS > "$OUT_DIR/console.log" 2>&1

# Copy the image
if [ -f "$OUT_DIR/$SAVE_GRAPH" ]; then
  echo "Saved graph to $OUT_DIR/$SAVE_GRAPH"
fi

# Save git branch/commit for reproducibility
if command -v git >/dev/null 2>&1; then
  git rev-parse --abbrev-ref HEAD > "$OUT_DIR/git_branch.txt" || true
  git rev-parse HEAD > "$OUT_DIR/git_commit.txt" || true
fi

# Create a results template
cat > "$OUT_DIR/results.md" <<- EOM
# Experiment results - $TIMESTAMP

CSV: $CSV_PATH
Graph: $OUT_DIR/$SAVE_GRAPH
Console log: $OUT_DIR/console.log

## Hypothesis

Describe the hypothesis tested.

## Configuration

Describe code changes, parameters, files modified.

## Metrics

- Sharpe: 
- PnL: 
- Max Drawdown: 

## Conclusion


EOM

echo "Experiment saved in: $OUT_DIR"

