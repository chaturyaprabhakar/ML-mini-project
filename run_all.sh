#!/usr/bin/env bash
# Runs the three experiments one after another. Each gets its OWN results folder, so nothing is overwritten:
#   results_1_main/      python src/run.py                                  (grouped + chrono, 5 seeds)
#   results_2_leakage/   python src/run.py --splits random grouped chrono   (adds paper-style random split)
#   results_3_backtest/  python src/backtest.py                             (walk-forward backtest)
# Usage:  bash run_all.sh        (progress is written to run_all.log)
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
TS=$(date +%s)
[ -d results ] && mv results "results_old_$TS"          # keep any earlier results

step () {   # step <folder-name> <command...>
  name=$1; shift
  echo "=== [$(date +%H:%M:%S)] START $name: $*"
  mkdir -p results
  "$@"
  status=$?
  rm -rf "$name"; mv results "$name"                    # freeze this step's results
  echo "=== [$(date +%H:%M:%S)] END   $name (exit code $status)"
}

step results_1_main      python -u src/run.py
step results_2_leakage   python -u src/run.py --splits random grouped chrono
step results_3_backtest  python -u src/backtest.py
python docs/build_docs.py            # builds docs/writeup.pdf and docs/presentation.pptx from the results
echo "ALL DONE"