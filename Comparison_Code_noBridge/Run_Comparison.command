#!/bin/zsh
# Double-click this file, or run it from Terminal.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  print "Build the SWAP-only TKET copy into .venv first."
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python run_comparison.py "$@"
