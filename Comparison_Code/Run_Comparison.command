#!/bin/zsh
# Double-click this file, or run it from Terminal.
cd "$(dirname "$0")/.." || exit 1
if [ ! -x .venv/bin/python ]; then
  print "Create the Python environment using Comparison_Code/README.md first."
  exit 1
fi
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python Comparison_Code/run_comparison.py "$@"
