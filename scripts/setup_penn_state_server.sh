#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
mkdir -p .bin .python-packages
ln -sf "$(command -v python3)" .bin/python
export PATH="$project_root/.bin:$PATH"
export PYTHONPATH="$project_root/.python-packages${PYTHONPATH:+:$PYTHONPATH}"
python3 -m pip install --target .python-packages -r requirements.txt
