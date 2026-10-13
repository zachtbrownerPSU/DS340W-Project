#!/usr/bin/env bash

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$project_root/.bin:$PATH"
export PYTHONPATH="$project_root/.python-packages${PYTHONPATH:+:$PYTHONPATH}"
