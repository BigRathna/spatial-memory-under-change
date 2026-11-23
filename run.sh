#!/usr/bin/env bash
# Run the spatial-memory benchmark demo with the project virtualenv.
# Creates nothing; expects .venv to exist (see README.md).
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m src.demo --seeds "${1:-3}"
