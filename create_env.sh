#!/usr/bin/env zsh
# Create the conda environment from environment.yml (macOS, zsh)
# Usage: ./create_env.sh
set -euo pipefail
ENV_FILE="$(cd "$(dirname "$0")" && pwd)/environment.yml"
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please install Miniconda or Anaconda and ensure 'conda' is in your PATH." >&2
  exit 1
fi

echo "Creating conda environment from $ENV_FILE..."
conda env create -f "$ENV_FILE"

echo "To activate the environment run:"
echo "  conda activate usps-py38"
