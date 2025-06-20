#!/usr/bin/env bash
set -e
mamba env create -f environment.yml
conda activate rlproj
pre-commit install
echo "✅ Local dev environment ready."
