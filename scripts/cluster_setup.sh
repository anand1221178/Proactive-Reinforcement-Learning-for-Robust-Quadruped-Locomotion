#!/usr/bin/env bash
set -e
module load miniconda/23.1 cuda/11.8 git
conda env create -f environment.yml
conda activate rlproj
pre-commit install
echo "✅ Cluster env ready."
