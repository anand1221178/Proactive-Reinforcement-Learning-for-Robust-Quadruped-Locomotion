#!/bin/bash
#SBATCH --job-name=${exp_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j.out

module load cuda/11.8  # or your cluster's module names
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rlproj

srun python -m rl_project.training.train "$@"


//run with sbatch scripts/submit_job.slurm algo=ppo total_timesteps=1e8
