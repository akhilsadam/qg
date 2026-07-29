#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm-%x-%j.out
module load miniforge
cd ~/QG/qg
python -u src/qg/train.py scenario=$1 wandb.mode=disabled hydra.run.dir=runs/qg/$1
