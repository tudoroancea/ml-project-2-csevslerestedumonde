#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

module purge
module load gcc cuda py-torch py-torchvision
source venv/bin/activate
python train_script.py