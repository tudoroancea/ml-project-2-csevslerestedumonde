#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G

module purge
module load gcc cuda py-torch py-torchvision
source venv/bin/activate
python train.py