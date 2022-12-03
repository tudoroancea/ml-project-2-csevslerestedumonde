#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=32G

module purge
module load gcc cuda py-torch py-torchvision
python train.py