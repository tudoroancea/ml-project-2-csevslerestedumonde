#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
#SBATCH --mem=32G

module purge
module load gcc cuda python py-torch py-torchvision
python train.py --loss $1 --epochs $2