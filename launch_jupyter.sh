#!/bin/bash -l
#SBATCH --job-name=ipython-trial
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output jupyter-log-%J.out

module load gcc cuda mvapich2

source venv/bin/activate

ipnport=$(shuf -i8000-9999 -n1)

jupyter-notebook --no-browser --port=${ipnport} --ip=$(hostname -i)
