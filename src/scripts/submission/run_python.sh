#!/bin/bash -ex
#SBATCH --partition=sm3090
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --time=0-40:00:00 # 2 days of runtime (can be set to 7 days)
#SBATCH --gres=gpu:RTX3090:8

source ~/.bashrc
conda activate sparkle

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/energy/amawi/projects/nn_ansatz/src
echo "${@:2}"

python /home/energy/amawi/projects/nn_ansatz/src/measure_time.py

