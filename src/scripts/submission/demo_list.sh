#!/bin/bash -ex
#SBATCH --partition=sm3090
#SBATCH -N 1      # Minimum of 1 node
#SBATCH -n 8     # 8 MPI processes per node
#SBATCH --time=0-24:00:00 # 2 days of runtime (can be set to 7 days)
# #SBATCH --gres=gpu:RTX3090:$1 # Request 1 GPU (can increase for more)

source ~/.bashrc

# module load foss
# # #SBATCH -o junk.out
# nvidia-smi
# nvcc --version
conda activate base

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/energy/amawi/projects/nn_ansatz/src

echo $1

python run_with_cfg_list.py "$@"
