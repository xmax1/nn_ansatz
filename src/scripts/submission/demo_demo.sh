#!/bin/bash

source ~/.bashrc
# module load foss
# nvidia-smi
# nvcc --version
conda activate pansatz

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd /home/energy/amawi/projects/nn_ansatz/src

echo "$@"
# python run_with_args.py 

## SBATCH --mail-type=END,FAIL
