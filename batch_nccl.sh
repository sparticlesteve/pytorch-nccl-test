#!/bin/bash
#SBATCH -C gpu -N 2 -t 5
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH -o slurm-nccl-%j.out

module load pytorch/v1.3.1-gpu
module list
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

srun -u -l python test_nccl.py
