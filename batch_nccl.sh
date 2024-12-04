#!/bin/bash
#SBATCH -C gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH -o slurm-nccl-%j.out
#SBATCH -t 0:02:00

module load pytorch/2.3.1
module list
export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

srun -u -l python test_nccl.py
