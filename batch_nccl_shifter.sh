#!/bin/bash
#SBATCH -C gpu -N 2 -t 5
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --image=registry.services.nersc.gov/wbhimji/nvidia-pytorch:19.12-py3
#SBATCH --volume="/dev/infiniband:/sys/class/infiniband_verbs"
#SBATCH --exclusive
#SBATCH -o slurm-nccl-shifter-%j.out

# This example script uses the nvidia-pytorch docker container via shifter

export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL

srun -u -l shifter python test_nccl.py
