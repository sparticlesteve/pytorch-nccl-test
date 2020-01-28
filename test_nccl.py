import os

import torch
import torch.distributed as dist

def init_workers_nccl_file():
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file_dir = '%s/tmp' % os.environ['SCRATCH']
    os.makedirs(sync_file_dir, exist_ok=True)
    sync_file = 'file://%s/pytorch_sync_%s' % (
        sync_file_dir, os.environ['SLURM_JOB_ID'])
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

# Print pytorch version
print('Pytorch version', torch.__version__)

# Configuration
ranks_per_node = 8
shape = 2**17
dtype = torch.float32

# Initialize MPI
rank, n_ranks = init_workers_nccl_file()
local_rank = rank % ranks_per_node

# Allocate a small tensor on every gpu from every rank.
# This is an attempt to force creation of all device contexts.
#for i in range(ranks_per_node):
#    _ = torch.randn(1).to(torch.device('cuda', i))

# Select our gpu
device = torch.device('cuda', local_rank)
print('Rank', rank, 'size', n_ranks, 'device', device, 'count', torch.cuda.device_count())

# Allocate a tensor on the gpu
x = torch.randn(shape, dtype=dtype).to(device)
print('local result:', x.sum())

# Do a broadcast from rank 0
dist.broadcast(x, 0)
print('broadcast result:', x.sum())

# Do an all-reduce
dist.all_reduce(x)
print('allreduce result:', x.sum())
