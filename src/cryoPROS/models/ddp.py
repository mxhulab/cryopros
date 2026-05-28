import os

def local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))

def rank():
    return int(os.environ.get('RANK', 0))

def world_size():
    return int(os.environ.get('WORLD_SIZE', 1))

def is_distributed():
    return world_size() > 1

def is_main_process():
    return rank() == 0

def initialize_process_group():
    if not is_distributed():
        return

    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError('Require GPU to perform cryopros-train')

    torch.cuda.set_device(local_rank())
    if not dist.is_initialized():
        backend = 'nccl' if dist.is_nccl_available() else 'gloo'
        dist.init_process_group(backend = backend, init_method = 'env://')
