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
