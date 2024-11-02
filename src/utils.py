import functools
import gc
import os
import random
import time
from pathlib import Path

import numpy as np
import torch


def get_current_dir() -> Path:
    try:
        return Path(__file__).parent.absolute()
    except NameError:
        return Path(os.getcwd())


def set_env(seed: int = -1) -> None:
    # reproducibility
    if seed == -1:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # perf
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # (range: 16-512) tune to your needs


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()


def get_device(disable_mps=False) -> str:
    if torch.backends.mps.is_available() and not disable_mps:
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def print_gpu_memory() -> None:
    if torch.cuda.is_available():
        print(f"memory summary: {torch.cuda.memory_summary(device='cuda')}")
        print(f"gpu memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"gpu memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"gpu memory peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"gpu memory peak cached: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")


def timeit(func) -> callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result

    return wrapper
