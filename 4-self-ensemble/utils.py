import functools
import gc
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def get_current_dir() -> Path:
    try:
        return Path(__file__).parent.absolute()
    except NameError:
        return Path(os.getcwd())


def timeit(func) -> callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} executed in {end - start:.2f}s")
        return result

    return wrapper


def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


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


def set_env(seed: int = -1) -> None:
    if seed == -1:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        # perf
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # (range: 16-512)


@contextmanager
def isolated_environment():
    np_random_state = np.random.get_state()
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state()
    cuda_random_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_print_options = np.get_printoptions()
    try:
        yield  # execute, then restore the saved state of random seeds and numpy precision
    finally:
        np.random.set_state(np_random_state)
        random.setstate(python_random_state)
        torch.set_rng_state(torch_random_state)
        if cuda_random_state:
            torch.cuda.set_rng_state_all(cuda_random_state)
        np.set_printoptions(**numpy_print_options)
