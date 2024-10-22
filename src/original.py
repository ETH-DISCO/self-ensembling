import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import random
import hashlib
import time
import copy

import matplotlib.pyplot as plt

from contextlib import contextmanager
from tqdm import tqdm


@contextmanager
def isolated_environment():
    # save and restore random states in a context manager
    # used to seperate 
    np_random_state = np.random.get_state()
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state()
    cuda_random_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_print_options = np.get_printoptions()
    try:
        yield # execute code block
    finally:
        np.random.set_state(np_random_state)
        random.setstate(python_random_state)
        torch.set_rng_state(torch_random_state)
        if cuda_random_state:
            torch.cuda.set_rng_state_all(cuda_random_state)
        np.set_printoptions(**numpy_print_options)

