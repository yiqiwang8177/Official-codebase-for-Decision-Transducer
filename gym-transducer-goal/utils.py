import os
import numpy as np
import torch 
import random

def set_seed(seed: int = 1) -> None:
    seed *= 1024
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # below option set seed for current gpu
    torch.cuda.manual_seed(seed)
    # below option set seed for ALL GPU
    # torch.cuda.manual_seed_all(seed)

    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed for training set as {seed}")

def set_seed2(seed: int = 1) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # below option set seed for current gpu
    torch.cuda.manual_seed(seed)
    # below option set seed for ALL GPU
    # torch.cuda.manual_seed_all(seed)

    # When running on the CuDNN backend, two further options must be set
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed for training set as {seed}")
