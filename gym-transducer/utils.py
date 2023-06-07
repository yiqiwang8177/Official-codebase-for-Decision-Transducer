import os
import numpy as np
import torch 
import random

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

def load_critic(env_name, device):
    from critic import ValueCritic
    SAVE = 'saved_model'
    
    if env_name == 'antmaze-medium-diverse' or 'antmaze-umaze':
        load_path = os.path.join('.', '..', '..', SAVE, env_name, 'value_s950000.pth' )
    if env_name == 'antmaze-medium-play':
        # the original 95000 is missing, so ...
        load_path = os.path.join('.', '..', '..', SAVE, env_name, 'value_s900000.pth' )
    if env_name == 'antmaze-umaze-diverse':
        load_path = os.path.join('.', '..', '..', SAVE, env_name, 'value_s1000000.pth' )
    else:
        assert False

    critic = ValueCritic(29, 256, 3)
    critic.load_state_dict(torch.load(load_path, map_location=torch.device(device) ))
    return critic.to(device)

def value_state(critic, states, device):

    with torch.no_grad():
        states = torch.tensor(states).to(device)
        values = critic(states).detach().flatten().cpu().numpy()
    stats = [ min(values), max(values)]
    # assert False, f"{values}"
    return values, stats

def scale_value(large, small, value):
    if value > large:
        return 0.0
    elif value < small:
        return 1.0
    else:
        
        return ( large - value ) / (large - small)

