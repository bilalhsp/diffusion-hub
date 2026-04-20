import time
from diffusion_hub.datasets import get_dataset

import numpy as np
import torch

def subsample_dataset(dataset, fraction=0.5, seed=42):
    """
    Randomly subsample a fraction of the dataset reproducibly.
    Same seed → always same subset.
    """
    rng = np.random.default_rng(seed)        # ← isolated RNG, doesn't affect global state
    
    n_total  = len(dataset)
    n_sample = int(n_total * fraction)
    
    indices = rng.choice(n_total, size=n_sample, replace=False)
    
    return torch.utils.data.Subset(dataset, indices)




# Always produces the same subset ✅


if __name__ == '__main__':

    START = time.time()

    name = 'nz_unlabel'
    config = {
        'root_dir': '/depot/jgmakin/data/NZ0000/NWB/',    
        'data_type': 'both',         # [hg, lfc, both]
        'duration': 30,
        'debug_mode': False,  
        'trial_duration': None,
        'is_train': False
    }

    train_data = get_dataset(name, **config)


    print(f"Number of training samples: {len(train_data)}")
    print(f"Training data duration (hours): {(len(train_data) * train_data.duration)/3600:.2f}")

    print(f"Time taken to load the dataset: {(time.time() - START)/60:.2f} minutes.")

