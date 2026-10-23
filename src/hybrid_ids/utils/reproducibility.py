"""
Reproducibility utilities for deterministic seeds across Python, NumPy, and Tree Boosters.
"""

import os
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets random seeds deterministically across all runtime environments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
