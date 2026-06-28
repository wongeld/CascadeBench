"""
utils/seed.py

Global deterministic seeding for full reproducibility.
Same seed → same experiment result.
"""

import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Set all RNG seeds used throughout the simulation."""
    random.seed(seed)
    np.random.seed(seed)
    # If torch is ever added: torch.manual_seed(seed)
    print(f"[seed] Global seed set to {seed}")


def get_rng(seed: int) -> random.Random:
    """Return a dedicated Random instance with given seed (for isolated use)."""
    rng = random.Random(seed)
    return rng
