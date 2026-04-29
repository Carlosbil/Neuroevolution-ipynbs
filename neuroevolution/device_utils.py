"""
Device configuration and seed management utilities.
"""

import torch
import numpy as np
import random


SEED = 42


def setup_seeds(seed: int = SEED) -> None:
    """
    Sets random seeds for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def setup_device() -> torch.device:
    """
    Configures and reports the compute device.
    
    Returns:
        torch.device instance (cuda or cpu)
    """
    device = get_device()
    print(f"Device configured: {device}")
    print(f"PyTorch version: {torch.__version__}")
    return device


def setup_device_and_seeds(seed: int = SEED) -> torch.device:
    """
    Configures device (CPU/CUDA) and sets random seeds for reproducibility.

    Args:
        seed: Random seed value (default: 42)

    Returns:
        torch.device instance (cuda or cpu)
    """
    setup_seeds(seed)
    return setup_device()


def get_device() -> torch.device:
    """
    Returns the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device instance
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
