"""
Device configuration and seed management utilities.
"""

import torch
import numpy as np
import random


SEED = 42


def setup_device_and_seeds(seed: int = SEED) -> torch.device:
    """
    Configures device (CPU/CUDA) and sets random seeds for reproducibility.
    
    Args:
        seed: Random seed value (default: 42)
    
    Returns:
        torch.device instance (cuda or cpu)
    """
    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Configure device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device configured: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    return device


def get_device() -> torch.device:
    """
    Returns the appropriate device (CUDA if available, else CPU).
    
    Returns:
        torch.device instance
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
