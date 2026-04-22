"""
Custom logging utilities for notebook execution.

Redirects print() output to log files while maintaining console output.
"""

import builtins
import logging
import os
import subprocess
import sys


def setup_notebook_logging(log_dir: str, log_filename: str = "execution_log.txt") -> logging.Logger:
    """
    Sets up logging to redirect print() to a file.
    
    Args:
        log_dir: Directory to store log files
        log_filename: Name of the log file
    
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("neuroevolution_notebook")
    
    # Clear existing handlers
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(
        os.path.join(log_dir, log_filename),
        mode="a",
        encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    
    # Override print() to write to logger
    def notebook_print(*args, sep=" ", end="\n", **kwargs):
        message = sep.join(str(arg) for arg in args)
        if end not in ("", "\n"):
            message += end
        logger.info(message)
    
    builtins.print = notebook_print
    return logger


def install_package(package: str) -> None:
    """
    Installs a package using pip if not available.
    
    Args:
        package: Package specification (e.g., "torch==2.11.0")
    """
    try:
        __import__(package.split('==')[0].split('[')[0])
        print(f"OK {package.split('==')[0]} is already installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"OK {package} installed correctly")


def verify_dependencies(required_packages: list) -> None:
    """
    Verifies and installs all required dependencies.
    
    Args:
        required_packages: List of package specifications
    """
    print("Starting dependency installation for Hybrid Neuroevolution...")
    print("=" * 60)
    
    for package in required_packages:
        install_package(package)
    
    print("\nAll dependencies have been verified/installed")
    print("Restart the kernel if this is the first time installing torch")
    print("=" * 60)
    
    # Verify PyTorch installation
    try:
        import torch
        print(f"\nPyTorch {torch.__version__} installed correctly")
        print(f"CUDA available: {'Yes' if torch.cuda.is_available() else 'No'}")
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    except ImportError:
        print("ERROR: PyTorch could not be installed correctly")
        print("Try installing manually with: pip install torch torchvision")
