"""
Dataset loading and verification for the 5-fold train/validation/test protocol.
"""

import os
import numpy as np
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
import torch


def load_dataset(config: dict) -> None:
    """
    Verifies that dataset files exist and loads first fold to detect sequence_length.
    During evolution, each individual will load all folds automatically.
    
    Args:
        config: Authoritative configuration dictionary, never modified
    
    Raises:
        FileNotFoundError: If data directory or files are missing
    """
    print("\n" + "="*60)
    print("VERIFICANDO DISPONIBILIDAD DE DATOS")
    print("="*60)
    print(f"Dataset ID: {config['dataset_id']}, Verificando los 5 folds...")
    
    # Build directory path using the configured subdirectory
    fold_files_directory = os.path.join(
        config['data_path'], 
        config['fold_files_subdirectory']
    )
    
    print(f"   Looking for: {os.path.abspath(fold_files_directory)}")
    
    if not os.path.isdir(fold_files_directory):
        raise FileNotFoundError(f"Configured data directory is missing: {fold_files_directory}; update data_path in config.py")
    print(f"   Directory found: {os.path.abspath(fold_files_directory)}")

    dataset_id = config['dataset_id']
    
    # Check that all 5 folds exist
    print(f"\nChecking for all 5 folds...")
    all_folds_ok = True
    
    for fold_num in range(1, int(config['num_folds']) + 1):
        required_files = [
            f'X_train_{dataset_id}_fold_{fold_num}.npy',
            f'y_train_{dataset_id}_fold_{fold_num}.npy',
            f'X_val_{dataset_id}_fold_{fold_num}.npy',
            f'y_val_{dataset_id}_fold_{fold_num}.npy',
        ]
        
        fold_ok = True
        for filename in required_files:
            filepath = os.path.join(fold_files_directory, filename)
            if not os.path.exists(filepath):
                fold_ok = False
                all_folds_ok = False
                print(f"   ✗ Fold {fold_num}: Missing {filename}")
                break
        
        if fold_ok:
            print(f"   ✓ Fold {fold_num}: All files present")
    
    if not all_folds_ok:
        raise FileNotFoundError(
            f"\n❌ Some fold files are missing!\n"
            f"   Please ensure all 5 folds have complete data files.\n"
            f"   dataset_id: '{dataset_id}'\n"
        )
    
    print(f"\n✓ All 5 folds verified successfully!")
    
    # Load first fold to detect sequence_length
    print(f"\nLoading Fold 1 to detect sequence length...")
    x_train = np.load(os.path.join(fold_files_directory, f'X_train_{dataset_id}_fold_1.npy'))
    
    print(f"   Train samples: {x_train.shape}")
    
    # Configuration is authoritative: reject mismatches rather than rewriting it.
    detected_length = x_train.shape[-1]
    if int(config['sequence_length']) != int(detected_length):
        raise ValueError(f"sequence_length={config['sequence_length']} differs from arrays ({detected_length}); update config.py")

    print(f"   Sequence length detected: {config['sequence_length']}")
    print(f"\n✓ Dataset verification complete!")
    print(f"   During evolution, each individual will train on all 5 folds.")
    print("="*60)


def load_fold_data(config: dict, fold_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads data for a specific fold.
    
    Args:
        config: Configuration dictionary
        fold_num: Fold number (1-5)
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    
    Raises:
        FileNotFoundError: If fold files are missing
    """
    # Build directory path
    fold_files_directory = os.path.join(
        config['data_path'], 
        config['fold_files_subdirectory']
    )
    
    dataset_id = config['dataset_id']
    
    # Load all splits for this fold
    X_train = np.load(os.path.join(fold_files_directory, f'X_train_{dataset_id}_fold_{fold_num}.npy'))
    y_train = np.load(os.path.join(fold_files_directory, f'y_train_{dataset_id}_fold_{fold_num}.npy'))
    X_val = np.load(os.path.join(fold_files_directory, f'X_val_{dataset_id}_fold_{fold_num}.npy'))
    y_val = np.load(os.path.join(fold_files_directory, f'y_val_{dataset_id}_fold_{fold_num}.npy'))
    X_test = np.load(os.path.join(fold_files_directory, f'X_test_{dataset_id}_fold_{fold_num}.npy'))
    y_test = np.load(os.path.join(fold_files_directory, f'y_test_{dataset_id}_fold_{fold_num}.npy'))
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def verify_fold_files(config: dict, fold_num: int) -> bool:
    """
    Checks if all files for a specific fold exist.
    
    Args:
        config: Configuration dictionary
        fold_num: Fold number (1-5)
    
    Returns:
        True if all files exist, False otherwise
    """
    fold_files_directory = os.path.join(
        config['data_path'], 
        config['fold_files_subdirectory']
    )
    
    if not os.path.exists(fold_files_directory):
        return False
    
    dataset_id = config['dataset_id']
    required_files = [
        f'X_train_{dataset_id}_fold_{fold_num}.npy',
        f'y_train_{dataset_id}_fold_{fold_num}.npy',
        f'X_val_{dataset_id}_fold_{fold_num}.npy',
        f'y_val_{dataset_id}_fold_{fold_num}.npy',
        f'X_test_{dataset_id}_fold_{fold_num}.npy',
        f'y_test_{dataset_id}_fold_{fold_num}.npy',
    ]
    
    for filename in required_files:
        filepath = os.path.join(fold_files_directory, filename)
        if not os.path.exists(filepath):
            return False
    
    return True
