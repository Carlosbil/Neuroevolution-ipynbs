"""
Dataset loading and verification for 5-fold cross-validation.
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
        config: Configuration dictionary (updated in-place with sequence_length)
    
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
    
    # If directory not found, try alternative locations
    if not os.path.exists(fold_files_directory):
        possible_paths = [
            os.path.join('..', 'data', 'sets', 'folds_5', config['fold_files_subdirectory']),
            os.path.join('data', 'sets', 'folds_5', config['fold_files_subdirectory']),
            os.path.join('.', 'data', 'sets', 'folds_5', config['fold_files_subdirectory']),
        ]
        
        print(f"\nSearching for data in alternative locations:")
        for path in possible_paths:
            abs_path = os.path.abspath(path)
            exists = os.path.exists(path)
            print(f"   {'✓' if exists else '✗'} {abs_path}")
            if exists:
                fold_files_directory = path
                print(f"\n✓ Found data at: {os.path.abspath(fold_files_directory)}")
                break
        else:
            raise FileNotFoundError(
                f"\n❌ Could not find data directory!\n"
                f"   Tried paths:\n" + 
                "\n".join([f"      - {os.path.abspath(p)}" for p in possible_paths]) +
                f"\n\n   Please check:\n"
                f"      1. CONFIG['data_path'] is correct\n"
                f"      2. The data files exist\n"
                f"      3. The fold_id '{config['fold_id']}' is correct\n"
            )
    else:
        print(f"   ✓ Directory found: {os.path.abspath(fold_files_directory)}")
    
    dataset_id = config['dataset_id']
    
    # Check that all 5 folds exist
    print(f"\nChecking for all 5 folds...")
    all_folds_ok = True
    
    for fold_num in range(1, 6):
        required_files = [
            f'X_train_{dataset_id}_fold_{fold_num}.npy',
            f'y_train_{dataset_id}_fold_{fold_num}.npy',
            f'X_val_{dataset_id}_fold_{fold_num}.npy',
            f'y_val_{dataset_id}_fold_{fold_num}.npy',
            f'X_test_{dataset_id}_fold_{fold_num}.npy',
            f'y_test_{dataset_id}_fold_{fold_num}.npy',
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
    
    # Update sequence length from actual data
    if len(x_train.shape) == 2:  # (samples, sequence_length)
        config['sequence_length'] = x_train.shape[1]
    elif len(x_train.shape) == 3:  # Already (samples, channels, sequence_length)
        config['sequence_length'] = x_train.shape[2]
    
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
    
    # Try alternative paths if not found
    if not os.path.exists(fold_files_directory):
        possible_paths = [
            os.path.join('..', 'data', 'sets', 'folds_5', config['fold_files_subdirectory']),
            os.path.join('data', 'sets', 'folds_5', config['fold_files_subdirectory']),
            os.path.join('.', 'data', 'sets', 'folds_5', config['fold_files_subdirectory']),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                fold_files_directory = path
                break
    
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
