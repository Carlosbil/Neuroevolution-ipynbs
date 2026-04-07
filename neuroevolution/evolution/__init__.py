"""Evolution package - contains the main neuroevolution engine and fitness evaluation."""

from .engine import HybridNeuroevolution
from .fitness import evaluate_fitness, train_fold_in_thread, load_fold_data

__all__ = [
    'HybridNeuroevolution',
    'evaluate_fitness',
    'train_fold_in_thread',
    'load_fold_data'
]
