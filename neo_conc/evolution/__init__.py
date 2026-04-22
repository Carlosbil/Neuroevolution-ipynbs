"""Evolution package - contains the main neuroevolution engine and fitness evaluation."""

from .engine import HybridNeuroevolution
from .fitness import (
    evaluate_fitness,
    evaluate_population_concurrent,
    train_individual_on_fold,
    load_fold_data,
)

__all__ = [
    'HybridNeuroevolution',
    'evaluate_fitness',
    'evaluate_population_concurrent',
    'train_individual_on_fold',
    'load_fold_data',
]
