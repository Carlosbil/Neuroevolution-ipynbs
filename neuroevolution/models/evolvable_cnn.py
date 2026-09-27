"""Select the 1D waveform or 2D spectrogram model from the experiment config."""

from .evolvable_cnn_1d import EvolvableCNN as EvolvableCNN1D
from .evolvable_cnn_2d import EvolvableCNN2D


def EvolvableCNN(genome: dict, config: dict):
    """Compatibility entry point used by training, checkpoints and reports."""
    modality = str(config.get('input_modality', 'audio')).lower()
    if modality in {'spectrogram', 'image'}:
        return EvolvableCNN2D(genome, config)
    if modality == 'audio':
        return EvolvableCNN1D(genome, config)
    raise ValueError(f"Unsupported input_modality: {modality}")
