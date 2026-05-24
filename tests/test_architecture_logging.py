import torch

from neuroevolution.config import get_default_config
from neuroevolution.evaluation.cross_validation import _format_architecture as format_cv_architecture
from neuroevolution.evolution.engine import HybridNeuroevolution
from neuroevolution.visualization.reports import _format_architecture as format_report_architecture


def small_config(tmp_path):
    config = get_default_config()
    config.update(
        {
            "artifacts_dir": str(tmp_path),
            "sequence_length": 32,
            "num_channels": 1,
            "num_classes": 2,
            "min_conv_layers": 1,
            "max_conv_layers": 8,
            "min_fc_layers": 1,
            "max_fc_layers": 3,
        }
    )
    return config


def test_residual_architecture_log_describes_conv_units_inside_blocks(tmp_path):
    genome = {
        "num_conv_layers": 7,
        "num_fc_layers": 1,
        "residual_enabled": True,
        "residual_block_size": 2,
        "inception_enabled": False,
    }
    expected = "residual Conv1D blocks, 7 conv units, block_size=2, 1 fc"
    engine = HybridNeuroevolution(small_config(tmp_path), torch.device("cpu"))

    assert engine._format_architecture(genome) == expected
    assert format_cv_architecture(genome) == expected
    assert format_report_architecture(genome) == expected


def test_inception_architecture_log_describes_modules_and_branch_options(tmp_path):
    genome = {
        "num_conv_layers": 7,
        "num_fc_layers": 1,
        "residual_enabled": False,
        "inception_enabled": True,
        "inception_reduction_ratio": 0.5,
        "inception_pool_branch": True,
    }
    expected = "inception Conv1D modules, 7 conv units, reduction_ratio=0.5, pool_branch=True, 1 fc"
    engine = HybridNeuroevolution(small_config(tmp_path), torch.device("cpu"))

    assert engine._format_architecture(genome) == expected
    assert format_cv_architecture(genome) == expected
    assert format_report_architecture(genome) == expected
