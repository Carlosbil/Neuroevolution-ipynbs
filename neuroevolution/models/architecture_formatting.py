"""Shared architecture formatting helpers for logs and reports."""


def _fc_label(num_fc_layers: int) -> str:
    """Formats the fully connected layer count compactly."""
    num_fc_layers = int(num_fc_layers)
    if num_fc_layers == 1:
        return "1 fc"
    return f"{num_fc_layers} fc layers"


def format_genome_architecture(genome: dict) -> str:
    """Returns a readable one-line architecture description for logs."""
    num_conv_layers = int(genome['num_conv_layers'])
    num_fc_layers = int(genome['num_fc_layers'])
    fc_label = _fc_label(num_fc_layers)

    if genome.get('inception_enabled', False):
        return (
            "inception Conv1D modules, "
            f"{num_conv_layers} conv units, "
            f"reduction_ratio={genome.get('inception_reduction_ratio', 0.5)}, "
            f"pool_branch={genome.get('inception_pool_branch', True)}, "
            f"{fc_label}"
        )

    if genome.get('residual_enabled', False):
        return (
            "residual Conv1D blocks, "
            f"{num_conv_layers} conv units, "
            f"block_size={genome.get('residual_block_size', 2)}, "
            f"{fc_label}"
        )

    if num_conv_layers == 1:
        conv_label = "1 conv layer"
    else:
        conv_label = f"{num_conv_layers} conv layers"
    return f"sequential Conv1D stack, {conv_label}, {fc_label}"
