"""Shared architecture formatting helpers for logs and reports."""


def _fc_label(num_fc_layers: int) -> str:
    """Formats the fully connected layer count compactly."""
    return f"{int(num_fc_layers)}FC"


def format_genome_architecture(genome: dict) -> str:
    """Returns a readable one-line architecture description for logs."""
    num_conv_layers = int(genome['num_conv_layers'])
    num_fc_layers = int(genome['num_fc_layers'])
    fc_label = _fc_label(num_fc_layers)
    conv_label = f"{num_conv_layers}C"

    if genome.get('inception_enabled', False):
        base = f"incep, {conv_label}, {fc_label}"
    elif genome.get('residual_enabled', False):
        base = f"res, {conv_label}, {fc_label}"
    else:
        base = f"seq, {conv_label}, {fc_label}"

    template_id = genome.get('architecture_template_id')
    if template_id:
        return f"{base}, tpl={template_id}"
    return base
