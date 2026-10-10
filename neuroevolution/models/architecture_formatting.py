"""Shared architecture formatting helpers for logs and reports."""


def format_fc_activations(genome: dict, config: dict) -> str:
    """Describe the hidden dense activations selected by the model config."""
    source = config.get('fc_activation_source', 'genome')
    names = [
        genome['activations'][i % len(genome['activations'])]
        if source == 'genome' else source
        for i in range(genome['num_fc_layers'])
    ]
    return '[' + ', '.join(names) + ']'


def _fc_label(num_fc_layers: int) -> str:
    """Formats the fully connected layer count compactly."""
    return f"{int(num_fc_layers)}FC"


def format_genome_architecture(genome: dict) -> str:
    """Returns a readable one-line architecture description for logs."""
    if genome.get('model_kind') == 'reference':
        return f"reference, {genome['reference_architecture_id']}, complete"
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
