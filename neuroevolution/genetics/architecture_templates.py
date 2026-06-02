"""Known architecture templates adapted to the searchable Conv1D genome."""

from __future__ import annotations

import copy
import random
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from neuroevolution.genetics.innovation import append_structural_event, build_innovation_genes
from neuroevolution.models.genome_validator import is_genome_valid, validate_and_fix_genome


RANDOM_TEMPLATE_ORIGIN = "random"
INITIAL_TEMPLATE_ORIGIN = "initial_seed"
MUTATION_TEMPLATE_ORIGIN = "mutation_module"


@dataclass(frozen=True)
class ArchitectureTemplate:
    """Descriptor for a genome-level architecture template."""

    template_id: str
    template_family: str
    conv_topology: str
    num_conv_layers: int
    filters: tuple
    kernel_sizes: tuple
    num_fc_layers: int
    fc_nodes: tuple
    activations: tuple = ("relu",)
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    optimizer: str = "adam"
    normalization_type: str = "batch"
    residual_block_size: int = 2
    residual_projection: str = "auto"
    inception_reduction_ratio: float = 0.5
    inception_pool_branch: bool = True


_TEMPLATES: Dict[str, ArchitectureTemplate] = {
    "resnet_conv1d_small": ArchitectureTemplate(
        template_id="resnet_conv1d_small",
        template_family="resnet",
        conv_topology="residual",
        num_conv_layers=4,
        filters=(8, 8, 16, 16),
        kernel_sizes=(3, 3, 3, 3),
        num_fc_layers=1,
        fc_nodes=(16,),
        residual_block_size=2,
        residual_projection="auto",
    ),
    "googlenet_inception_conv1d_small": ArchitectureTemplate(
        template_id="googlenet_inception_conv1d_small",
        template_family="googlenet",
        conv_topology="inception",
        num_conv_layers=3,
        filters=(8, 12, 16),
        kernel_sizes=(5, 7, 7),
        num_fc_layers=1,
        fc_nodes=(16,),
        inception_reduction_ratio=0.5,
        inception_pool_branch=True,
    ),
}


DEFAULT_TEMPLATE_IDS = tuple(_TEMPLATES.keys())


def get_template_registry() -> Dict[str, ArchitectureTemplate]:
    """Returns the known architecture template registry."""
    return dict(_TEMPLATES)


def get_template(template_id: str) -> ArchitectureTemplate:
    """Returns one architecture template or raises a clear validation error."""
    try:
        return _TEMPLATES[str(template_id)]
    except KeyError as exc:
        raise ValueError(f"unknown architecture template: {template_id}") from exc


def get_configured_templates(config: dict) -> List[ArchitectureTemplate]:
    """Returns configured templates in configured order after validation."""
    template_ids = config.get("architecture_template_ids", DEFAULT_TEMPLATE_IDS)
    return [get_template(template_id) for template_id in template_ids]


def _cap_from_config(config: dict, current_key: str, maximum_key: str) -> int:
    return int(config.get(current_key, config[maximum_key]))


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(int(minimum), min(int(maximum), int(value)))


def _repeat_to_length(values: Iterable, length: int) -> list:
    values = list(values)
    if not values:
        return []
    result = list(values[:length])
    while len(result) < length:
        result.append(values[len(result) % len(values)])
    return result


def _choose_supported(value, options: Iterable):
    options = list(options)
    if not options:
        return value
    if value in options:
        return value
    return options[0]


def _template_conv_fields(template: ArchitectureTemplate, config: dict) -> tuple:
    conv_cap = _cap_from_config(config, "current_max_conv_layers", "max_conv_layers")
    num_conv_layers = _clamp(template.num_conv_layers, config["min_conv_layers"], conv_cap)
    filters = [
        _clamp(value, config["min_filters"], config["max_filters"])
        for value in _repeat_to_length(template.filters, num_conv_layers)
    ]
    kernel_options = list(config.get("kernel_size_options", []))
    kernel_sizes = [
        int(_choose_supported(int(value), kernel_options))
        for value in _repeat_to_length(template.kernel_sizes, num_conv_layers)
    ]
    return num_conv_layers, filters, kernel_sizes


def _template_fc_fields(template: ArchitectureTemplate, config: dict) -> tuple:
    fc_cap = _cap_from_config(config, "current_max_fc_layers", "max_fc_layers")
    num_fc_layers = _clamp(template.num_fc_layers, config["min_fc_layers"], fc_cap)
    fc_nodes = [
        _clamp(value, config["min_fc_nodes"], config["max_fc_nodes"])
        for value in _repeat_to_length(template.fc_nodes, num_fc_layers)
    ]
    return num_fc_layers, fc_nodes


def _template_base_genome(template: ArchitectureTemplate, config: dict) -> dict:
    num_conv_layers, filters, kernel_sizes = _template_conv_fields(template, config)
    num_fc_layers, fc_nodes = _template_fc_fields(template, config)
    activations = _repeat_to_length(
        template.activations,
        max(num_conv_layers, num_fc_layers),
    )
    learning_rate = _choose_supported(
        template.learning_rate,
        config.get("learning_rate_options", [template.learning_rate]),
    )
    residual_enabled = template.conv_topology == "residual"
    inception_enabled = template.conv_topology == "inception"

    return {
        "num_conv_layers": num_conv_layers,
        "num_fc_layers": num_fc_layers,
        "filters": filters,
        "kernel_sizes": kernel_sizes,
        "fc_nodes": fc_nodes,
        "activations": activations,
        "dropout_rate": float(template.dropout_rate),
        "learning_rate": learning_rate,
        "optimizer": template.optimizer,
        "normalization_type": template.normalization_type,
        "conv_topology": template.conv_topology,
        "residual_enabled": residual_enabled,
        "residual_block_size": template.residual_block_size,
        "residual_projection": template.residual_projection,
        "inception_enabled": inception_enabled,
        "inception_reduction_ratio": template.inception_reduction_ratio,
        "inception_pool_branch": template.inception_pool_branch,
        "fitness": 0.0,
        "id": str(uuid.uuid4())[:8],
        "structural_history": [],
    }


def _set_template_provenance(genome: dict, template: ArchitectureTemplate, origin: str) -> None:
    genome["architecture_template_id"] = template.template_id
    genome["architecture_template_family"] = template.template_family
    genome["architecture_template_origin"] = origin


def _clear_template_provenance(genome: dict) -> None:
    for key in (
        "architecture_template_id",
        "architecture_template_family",
        "architecture_template_origin",
    ):
        genome.pop(key, None)


def normalize_template_provenance(genome: dict) -> dict:
    """Normalizes optional template provenance fields in-place."""
    origin = genome.get("architecture_template_origin")
    template_id = genome.get("architecture_template_id")
    template_family = genome.get("architecture_template_family")

    if not template_id and not template_family and origin in (None, RANDOM_TEMPLATE_ORIGIN):
        _clear_template_provenance(genome)
        return genome

    if origin not in {INITIAL_TEMPLATE_ORIGIN, MUTATION_TEMPLATE_ORIGIN}:
        genome["architecture_template_origin"] = RANDOM_TEMPLATE_ORIGIN if origin else origin

    return genome


def create_template_genome(
    template_id: str,
    config: dict,
    origin: str = INITIAL_TEMPLATE_ORIGIN,
    event_payload: Optional[dict] = None,
) -> dict:
    """Creates a valid genome from a known architecture template."""
    template = get_template(template_id)
    genome = _template_base_genome(template, config)
    _set_template_provenance(genome, template, origin)
    genome = validate_and_fix_genome(genome, config)
    if not is_genome_valid(genome, config):
        raise ValueError(f"architecture template produced invalid genome: {template_id}")

    payload = {
        "template_id": template.template_id,
        "template_family": template.template_family,
        "origin": origin,
    }
    if event_payload:
        payload.update(event_payload)
    event_type = (
        "architecture_template_module"
        if origin == MUTATION_TEMPLATE_ORIGIN
        else "architecture_template_seed"
    )
    append_structural_event(genome, event_type, payload)
    genome["innovation_genes"] = build_innovation_genes(genome)
    return genome


def choose_template(config: dict) -> ArchitectureTemplate:
    """Chooses one configured template for stochastic seeding or mutation."""
    templates = get_configured_templates(config)
    if not templates:
        raise ValueError("architecture_template_ids must contain at least one template")
    return random.choice(templates)


def apply_template_module_to_genome(genome: dict, config: dict, template_id: Optional[str] = None) -> dict:
    """Applies a bounded template-inspired topology and Conv1D segment mutation."""
    template = get_template(template_id) if template_id else choose_template(config)
    mutated = copy.deepcopy(genome)
    for cached_key in ("skip_next_evaluation", "cached_from_generation", "metrics", "evaluation_status"):
        mutated.pop(cached_key, None)

    num_conv_layers = int(mutated.get("num_conv_layers", template.num_conv_layers))
    num_conv_layers = _clamp(
        num_conv_layers,
        config["min_conv_layers"],
        _cap_from_config(config, "max_conv_layers", "max_conv_layers"),
    )
    template_filters = [
        _clamp(value, config["min_filters"], config["max_filters"])
        for value in _repeat_to_length(template.filters, num_conv_layers)
    ]
    template_kernels = [
        int(_choose_supported(value, config.get("kernel_size_options", [])))
        for value in _repeat_to_length(template.kernel_sizes, num_conv_layers)
    ]

    if num_conv_layers > 0:
        segment_length = min(num_conv_layers, max(1, len(template.filters)))
        start = 0 if num_conv_layers == segment_length else random.randint(0, num_conv_layers - segment_length)
        for offset in range(segment_length):
            index = start + offset
            mutated["filters"][index] = template_filters[index]
            mutated["kernel_sizes"][index] = template_kernels[index]
    else:
        start = 0
        segment_length = 0

    mutated["conv_topology"] = template.conv_topology
    mutated["residual_enabled"] = template.conv_topology == "residual"
    mutated["residual_block_size"] = template.residual_block_size
    mutated["residual_projection"] = template.residual_projection
    mutated["inception_enabled"] = template.conv_topology == "inception"
    mutated["inception_reduction_ratio"] = template.inception_reduction_ratio
    mutated["inception_pool_branch"] = template.inception_pool_branch
    mutated["id"] = str(uuid.uuid4())[:8]
    mutated["fitness"] = 0.0
    _set_template_provenance(mutated, template, MUTATION_TEMPLATE_ORIGIN)

    mutated = validate_and_fix_genome(mutated, config)
    if not is_genome_valid(mutated, config):
        raise ValueError(f"architecture template mutation produced invalid genome: {template.template_id}")

    append_structural_event(
        mutated,
        "architecture_template_module",
        {
            "template_id": template.template_id,
            "template_family": template.template_family,
            "origin": MUTATION_TEMPLATE_ORIGIN,
            "segment_start": int(start),
            "segment_length": int(segment_length),
            "conv_topology": template.conv_topology,
        },
    )
    mutated["innovation_genes"] = build_innovation_genes(mutated)
    return mutated
