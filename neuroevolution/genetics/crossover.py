"""
Crossover operators for genome recombination.
"""

import random
import copy
import uuid
from typing import Tuple
from neuroevolution.models.genome_validator import is_genome_valid, validate_and_fix_genome
from neuroevolution.genetics.innovation import build_innovation_genes, append_structural_event


def _innovation_aligned_child(dominant_parent: dict, other_parent: dict, config: dict) -> dict:
    """
    Builds one child by aligning homologous genes using innovation_id.
    
    Args:
        dominant_parent: Parent with higher fitness (contributes all disjoint/excess genes)
        other_parent: Parent with lower fitness (contributes some excess genes)
        config: Configuration dictionary
    
    Returns:
        Child genome
    """
    dominant = copy.deepcopy(dominant_parent)
    other = copy.deepcopy(other_parent)

    if 'innovation_genes' not in dominant:
        dominant['innovation_genes'] = build_innovation_genes(dominant)
    if 'innovation_genes' not in other:
        other['innovation_genes'] = build_innovation_genes(other)

    dom_by_id = {g['innovation_id']: g for g in dominant['innovation_genes']}
    other_by_id = {g['innovation_id']: g for g in other['innovation_genes']}

    merged_genes = []
    all_ids = sorted(set(dom_by_id.keys()) | set(other_by_id.keys()))
    for innovation_id in all_ids:
        dom_gene = dom_by_id.get(innovation_id)
        other_gene = other_by_id.get(innovation_id)

        if dom_gene and other_gene:
            chosen = dom_gene if random.random() < 0.5 else other_gene
            merged_genes.append(copy.deepcopy(chosen))
        elif dom_gene:
            merged_genes.append(copy.deepcopy(dom_gene))
        elif other_gene and random.random() < 0.2:
            merged_genes.append(copy.deepcopy(other_gene))

    child = copy.deepcopy(dominant)

    conv_filters = []
    conv_kernels = []
    fc_nodes = []

    for gene in merged_genes:
        parts = gene['gene_key'].split('_')
        if len(parts) < 3:
            continue
        idx = int(parts[-1])
        key_prefix = '_'.join(parts[:2])

        if key_prefix == 'conv_filter':
            conv_filters.append((idx, int(gene['value'])))
        elif key_prefix == 'conv_kernel':
            conv_kernels.append((idx, int(gene['value'])))
        elif key_prefix == 'fc_node':
            fc_nodes.append((idx, int(gene['value'])))

    conv_filters = [v for _, v in sorted(conv_filters, key=lambda x: x[0])]
    conv_kernels = [v for _, v in sorted(conv_kernels, key=lambda x: x[0])]
    fc_nodes = [v for _, v in sorted(fc_nodes, key=lambda x: x[0])]

    if conv_filters:
        child['filters'] = conv_filters
    if conv_kernels:
        child['kernel_sizes'] = conv_kernels
    if fc_nodes:
        child['fc_nodes'] = fc_nodes

    child['num_conv_layers'] = min(len(child.get('filters', [])), len(child.get('kernel_sizes', [])))
    child['num_fc_layers'] = len(child.get('fc_nodes', []))

    # Fallback to minimal architecture if gene merge became too sparse
    child['num_conv_layers'] = max(config['min_conv_layers'], child['num_conv_layers'])
    child['num_fc_layers'] = max(config['min_fc_layers'], child['num_fc_layers'])

    # Enforce incremental caps
    child['num_conv_layers'] = min(child['num_conv_layers'], config.get('current_max_conv_layers', config['max_conv_layers']))
    child['num_fc_layers'] = min(child['num_fc_layers'], config.get('current_max_fc_layers', config['max_fc_layers']))

    child = validate_and_fix_genome(child, config)
    child['innovation_genes'] = build_innovation_genes(child)
    child['id'] = str(uuid.uuid4())[:8]
    child['fitness'] = 0.0

    append_structural_event(
        child,
        'innovation_crossover',
        {
            'dominant_parent': dominant_parent.get('id', 'unknown'),
            'other_parent': other_parent.get('id', 'unknown'),
            'num_merged_genes': len(merged_genes)
        }
    )
    return child


def crossover_genomes(parent1: dict, parent2: dict, config: dict) -> Tuple[dict, dict]:
    """
    Performs innovation-aware crossover between two genomes.
    Homologous alignment is done by innovation UUIDs.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome
        config: Configuration dictionary
    
    Returns:
        Tuple of (child1, child2)
    """
    if random.random() > config['crossover_rate']:
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
        child1['id'] = str(uuid.uuid4())[:8]
        child2['id'] = str(uuid.uuid4())[:8]
        child1['fitness'] = 0.0
        child2['fitness'] = 0.0
        child1['innovation_genes'] = build_innovation_genes(child1)
        child2['innovation_genes'] = build_innovation_genes(child2)
        return child1, child2

    # Fitter parent acts as dominant donor for each child direction
    if parent2.get('fitness', 0.0) > parent1.get('fitness', 0.0):
        parent1, parent2 = parent2, parent1

    max_attempts = 20
    for attempt in range(max_attempts):
        child1 = _innovation_aligned_child(parent1, parent2, config)
        child2 = _innovation_aligned_child(parent2, parent1, config)

        if is_genome_valid(child1, config) and is_genome_valid(child2, config):
            return child1, child2

    print(f"⚠️ Warning: Could not create valid crossover after {max_attempts} attempts. Returning parent copies.")
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    child1['id'] = str(uuid.uuid4())[:8]
    child2['id'] = str(uuid.uuid4())[:8]
    child1['fitness'] = 0.0
    child2['fitness'] = 0.0
    child1 = validate_and_fix_genome(child1, config)
    child2 = validate_and_fix_genome(child2, config)
    child1['innovation_genes'] = build_innovation_genes(child1)
    child2['innovation_genes'] = build_innovation_genes(child2)

    return child1, child2
