"""
Innovation tracking system for NEAT-like genome evolution.
"""

import uuid
import json


# Fixed UUID namespace for deterministic innovation IDs
INNOVATION_NAMESPACE = uuid.UUID('12345678-1234-5678-1234-567812345678')


def innovation_uuid(event_type: str, payload: dict) -> str:
    """
    Creates a deterministic UUID for structural innovations.
    
    Args:
        event_type: Type of innovation (e.g., 'conv_filter', 'conv_kernel', 'fc_node')
        payload: Dictionary containing innovation details
    
    Returns:
        String UUID for this innovation
    """
    canonical = json.dumps({'event': event_type, 'payload': payload}, sort_keys=True)
    return str(uuid.uuid5(INNOVATION_NAMESPACE, canonical))


def build_innovation_genes(genome: dict) -> list:
    """
    Builds innovation genes used for homologous alignment in crossover.
    
    Args:
        genome: Genome dictionary
    
    Returns:
        List of gene dictionaries with innovation_id, gene_key, value, enabled
    """
    genes = []

    for i, value in enumerate(genome.get('filters', [])):
        genes.append({
            'innovation_id': innovation_uuid('conv_filter', {'index': i, 'value': int(value)}),
            'gene_key': f'conv_filter_{i}',
            'value': int(value),
            'enabled': True
        })

    for i, value in enumerate(genome.get('kernel_sizes', [])):
        genes.append({
            'innovation_id': innovation_uuid('conv_kernel', {'index': i, 'value': int(value)}),
            'gene_key': f'conv_kernel_{i}',
            'value': int(value),
            'enabled': True
        })

    for i, value in enumerate(genome.get('fc_nodes', [])):
        genes.append({
            'innovation_id': innovation_uuid('fc_node', {'index': i, 'value': int(value)}),
            'gene_key': f'fc_node_{i}',
            'value': int(value),
            'enabled': True
        })

    return genes


def append_structural_event(genome: dict, event_type: str, payload: dict):
    """
    Stores structural mutation/crossover events with innovation UUIDs.
    
    Args:
        genome: Genome dictionary (modified in-place)
        event_type: Type of structural event
        payload: Event details
    """
    history = genome.setdefault('structural_history', [])
    history.append({
        'innovation_id': innovation_uuid(event_type, payload),
        'event_type': event_type,
        'payload': payload
    })
