"""Waveform references preserve their graphs while executing temporal operators."""

import io

import pytest
import torch
from torch import nn

from neuroevolution.config import get_default_config, REFERENCE_ARCHITECTURE_IDS, validate_config
from neuroevolution.evolution.engine import HybridNeuroevolution
from neuroevolution.genetics.architecture_templates import create_template_genome, get_configured_templates
from neuroevolution.models.architecture_formatting import format_genome_architecture
from neuroevolution.models.evolvable_cnn import EvolvableCNN
from neuroevolution.models.genome_validator import estimate_genome_parameter_count, stable_genome_signature


@pytest.mark.parametrize('architecture_id', REFERENCE_ARCHITECTURE_IDS)
def test_every_audio_reference_has_temporal_graph_and_valid_training_shapes(architecture_id):
    config = {**get_default_config(), 'input_modality': 'audio', 'sequence_length': 512}
    genome = create_template_genome(architecture_id, config)
    with torch.device('meta'):
        model = EvolvableCNN(genome, config).train()
        logits = model(torch.empty(2, 1, 512))
    assert logits.shape == (2, 2)
    assert genome['reference_dimensions'] == 1
    assert ', 1D,' in format_genome_architecture(genome)
    assert any(isinstance(layer, nn.Conv1d) for layer in model.modules())
    assert not any(isinstance(layer, (nn.Conv2d, nn.BatchNorm2d, nn.MaxPool2d,
                                      nn.AvgPool2d, nn.AdaptiveAvgPool2d))
                   for layer in model.modules())
    assert estimate_genome_parameter_count(genome, config) == sum(p.numel() for p in model.parameters())
    if architecture_id in {'googlenet', 'inception_v3'}:
        assert model.auxiliary_logits


@pytest.mark.parametrize('architecture_id', [
    'lenet5', 'alexnet', 'vgg11_bn', 'resnet18', 'resnext50_32x4d',
    'googlenet', 'inception_v3', 'densenet121', 'mobilenet_v2',
    'mobilenet_v3_large', 'efficientnet_b0', 'squeezenet1_0',
])
def test_audio_families_forward_with_real_temporal_weights(architecture_id):
    torch.set_num_threads(1)
    config = {**get_default_config(), 'input_modality': 'audio', 'sequence_length': 512}
    model = EvolvableCNN(create_template_genome(architecture_id, config), config).eval()
    seen = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv1d):
            layer.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].ndim))
    with torch.no_grad():
        result = model(torch.randn(1, 1, 512))
    assert result.shape == (1, 2) and torch.isfinite(result).all()
    assert seen and set(seen) == {3}


def test_default_audio_population_contains_googlenet_and_injects_more_references(tmp_path):
    config = {**get_default_config(str(tmp_path)), 'input_modality': 'audio',
              'resume_evolution': False}
    validate_config(config)
    assert [template.template_id for template in get_configured_templates(config)] == list(REFERENCE_ARCHITECTURE_IDS)
    evolution = HybridNeuroevolution(config, torch.device('cpu'))
    evolution.initialize_population()
    references = [g for g in evolution.population if g.get('model_kind') == 'reference']
    assert len(references) == 7
    assert 'googlenet' in {g['reference_architecture_id'] for g in references}
    assert all(g['reference_dimensions'] == 1 for g in references)
    for index, genome in enumerate(evolution.population):
        genome['fitness'] = 50 + index
        genome['metrics'] = {'f1_score': 50 + index}
    evolution.generation = 1
    evolution.selection_and_reproduction()
    injected = [g for g in evolution.population if g.get('architecture_template_origin') == 'initial_seed'
                and not g.get('skip_next_evaluation')]
    assert len(injected) >= 2
    assert all(g['reference_dimensions'] == 1 for g in injected)


def test_audio_checkpoint_roundtrip_and_dimension_identity():
    audio = {**get_default_config(), 'input_modality': 'audio'}
    image = {**audio, 'input_modality': 'spectrogram'}
    genome = create_template_genome('googlenet', audio)
    image_genome = {**genome, 'reference_dimensions': 2}
    assert stable_genome_signature(genome, audio) != stable_genome_signature(image_genome, image)
    model = EvolvableCNN(genome, audio).eval()
    data = torch.randn(1, 1, audio['sequence_length'])
    checkpoint = io.BytesIO()
    torch.save({'genome': genome, 'state': model.state_dict()}, checkpoint)
    checkpoint.seek(0)
    saved = torch.load(checkpoint, weights_only=True)
    restored = EvolvableCNN(saved['genome'], audio).eval()
    restored.load_state_dict(saved['state'])
    with torch.no_grad():
        torch.testing.assert_close(model(data), restored(data))
    with pytest.raises(ValueError, match='dimensions'):
        EvolvableCNN(genome, image)
