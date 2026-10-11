"""Complete Torchvision classifiers, with explicit task input/output adapters.

No blocks, widths or hidden classifier layers are clipped to genetic bounds.
Weights are initialized from scratch; constructing a model never downloads files.
"""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F
from ..config import reference_input_size


def is_reference_genome(genome):
    return genome.get('model_kind') == 'reference'


class ScaledTanh(nn.Module):
    def forward(self, x):
        return 1.7159 * torch.tanh((2.0 / 3.0) * x)


class LeNetSubsampling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.activation = ScaledTanh()

    def forward(self, x):
        return self.activation(F.avg_pool2d(x, 2) * self.weight + self.bias)


class LeNetSparseC3(nn.Module):
    # Table I, LeCun et al. (1998): 60 input-map connections, 1,516 parameters.
    connections = ((0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (0, 4, 5), (0, 1, 5),
                   (0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (0, 3, 4, 5),
                   (0, 1, 4, 5), (0, 1, 2, 5), (0, 1, 3, 4), (1, 2, 4, 5),
                   (0, 2, 3, 5), (0, 1, 2, 3, 4, 5))

    def __init__(self):
        super().__init__()
        self.units = nn.ModuleList(nn.Conv2d(len(indices), 1, 5) for indices in self.connections)
        self.activation = ScaledTanh()

    def forward(self, x):
        return self.activation(torch.cat([unit(x[:, indices]) for unit, indices in zip(self.units, self.connections)], dim=1))


class LeNet5(nn.Module):
    """Complete C1/S2/C3/S4/C5/F6 trunk; task head replaces digit-specific RBFs."""
    def __init__(self, channels, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 6, 5)
        self.activation = ScaledTanh()
        self.s2 = LeNetSubsampling(6)
        self.c3 = LeNetSparseC3()
        self.s4 = LeNetSubsampling(16)
        self.c5 = nn.Conv2d(16, 120, 5)
        self.f6 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, classes)

    def forward(self, x):
        x = self.s2(self.activation(self.conv1(x)))
        x = self.s4(self.c3(x))
        x = self.activation(self.c5(x)).flatten(1)
        return self.classifier(self.activation(self.f6(x)))


def _native_model(architecture_id, config):
    if architecture_id == 'lenet5':
        return LeNet5(int(config['num_channels']), int(config['num_classes']))
    from torchvision import models

    options = {'weights': None, 'num_classes': int(config['num_classes'])}
    if architecture_id in {'googlenet', 'inception_v3'}:
        options.update(aux_logits=bool(config['reference_aux_logits']),
                       init_weights=False, transform_input=False)
    model = models.get_model(architecture_id, **options)
    # Replace only the input convolution, keeping its original output width,
    # kernel, stride, padding and all remaining model modules.
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            channels = int(config['num_channels'])
            if module.in_channels != channels:
                parent = model.get_submodule(name.rsplit('.', 1)[0]) if '.' in name else model
                key = name.rsplit('.', 1)[-1]
                parent._modules[key] = nn.Conv2d(
                    channels, module.out_channels, module.kernel_size,
                    module.stride, module.padding, module.dilation, module.groups,
                    module.bias is not None, module.padding_mode,
                )
            break
    dropout = config['reference_dropout_rate']
    if dropout is not None:
        for module in model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout2d)):
                module.p = float(dropout)
    return model


class ReferenceClassifier(nn.Module):
    def __init__(self, genome, config):
        super().__init__()
        if str(config['input_modality']).lower() not in {'image', 'spectrogram'}:
            raise ValueError('Complete 2D reference architectures require spectrogram input')
        self.architecture_id = genome['reference_architecture_id']
        self.model = _native_model(self.architecture_id, config)
        self.input_size = reference_input_size(config, self.architecture_id)
        self.auxiliary_logits = ()

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if tuple(x.shape[-2:]) != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)
        output = self.model(x)
        self.auxiliary_logits = ()
        if isinstance(output, torch.Tensor):
            return output
        self.auxiliary_logits = tuple(value for value in output[1:] if value is not None)
        return output.logits

    def get_architecture_summary(self):
        return f"Complete reference: {self.architecture_id} | input adapter: {self.input_size}"


def classification_loss(model, logits, target, criterion, config):
    loss = criterion(logits, target)
    for auxiliary in getattr(model, 'auxiliary_logits', ()):
        loss = loss + float(config['reference_aux_loss_weight']) * criterion(auxiliary, target)
    if hasattr(model, 'auxiliary_logits'):
        model.auxiliary_logits = ()
    return loss


@lru_cache(maxsize=256)
def _parameter_count(architecture_id, channels, classes, auxiliary):
    # Torchvision initializes Dynamo during import; do so before entering the
    # meta device context (Torch 2.8 otherwise sees a circular initialization).
    from torchvision import models  # noqa: F401
    # Meta tensors retain shapes but allocate no model weights. Counts include
    # the complete hidden classifier and any enabled auxiliary classifiers.
    config = {'num_channels': channels, 'num_classes': classes,
              'reference_aux_logits': auxiliary, 'reference_dropout_rate': None}
    with torch.device('meta'):
        model = _native_model(architecture_id, config)
    return sum(parameter.numel() for parameter in model.parameters())


def reference_parameter_count(genome, config):
    return _parameter_count(genome['reference_architecture_id'], int(config['num_channels']),
                            int(config['num_classes']), bool(config['reference_aux_logits']))


def validate_reference_genome(genome, config):
    from ..config import REFERENCE_ARCHITECTURE_IDS
    if genome.get('reference_architecture_id') not in REFERENCE_ARCHITECTURE_IDS:
        raise ValueError('Unknown complete reference architecture')
    if str(config['input_modality']).lower() not in {'image', 'spectrogram'}:
        raise ValueError('Complete reference models require 2D spectrograms; they are not reduced to Conv1D')
    architecture_id = genome['reference_architecture_id']
    size = reference_input_size(config, architecture_id)
    minimum = 299 if architecture_id == 'inception_v3' and config['reference_aux_logits'] else 75 if architecture_id == 'inception_v3' else 224 if architecture_id == 'googlenet' and config['reference_aux_logits'] else 64
    if architecture_id == 'lenet5':
        if size != (32, 32):
            raise ValueError('Complete LeNet-5 requires reference_lenet_input_size=(32, 32)')
    elif len(size) != 2 or min(size) < minimum:
        raise ValueError(f'Reference input dimensions for {architecture_id} must be >= {minimum}')
    limit = config['reference_max_model_parameters']
    if limit is not None and reference_parameter_count(genome, config) > int(limit):
        raise ValueError('Complete architecture exceeds reference_max_model_parameters; it will not be shrunk')
    return genome
