"""Check resources retained by completed folds, including unconsumed results."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from neuroevolution.config import get_default_config
from neuroevolution.evolution import fitness


@pytest.mark.parametrize('device_type', ['cpu', 'cuda'])
def test_completed_folds_keep_only_cpu_models(monkeypatch, device_type):
    if device_type == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA device unavailable')
    device = torch.device(device_type)
    optimizer_refs = []
    snapshot_devices = []
    output_refs = []

    class TinyModel(torch.nn.Linear):
        def __init__(self, *args):
            super().__init__(1, 2)

        def forward(self, data):
            # A validation pass must not retain the previous training output.
            if not self.training:
                assert all(ref() is None for ref in output_refs)
            output = super().forward(data)
            if self.training:
                output_refs.append(weakref.ref(output))
            return output

        def load_state_dict(self, state, *args, **kwargs):
            snapshot_devices.extend(value.device.type for value in state.values())
            # The checkpoint must own a copy, including when training on CPU.
            assert state['weight'].data_ptr() != self.weight.data_ptr()
            return super().load_state_dict(state, *args, **kwargs)

    dataset = torch.utils.data.TensorDataset(
        torch.tensor([[-1.0], [1.0], [-0.5], [0.5]]), torch.tensor([0, 1, 0, 1])
    )

    def loaders(*args, **kwargs):
        return (torch.utils.data.DataLoader(dataset, batch_size=2),
                torch.utils.data.DataLoader(dataset, batch_size=2))

    def make_optimizer(model, *args):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer_refs.append(weakref.ref(optimizer))
        return optimizer

    monkeypatch.setattr(fitness, 'EvolvableCNN', TinyModel)
    monkeypatch.setattr(fitness, 'load_fold_data', loaders)
    monkeypatch.setattr(fitness, 'make_optimizer', make_optimizer)
    config = {**get_default_config(), 'num_epochs': 2, 'use_amp': False,
              'fold_parallel_workers': 1}
    train = fitness.train_fold_on_cuda_stream if device_type == 'cuda' else fitness.train_fold_in_thread
    # Keep all Futures alive to reproduce delayed consumption by the scheduler.
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        for fold in range(1, 4):
            future = executor.submit(train, {'id': 'tiny'}, fold, config, device)
            futures.append(future)
            result_fold, _, model, metrics = future.result(timeout=30)
            assert result_fold == fold
            assert metrics is not None
            assert all(parameter.device.type == 'cpu' and parameter.grad is None
                       for parameter in model.parameters())
            gc.collect()
            assert all(ref() is None for ref in optimizer_refs)
    assert len(futures) == 3
    assert snapshot_devices and set(snapshot_devices) == {'cpu'}
