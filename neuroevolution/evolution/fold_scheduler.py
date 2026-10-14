"""Conservative admission control for folds sharing one CUDA device."""

from dataclasses import dataclass
import time

import torch


GIB = 1024 ** 3


def is_cuda_memory_error(error):
    """NVML can mask an allocation failure while PyTorch constructs its OOM report."""
    message = str(error).lower()
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and (
            'cuda out of memory' in message or
            ('nvml_success == r' in message and 'cudacachingallocator' in message)
        )
    )


@dataclass(frozen=True)
class FoldMemoryFailure:
    # Store text only: retaining an exception traceback also retains GPU tensors.
    message: str


class VramFoldScheduler:
    """Ramp up slowly, retain headroom, and back off without cancelling training."""

    def __init__(self, config, device):
        self.device = device
        self.maximum = int(config.get('fold_auto_max_workers', 10))
        self.limit = 1
        self.reserve_gib = float(config.get('fold_vram_reserve_gib', 2.0))
        self.reserve_fraction = float(config.get('fold_vram_reserve_fraction', 0.10))
        self.estimate = float(config.get('fold_vram_initial_fold_gib', 4.0)) * GIB
        self.safety_factor = float(config.get('fold_vram_safety_factor', 1.5))
        self.growth_interval = float(config.get('fold_auto_growth_interval_s', 30.0))
        self.last_growth = time.monotonic()
        try:
            self.baseline = torch.cuda.memory_allocated(device)
        except RuntimeError:
            self.baseline = 0
        self.free = self.total = self.reserve = None
        self.telemetry_warning = False

    def update(self, active):
        try:
            self.free, self.total = torch.cuda.mem_get_info(self.device)
            self.reserve = max(self.reserve_gib * GIB, self.total * self.reserve_fraction)
            allocated = torch.cuda.memory_allocated(self.device)
        except RuntimeError as error:
            self.free = self.total = self.reserve = None
            self.limit = 1
            if not self.telemetry_warning:
                print(f"      VRAM telemetry unavailable; using one fold: {error}")
                self.telemetry_warning = True
            return
        if active:
            observed = max(0, allocated - self.baseline) / active
            self.estimate = max(self.estimate, observed * self.safety_factor)
        if self.free < self.reserve:
            self.limit = max(1, min(self.limit, active - 1))
            self.last_growth = time.monotonic()

    def admit(self, active):
        # Always allow an isolated fold to prove whether it fits. This also
        # prevents a deadlock when the conservative estimate exceeds its cost.
        if active == 0:
            return True
        if self.free is None or self.free < self.reserve + self.estimate:
            return False
        if active < self.limit:
            return True
        now = time.monotonic()
        if self.limit < self.maximum and now - self.last_growth >= self.growth_interval:
            self.limit += 1
            self.last_growth = now
            print(f"      VRAM admission: concurrency={self.limit}/{self.maximum}, "
                  f"free={self.free / GIB:.2f} GiB, reserve={self.reserve / GIB:.2f} GiB, "
                  f"estimated fold={self.estimate / GIB:.2f} GiB")
            return True
        return False

    def backoff(self):
        self.limit = max(1, self.limit // 2)
        self.maximum = min(self.maximum, self.limit)
        self.last_growth = time.monotonic()
        print(f"      CUDA memory backoff: concurrency capped at {self.limit} "
              "for the rest of this generation.")

    def snapshot(self):
        return {'mode': 'auto', 'worker_limit': self.limit, 'maximum_workers': self.maximum,
                'free_vram_bytes': self.free, 'total_vram_bytes': self.total,
                'reserve_vram_bytes': self.reserve,
                'estimated_fold_vram_bytes': self.estimate}
