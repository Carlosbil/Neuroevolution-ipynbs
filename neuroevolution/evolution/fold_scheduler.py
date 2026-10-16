"""Conservative admission control for folds sharing one CUDA device."""

from dataclasses import dataclass
from collections import deque
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
        self.limit = min(self.maximum, max(1, int(config.get('fold_parallel_workers', 3))))
        self.reserve_gib = float(config.get('fold_vram_reserve_gib', 2.0))
        self.reserve_fraction = float(config.get('fold_vram_reserve_fraction', 0.10))
        self.initial_estimate = float(config.get('fold_vram_initial_fold_gib', 4.0)) * GIB
        self.estimate = self.initial_estimate
        self.observations = deque(maxlen=12)
        self.safety_factor = float(config.get('fold_vram_safety_factor', 1.5))
        self.growth_interval = float(config.get('fold_auto_growth_interval_s', 30.0))
        self.last_growth = time.monotonic()
        self.cooldown_until = 0.0
        try:
            self.baseline = torch.cuda.memory_allocated(device)
        except RuntimeError:
            self.baseline = 0
        self.free = self.total = self.reserve = None
        self.telemetry_warning = False
        self.admission_reserved = 0
        self.allocated = self.reserved = None
        self.block_reason = 'startup'

    def update(self, active, warming=0):
        try:
            self.free, self.total = torch.cuda.mem_get_info(self.device)
            self.reserve = max(self.reserve_gib * GIB, self.total * self.reserve_fraction)
            self.allocated = torch.cuda.memory_allocated(self.device)
        except RuntimeError as error:
            self.free = self.total = self.reserve = None
            self.limit = 1
            self.block_reason = 'telemetry_unavailable'
            self.admission_reserved = 0
            if not self.telemetry_warning:
                print(f"      VRAM telemetry unavailable; using one fold: {error}")
                self.telemetry_warning = True
            return
        try:
            self.reserved = torch.cuda.memory_reserved(self.device)
        except (RuntimeError, AssertionError):
            self.reserved = None
        now = time.monotonic()
        if active == 0:
            self.baseline = self.allocated
        elif active > warming:
            observed = max(0, self.allocated - self.baseline) / active
            self.observations.append((now, observed))
        # A large previous architecture must not constrain all later small ones.
        recent = [value for stamp, value in self.observations if now - stamp <= 60]
        self.estimate = max(self.initial_estimate, max(recent, default=0) * self.safety_factor)
        # Workers may still be loading data and not yet have allocated tensors.
        self.admission_reserved = warming * self.estimate
        if self.free < self.reserve:
            self.limit = max(1, min(self.limit, active - 1))
            self.last_growth = now
            self.block_reason = 'low_vram'

    def reserve_slot(self):
        """Budget submissions whose allocations are not visible to CUDA yet."""
        self.admission_reserved += self.estimate

    def admit(self, active):
        # Always allow an isolated fold to prove whether it fits. This also
        # prevents a deadlock when the conservative estimate exceeds its cost.
        if active == 0:
            self.block_reason = 'isolated_fold'
            return True
        if self.free is None:
            self.block_reason = 'telemetry_unavailable'
            return False
        if self.free - self.admission_reserved < self.reserve + self.estimate:
            self.block_reason = 'memory_headroom'
            return False
        if active < self.limit:
            self.block_reason = 'filling_workers'
            return True
        now = time.monotonic()
        if now < self.cooldown_until:
            self.block_reason = 'oom_cooldown'
            return False
        if self.limit < self.maximum and now - self.last_growth >= self.growth_interval:
            self.limit += 1
            self.last_growth = now
            self.block_reason = 'growing'
            print(f"      VRAM admission: concurrency={self.limit}/{self.maximum}, "
                  f"free={self.free / GIB:.2f} GiB, reserve={self.reserve / GIB:.2f} GiB, "
                  f"estimated fold={self.estimate / GIB:.2f} GiB")
            return True
        self.block_reason = 'maximum_workers' if self.limit >= self.maximum else 'growth_interval'
        return False

    def backoff(self):
        self.limit = max(1, self.limit // 2)
        self.last_growth = time.monotonic()
        self.cooldown_until = self.last_growth + max(30.0, 2 * self.growth_interval)
        self.block_reason = 'oom_cooldown'
        print(f"      CUDA memory backoff: concurrency reduced to {self.limit}; "
              "growth resumes after cooldown if VRAM permits.")

    def snapshot(self):
        return {'mode': 'auto', 'worker_limit': self.limit, 'maximum_workers': self.maximum,
                'free_vram_bytes': self.free, 'total_vram_bytes': self.total,
                'reserve_vram_bytes': self.reserve,
                'estimated_fold_vram_bytes': self.estimate,
                'allocated_vram_bytes': self.allocated, 'reserved_vram_bytes': self.reserved,
                'startup_reserved_bytes': self.admission_reserved,
                'block_reason': self.block_reason,
                'oom_cooldown_seconds': max(0, self.cooldown_until - time.monotonic())}
