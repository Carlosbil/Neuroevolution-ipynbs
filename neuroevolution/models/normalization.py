"""Normalization shared by audio and spectrogram classifiers."""

from torch import nn
from torch.nn import functional as F


class SingletonSafeBatchNorm1d(nn.BatchNorm1d):
    """Use running statistics for a singleton without dropping the training sample.

    Ordinary batches retain BatchNorm1d behavior and checkpoint keys. A single
    value per channel cannot estimate variance, so only that forward uses the
    stored statistics. Gradients still reach the input and affine parameters;
    running statistics and num_batches_tracked are left unchanged.
    """

    def forward(self, x):
        self._check_input_dim(x)
        if self.training and x.numel() // x.size(1) == 1 and self.track_running_stats:
            return F.batch_norm(
                x, self.running_mean, self.running_var, self.weight, self.bias,
                training=False, momentum=0.0, eps=self.eps,
            )
        return super().forward(x)
