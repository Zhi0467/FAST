import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SINCNET_DIR = os.path.join(os.path.dirname(__file__), "third_party", "SincNet")
if SINCNET_DIR not in sys.path:
    sys.path.append(SINCNET_DIR)

from dnn_models import SincConv_fast  # noqa: E402

try:
    from mamba_ssm import Mamba  # type: ignore
except Exception as exc:  # pragma: no cover
    Mamba = None  # type: ignore
    _MAMBA_IMPORT_ERROR = exc


class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels, sample_rate=250, min_low_hz=50, min_band_hz=50):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.padding = kernel_size // 2

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)
        mel = np.linspace(SincConv_fast.to_mel(low_hz), SincConv_fast.to_mel(high_hz), self.out_channels + 1)
        hz = SincConv_fast.to_hz(mel).astype(np.float32)

        base_low = torch.from_numpy(hz[:-1]).view(1, self.out_channels, 1)
        base_band = torch.from_numpy(np.diff(hz)).view(1, self.out_channels, 1)

        self.low_hz_ = nn.Parameter(base_low.repeat(self.in_channels, 1, 1))
        self.band_hz_ = nn.Parameter(base_band.repeat(self.in_channels, 1, 1))

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        self.register_buffer("window_", window)

        n = (self.kernel_size - 1) / 2.0
        n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        self.register_buffer("n_", n_)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("Expected input shape (B, C, T).")
        bsz, channels, _length = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}.")

        n_ = self.n_.to(x.device)
        window_ = self.window_.to(x.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, :, 0]

        f_times_t_low = torch.matmul(low, n_)
        f_times_t_high = torch.matmul(high, n_)

        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n_ / 2)) * window_
        band_pass_center = 2 * band.view(self.in_channels, self.out_channels, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[2])
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=2)
        band_pass = band_pass / (2 * band).unsqueeze(-1)

        filters = band_pass.view(self.in_channels * self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            x,
            filters,
            stride=1,
            padding=self.padding,
            dilation=1,
            bias=None,
            groups=self.in_channels,
        )


class EEG_SincMamba(nn.Module):
    def __init__(
        self,
        num_electrodes=64,
        num_sinc_filters=8,
        sinc_kernel=65,
        d_model=64,
        num_classes=2,
        backend="mamba",
        transformer_layers=4,
        transformer_heads=None,
        transformer_dropout=0.1,
    ):
        super().__init__()
        self.backend = backend

        self.sinc_block = SincConv(
            out_channels=num_sinc_filters,
            kernel_size=sinc_kernel,
            in_channels=num_electrodes,
        )

        self.bn1 = nn.BatchNorm1d(num_electrodes * num_sinc_filters)

        self.spatial_mix = nn.Sequential(
            nn.Conv1d(
                in_channels=num_electrodes * num_sinc_filters,
                out_channels=d_model,
                kernel_size=1,
            ),
            nn.BatchNorm1d(d_model),
            nn.ELU(),
            nn.AvgPool1d(4),
        )

        if self.backend == "mamba":
            if Mamba is None:  # pragma: no cover
                raise ImportError(
                    "Failed to import mamba_ssm.Mamba. Install mamba-ssm first.\n"
                    f"Original error:\n{_MAMBA_IMPORT_ERROR}"
                )
            self.mamba = Mamba(d_model=d_model, d_state=16, expand=2)
            self.transformer = None
        elif self.backend == "transformer":
            if transformer_heads is None:
                for heads in (8, 4, 2, 1):
                    if d_model % heads == 0:
                        transformer_heads = heads
                        break
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=transformer_heads,
                dim_feedforward=d_model * 4,
                dropout=transformer_dropout,
                batch_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.mamba = None
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.sinc_block(x)
        x = self.bn1(x)
        x = self.spatial_mix(x)
        x = x.permute(0, 2, 1)
        if self.backend == "mamba":
            x = self.mamba(x)
        else:
            x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)
