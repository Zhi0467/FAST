import torch
import torch.nn as nn

from third_party.SincNet.dnn_models import SincNet

try:
    from mamba_ssm import Mamba  # type: ignore
except Exception as exc:  # pragma: no cover
    Mamba = None  # type: ignore
    _MAMBA_IMPORT_ERROR = exc


def _build_sincnet_options(input_len, sample_rate, num_sinc_filters, sinc_kernel):
    return {
        "cnn_N_filt": [num_sinc_filters, num_sinc_filters, num_sinc_filters],
        "cnn_len_filt": [sinc_kernel, 5, 5],
        "cnn_max_pool_len": [3, 3, 3],
        "cnn_act": ["leaky_relu", "leaky_relu", "leaky_relu"],
        "cnn_drop": [0.0, 0.0, 0.0],
        "cnn_use_laynorm_inp": True,
        "cnn_use_batchnorm_inp": False,
        "cnn_use_laynorm": [True, True, True],
        "cnn_use_batchnorm": [False, False, False],
        "input_dim": int(input_len),
        "fs": sample_rate,
    }


class EEG_SincMamba(nn.Module):
    def __init__(
        self,
        num_electrodes=64,
        num_sinc_filters=8,
        sinc_kernel=65,
        d_model=64,
        num_classes=2,
        input_len=800,
        sample_rate=250,
        backend="mamba",
        transformer_layers=4,
        transformer_heads=None,
        transformer_dropout=0.1,
    ):
        super().__init__()
        self.backend = backend
        self.num_electrodes = num_electrodes
        self.input_len = input_len

        options = _build_sincnet_options(input_len, sample_rate, num_sinc_filters, sinc_kernel)
        self.sinc_block = SincNet(options)
        self.sinc_channels = int(options["cnn_N_filt"][-1])
        if self.sinc_block.out_dim % self.sinc_channels != 0:
            raise ValueError("SincNet output is not divisible by filter count.")
        self.sinc_seq_len = self.sinc_block.out_dim // self.sinc_channels

        self.bn1 = nn.BatchNorm1d(num_electrodes * self.sinc_channels)

        self.spatial_mix = nn.Sequential(
            nn.Conv1d(
                in_channels=num_electrodes * self.sinc_channels,
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
        if x.dim() != 3:
            raise ValueError("Expected input shape (B, C, T).")
        bsz, channels, length = x.shape
        if channels != self.num_electrodes:
            raise ValueError(f"Expected {self.num_electrodes} channels, got {channels}.")
        if length != self.input_len:
            raise ValueError(f"Expected input length {self.input_len}, got {length}.")

        # SincNet expects (B, T); run it per electrode by folding channels into the batch.
        x = x.reshape(bsz * channels, length)
        x = self.sinc_block(x)
        x = x.view(bsz, channels, self.sinc_channels, self.sinc_seq_len)
        x = x.reshape(bsz, channels * self.sinc_channels, self.sinc_seq_len)
        x = self.bn1(x)
        x = self.spatial_mix(x)
        x = x.permute(0, 2, 1)
        if self.backend == "mamba":
            x = self.mamba(x)
        else:
            x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)
