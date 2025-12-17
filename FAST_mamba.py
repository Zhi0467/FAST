"""
FAST-Mamba2
-----------
A drop-in variant of FAST where the Transformer encoder stack is replaced by a stack
of Mamba-2 (Mamba2) blocks from https://github.com/state-spaces/mamba.

Key changes vs original FAST:
- CNN-based tokenization (Head + input_layer) is unchanged.
- CLS token is appended to the *end* of the token sequence (per request).
- The Transformer stack is replaced with a residual-pre-norm stack of Mamba2 layers.
- Classification uses the CLS token output (last token).

NOTE: this is only supported on CUDA devices. Not MPS.
"""

from __future__ import annotations

import einops
import torch
import torch.nn as nn

# Reuse the CNN tokenization head from the original FAST implementation.
from FAST import Head  # noqa: E402

try:
    # Upstream usage: `from mamba_ssm import Mamba2`
    from mamba_ssm import Mamba2  # type: ignore
except Exception as e:  # pragma: no cover
    Mamba2 = None  # type: ignore
    _MAMBA_IMPORT_ERROR = e


class Mamba2EncoderBlock(nn.Module):
    """
    Lightweight residual pre-norm wrapper around a single Mamba2 mixer:
        x <- x + Dropout(Mamba2(LN(x)))
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if Mamba2 is None:  # pragma: no cover
            raise ImportError(
                "Failed to import mamba_ssm.Mamba2. Install with e.g.\n"
                "  pip install mamba-ssm[causal-conv1d]\n"
                "Original import error:\n"
                f"{_MAMBA_IMPORT_ERROR}"
            )
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            headdim=headdim,
            ngroups=ngroups,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.mamba(self.norm(x)))


class FAST_Mamba2(nn.Module):
    name = "FAST_Mamba2"

    def __init__(self, config):
        super().__init__()
        self.config = config

        electrodes = config.electrodes
        zone_dict = config.zone_dict
        head = config.head
        dim_cnn = config.dim_cnn
        dim_token = config.dim_token
        seq_len = config.seq_len
        window_len = config.window_len
        slide_step = config.slide_step
        n_classes = config.n_classes
        num_layers = config.num_layers
        dropout = config.dropout
        self.use_spatial_projection = getattr(config, "use_spatial_projection", True)

        # Mamba2 hyperparams (safe defaults if not provided in config)
        mamba_d_state = getattr(config, "mamba_d_state", 64)
        mamba_d_conv = getattr(config, "mamba_d_conv", 4)
        mamba_expand = getattr(config, "mamba_expand", 2)
        mamba_headdim = getattr(config, "mamba_headdim", 64)
        mamba_ngroups = getattr(config, "mamba_ngroups", 1)

        self.n_tokens = (seq_len - window_len) // slide_step + 1

        # CNN tokenization per zone (unchanged from FAST)
        self.head = Head(head, electrodes, zone_dict, dim_cnn)

        # Optional spatial projection (matches FAST behavior)
        if self.use_spatial_projection:
            d_model = dim_token
            self.input_layer = nn.Sequential(
                nn.Linear(dim_cnn * len(zone_dict), dim_token),
                nn.GELU(),
            )
        else:
            d_model = dim_cnn * len(zone_dict)
            self.input_layer = nn.Identity()

        # Transformer -> Mamba2 stack
        self.mamba_stack = nn.Sequential(
            *[
                Mamba2EncoderBlock(
                    d_model,
                    d_state=mamba_d_state,
                    d_conv=mamba_d_conv,
                    expand=mamba_expand,
                    headdim=mamba_headdim,
                    ngroups=mamba_ngroups,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Keep learned positional embedding (optional, but matches FAST style)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_tokens + 1, d_model))

        # CLS token (APPENDED at end in forward_mamba)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Classification head
        self.last_layer = nn.Linear(d_model, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward_head(self, x: torch.Tensor, step_override: int | None = None) -> torch.Tensor:
        if step_override is not None:
            slide_step = step_override
        else:
            slide_step = self.config.slide_step

        x = x.unfold(-1, self.config.window_len, slide_step)  # (B, C, N, T)
        B, C, N, T = x.shape
        x = einops.rearrange(x, "B C N T -> (B N) C T")
        feature = self.head(x)  # (B*N, Z, F)
        feature = einops.rearrange(feature, "(B N) Z F -> B N Z F", B=B)
        return feature

    def batched_forward_head(self, x: torch.Tensor, step: int, batch_size: int) -> torch.Tensor:
        feature = []
        for mb in torch.split(x, batch_size, dim=0):
            feature.append(self.forward_head(mb, step))
        return torch.cat(feature, dim=0)

    def forward_mamba(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten zone features to tokens
        x = einops.rearrange(x, "B N Z F -> B N (Z F)")
        x = self.input_layer(x)  # (B, N, D)

        # Append CLS token at END (requested behavior)
        cls_token_expand = self.cls_token.expand(x.shape[0], -1, -1)  # (B, 1, D)
        x = torch.cat((x, cls_token_expand), dim=1)  # (B, N+1, D)

        # Positional embedding
        x = x + self.pos_embedding[:, : x.shape[1]]

        # Mamba2 stack
        tokens = self.mamba_stack(x)  # (B, N+1, D)

        # Classify from CLS token output (last token)
        cls_out = tokens[:, -1]
        logits = self.last_layer(self.dropout(cls_out))
        return logits

    def forward(self, x: torch.Tensor, forward_mode: str = "default") -> torch.Tensor:
        if forward_mode == "default":
            return self.forward_mamba(self.forward_head(x))
        elif forward_mode == "train_head":
            x = self.forward_head(x)
            x = einops.rearrange(x, "B N Z F -> B N (Z F)")
            tokens = self.input_layer(x)
            logits = self.last_layer(tokens).mean(dim=1)
            return logits
        elif forward_mode == "train_transformer":
            # kept for interface compatibility
            with torch.no_grad():
                x = self.forward_head(x)
            return self.forward_mamba(x)
        else:
            raise NotImplementedError(f"Unknown forward_mode: {forward_mode}")
