"""Round-12 Elman decoder controls; see docs/ROUND12D_RNN.md.

The observed-history encoder remains a GRU. Only the rollout decoder is an
ungated tanh RNNCell; all four arms have the same trainable parameter layout.
"""

from __future__ import annotations

from torch import nn

from .base import ENCODER_INPUT_DIM, TRANSITION_INPUT_DIM, RolloutModel
from .context_controls import ContextTransform


class ContextRNNControl(RolloutModel):
    """Frozen/updated raw memory crossed with non-affine hidden-input LN.

    At each step, c = tanh(W_x f + b_x + W_h N_r(memory) + b_h).
    The head reads raw c. Updated arms carry raw c; frozen arms keep the
    original history encoding. There is no gated carry or readout transform.
    """

    latent_is_context = True

    def __init__(
        self,
        normalizer,
        update_context=False,
        context_norm="none",
        width=64,
        dropout=0.1,
        readout_norm="none",
    ):
        if context_norm not in ("none", "ln"):
            raise ValueError("Elman controls support only none/ln hidden-input normalization")
        if readout_norm != "none":
            raise ValueError("Elman controls require an identity readout transform")
        super().__init__(normalizer)
        self.update_context = update_context
        self.encoder = nn.GRU(ENCODER_INPUT_DIM, width, batch_first=True)
        self.projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.context_norm = ContextTransform(width, context_norm)
        self.readout_norm = ContextTransform(width, "none")
        self.decoder = nn.RNNCell(TRANSITION_INPUT_DIM, width, nonlinearity="tanh")
        self.head = nn.Linear(width, 9)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def encode(self, batch):
        _, h = self.encoder(self.history_features(batch))
        memory = self.projection(self.dropout(h[-1]))
        return self.context_norm(memory), memory

    def step_delta(self, feat, context, memory):
        candidate = self.decoder(feat, self.context_norm(memory))
        return self.head(candidate), candidate if self.update_context else memory
