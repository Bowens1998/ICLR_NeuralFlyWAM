"""M1 -- matched generic history predictor.

A GRU reads the same interaction history as M2 (states, actions *and* observed
increments, so the two models see identical features), and a second GRU decodes
the future conditioned on the given actions. The decoder's hidden state is
updated at every rollout step: nothing in the architecture separates a slowly
varying dynamics state from the fast motion state. That separation is the only
thing M2 adds, and hidden sizes are chosen so M1 is never the smaller model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.normalize import Normalizer
from ..data.parser import DELTA_DIM
from .base import ENCODER_INPUT_DIM, TRANSITION_INPUT_DIM, RolloutModel


class HistoryGRUModel(RolloutModel):
    latent_is_context = True

    def __init__(
        self,
        normalizer: Normalizer,
        encoder_hidden: int = 64,
        decoder_hidden: int = 72,
        dropout: float = 0.0,
    ):
        super().__init__(normalizer)
        self.encoder = nn.GRU(ENCODER_INPUT_DIM, encoder_hidden, batch_first=True)
        self.init_hidden = nn.Linear(encoder_hidden, decoder_hidden)
        self.decoder = nn.GRUCell(TRANSITION_INPUT_DIM, decoder_hidden)
        self.head = nn.Linear(decoder_hidden, DELTA_DIM)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def encode(self, batch: dict):
        _, h = self.encoder(self.history_features(batch, include_action=True))
        h = self.dropout(h[-1])
        return h, torch.tanh(self.init_hidden(h))

    def step_delta(self, feat: torch.Tensor, context: torch.Tensor, memory: torch.Tensor):
        memory = self.decoder(feat, memory)
        return self.head(memory), memory
