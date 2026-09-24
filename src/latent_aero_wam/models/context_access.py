"""Capacity-matched auxiliary-context intervention; docs/ROUND14_PROTOCOL.md."""
import torch
from torch import nn
from .base import TRANSITION_INPUT_DIM
from .context_controls import ContextMemoryControl


class ContextAccessControl(ContextMemoryControl):
    """Updating GRU with fixed original or current dynamic auxiliary context.

    Both arms have the same active parameter budget. Original context is passed
    separately by RolloutModel and is not overwritten by the recurrent state.
    """
    def __init__(self, normalizer, auxiliary_context='fixed', width=64, dropout=0.1):
        if auxiliary_context not in ('fixed', 'dynamic'):
            raise ValueError(auxiliary_context)
        super().__init__(normalizer, update_context=True, context_norm='none',
                         readout_norm='none', width=width, dropout=dropout)
        self.auxiliary_context = auxiliary_context
        self.decoder = nn.GRUCell(TRANSITION_INPUT_DIM + width, width)

    def step_delta(self, feat, context, memory):
        auxiliary = context if self.auxiliary_context == 'fixed' else memory
        candidate = self.decoder(torch.cat([feat, auxiliary], dim=-1), memory)
        return self.head(candidate), candidate
