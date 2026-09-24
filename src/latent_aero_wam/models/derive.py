"""Recompute the deterministic history features a controller would not carry."""

from __future__ import annotations

import torch

from ..utils.rotation import rot6d_to_matrix, so3_log


def derive_history_features(
    history_state: torch.Tensor, current_state: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(v_body, delta_state)`` for a (B, K, 12) history.

    ``delta_state[:, k]`` is the increment from step ``k`` to ``k+1``; the final
    entry uses ``current_state``, so the history never reads past time ``t``.
    """
    seq = torch.cat([history_state, current_state.unsqueeze(1)], dim=1)  # (B, K+1, 12)
    v, R = seq[..., 0:3], rot6d_to_matrix(seq[..., 3:9])
    omega = seq[..., 9:12]
    v_body = torch.einsum("bkji,bkj->bki", R[:, :-1], v[:, :-1])
    R_rel = R[:, :-1].transpose(-1, -2) @ R[:, 1:]
    delta = torch.cat(
        [v[:, 1:] - v[:, :-1], so3_log(R_rel), omega[:, 1:] - omega[:, :-1]], dim=-1
    )
    return v_body, delta
