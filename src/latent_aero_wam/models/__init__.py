"""Model registry. First round is M0-M3 plus one eval-only diagnostic."""

from __future__ import annotations

from ..data.normalize import Normalizer
from .base import RolloutModel, RolloutOutput
from .context_controls import ContextMemoryControl, HistoryWindWAM, InformationControlWAM
from .context_rnn import ContextRNNControl
from .context_access import ContextAccessControl
from .m0_persistence import PersistenceModel
from .m1_history_gru import HistoryGRUModel
from .m2_latent_wam import JEPALatentWAM, LatentAeroWAM, OracleLatentWAM, StateOnlyLatentWAM

MODEL_REGISTRY = {
    "context_access_control": ContextAccessControl,
    "context_memory_control": ContextMemoryControl,
    "context_rnn_control": ContextRNNControl,
    "information_control_wam": InformationControlWAM,
    "history_wind_wam": HistoryWindWAM,
    "m0_persistence": PersistenceModel,
    "m1_history_gru": HistoryGRUModel,
    "m2_latent_wam": LatentAeroWAM,
    "m2_jepa": JEPALatentWAM,
    "m3_state_only": StateOnlyLatentWAM,
    "m2_oracle": OracleLatentWAM,
    # Round-3 width ablation: same architecture, distinct names so runs land
    # in their own directories alongside the round-1 16-d baseline.
    "m2_latent_wam_w32": LatentAeroWAM,
    "m2_latent_wam_w64": LatentAeroWAM,
    # Round-4 mechanism test: latent LayerNorm removed (D13).
    "m2_latent_wam_nonorm": LatentAeroWAM,
    "m2_latent_wam_w64_nonorm": LatentAeroWAM,
    # Round-5 mechanism 2x2 (D15): normalization x magnitude-channel.
    "m2_latent_wam_magpass": LatentAeroWAM,
    "m2_latent_wam_w64_magpass": LatentAeroWAM,
    "m2_latent_wam_rmsnorm": LatentAeroWAM,
    "m2_latent_wam_lnnoaffine": LatentAeroWAM,
}

__all__ = [
    "MODEL_REGISTRY",
    "RolloutModel",
    "RolloutOutput",
    "build_model",
]


def build_model(cfg: dict, normalizer: Normalizer, dt: float) -> RolloutModel:
    """Instantiate a model from ``cfg['model'] = {name: ..., **kwargs}``."""
    spec = dict(cfg)
    name = spec.pop("name")
    family = spec.pop("family", name)
    if family not in MODEL_REGISTRY:
        raise KeyError(f"unknown model family '{family}'; known: {sorted(MODEL_REGISTRY)}")
    if family == "m0_persistence":
        spec["dt"] = dt
    return MODEL_REGISTRY[family](normalizer, **spec)
