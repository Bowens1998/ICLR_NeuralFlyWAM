"""Generate the simulated corpus in the main-platform CSV schema (D15).

Mirrors the real protocol exactly: random3-like training flights at
{nowind, 10wind, 20wind, 30wind, 40wind, 50wind}, figure-8 stress flights at
{nowind, 35wind, 70wind, 100wind, 70p20sint}. Files land in data/sim/data/
and parse with the unmodified pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from latent_aero_wam.data.parser import WIND_MPS  # noqa: E402
from latent_aero_wam.sim import QuadrotorSim, figure8_trajectory, random_trajectory  # noqa: E402
from latent_aero_wam.utils.rotation import matrix_to_quat_np  # noqa: E402

OUT = REPO / "data" / "sim" / "data"
DT = 0.02
WIND_DIR = np.array([1.0, 0.0, 0.0])  # fan array blows along +x

TRAIN_CONDS = ["nowind", "10wind", "20wind", "30wind", "40wind", "50wind"]
FIG8_CONDS = ["nowind", "35wind", "70wind", "100wind", "70p20sint"]


matrix_to_quat = matrix_to_quat_np


def wind_fn_for(condition: str):
    if condition == "70p20sint":
        return lambda t: (8.5 + 2.4 * np.sin(t)) * WIND_DIR
    speed = WIND_MPS[condition]
    return lambda t: speed * WIND_DIR


def fmt(a: np.ndarray) -> list[str]:
    return [str([float(x) for x in np.round(row, 8)]) for row in a]


def write_flight(name: str, logs: dict, traj: np.ndarray, v_ref: np.ndarray) -> None:
    n = len(logs["p"])
    q = matrix_to_quat(logs["R"])
    q_sp = matrix_to_quat(logs["R_sp"])
    df = pd.DataFrame(
        {
            "t": np.round(np.arange(n) * DT, 4),
            "p": fmt(logs["p"]),
            "p_d": fmt(traj[:n]),
            "v": fmt(logs["v"]),
            "v_d": fmt(v_ref[:n]),
            "q": fmt(q),
            "R": [str([[float(x) for x in np.round(r, 8)] for r in R]) for R in logs["R"]],
            "w": fmt(logs["w"]),
            "T_sp": [str([round(float(x), 8)]) for x in logs["T_sp"]],
            "q_sp": fmt(q_sp),
            "hover_throttle": [str([0.3924])] * n,
            "fa": fmt(logs["fa"]),
            "pwm": fmt(np.tile([1500.0, 1500.0, 1500.0, 1500.0], (n, 1))),
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv")
    print(f"{name}: {n} steps, wind mean |v|={np.linalg.norm(logs['wind'], axis=1).mean():.2f}")


def main() -> None:
    for i, cond in enumerate(TRAIN_CONDS):
        rng = np.random.default_rng(1000 + i)
        sim = QuadrotorSim(seed=2000 + i)
        traj = random_trajectory(rng, duration=120.0, dt=DT)
        v_ref = np.gradient(traj, DT, axis=0)
        logs = sim.fly(traj, wind_fn_for(cond))
        write_flight(f"sim_random3_baseline_{cond}", logs, traj, v_ref)
    for i, cond in enumerate(FIG8_CONDS):
        sim = QuadrotorSim(seed=3000 + i)
        traj = figure8_trajectory(duration=50.0, dt=DT)
        v_ref = np.gradient(traj, DT, axis=0)
        logs = sim.fly(traj, wind_fn_for(cond))
        write_flight(f"sim_figure8_baseline_{cond}", logs, traj, v_ref)


if __name__ == "__main__":
    main()
