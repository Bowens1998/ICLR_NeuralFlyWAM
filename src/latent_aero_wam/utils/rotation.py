"""Rotation utilities: SO(3) projection, 6D representation, exp/log maps.

Conventions used throughout the project:
  * quaternions from Neural-Fly CSVs are scalar-LAST ``[x, y, z, w]``;
  * ``R`` is the body-to-world rotation matrix, i.e. ``v_world = R @ v_body``;
  * rotation increments live in the body frame and compose on the right:
    ``R_{t+1} = R_t @ exp(theta_hat)``.

All torch helpers are batched over leading dimensions and differentiable.
"""

from __future__ import annotations

import numpy as np
import torch

EPS = 1e-8


# --------------------------------------------------------------------------
# numpy side (data loading / preprocessing)
# --------------------------------------------------------------------------
def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project (..., 3, 3) matrices onto SO(3) via SVD.

    The logged Neural-Fly rotation matrices deviate from orthonormality by up to
    ~1.5e-3, which is large enough to bias geodesic errors, so every matrix is
    projected once at load time.
    """
    U, _, Vt = np.linalg.svd(R)
    Rp = U @ Vt
    det = np.linalg.det(Rp)
    # flip the last singular vector where the determinant went negative
    U[..., :, -1] *= np.sign(det)[..., None]
    return U @ Vt


def quat_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Scalar-last quaternion (..., 4) -> rotation matrix (..., 3, 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def enforce_quat_hemisphere(q: np.ndarray) -> np.ndarray:
    """Make a (T, 4) quaternion track sign-continuous in time."""
    q = q.copy()
    flip = np.cumprod(np.sign(np.sum(q[1:] * q[:-1], axis=-1) + EPS))
    q[1:] *= flip[:, None]
    return q


def matrix_to_rot6d_np(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): first two columns of R, stacked."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def matrix_to_quat_np(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 4) scalar-last [x, y, z, w], hemisphere-continuous."""
    w = 0.5 * np.sqrt(np.maximum(1.0 + np.trace(R, axis1=-2, axis2=-1), 1e-12))
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w)
    q = np.stack([x, y, z, w], axis=-1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    for i in range(1, len(q)):
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    return q


def so3_exp_np(w: np.ndarray) -> np.ndarray:
    """(..., 3) rotation vector -> (..., 3, 3) via Rodrigues."""
    angle = np.linalg.norm(w, axis=-1, keepdims=True)
    axis = w / np.maximum(angle, EPS)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -axis[..., 2], axis[..., 1]
    K[..., 1, 0], K[..., 1, 2] = axis[..., 2], -axis[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -axis[..., 1], axis[..., 0]
    a = angle[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(a) * K + (1.0 - np.cos(a)) * (K @ K)


def so3_log_np(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 3) rotation vector."""
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    skew = np.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        axis=-1,
    )
    small = angle < 1e-4
    # near 0: log(R) ~ skew/2; elsewhere: angle / (2 sin angle) * skew
    denom = np.where(small, 2.0, 2.0 * np.sin(angle))
    scale = np.where(small, 0.5, angle / np.maximum(denom, EPS))
    return scale[..., None] * skew


# --------------------------------------------------------------------------
# torch side (model / loss)
# --------------------------------------------------------------------------
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt on the two column vectors."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + EPS)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / (a2.norm(dim=-1, keepdim=True) + EPS)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # columns


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def hat(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) skew-symmetric matrix."""
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues exponential map, (..., 3) -> (..., 3, 3). Safe at w -> 0."""
    theta = w.norm(dim=-1, keepdim=True)
    theta_sq = theta * theta
    small = theta < 1e-4
    # Taylor expansions keep the gradient finite at the origin.
    sin_c = torch.where(small, 1.0 - theta_sq / 6.0, torch.sin(theta) / theta.clamp_min(EPS))
    cos_c = torch.where(
        small, 0.5 - theta_sq / 24.0, (1.0 - torch.cos(theta)) / theta_sq.clamp_min(EPS)
    )
    K = hat(w)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(K.shape)
    return eye + sin_c[..., None] * K + cos_c[..., None] * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) rotation vector. Safe at R -> I."""
    tr = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos)
    skew = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )
    small = angle < 1e-4
    scale = torch.where(small, torch.full_like(angle, 0.5), angle / (2.0 * torch.sin(angle)))
    return scale[..., None] * skew


def geodesic_angle(R_a: torch.Tensor, R_b: torch.Tensor) -> torch.Tensor:
    """Geodesic distance in radians between two batches of rotations."""
    rel = R_a.transpose(-1, -2) @ R_b
    tr = torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1)
    cos = ((tr - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos)
