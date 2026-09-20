"""Pixel -> unit-ray helpers shared by pipeline.py and the report scripts.

Parameters (fx, cx, k's) trade off against each other, so two calibrations are compared
by the bearing each assigns to the same pixel, not by their numbers.
"""
import cv2, numpy as np

ITC = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-12)


def pixel_grid(W, H, step=20.0):
    us, vs = np.meshgrid(np.arange(step / 2, W, step), np.arange(step / 2, H, step))
    return us, vs, np.stack([us.ravel(), vs.ravel()], 1)


def _unit(n):
    r = np.c_[n, np.ones(len(n))]
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def fisheye_rays(K, D, pix):
    p = np.ascontiguousarray(pix, np.float64).reshape(-1, 1, 2)
    return _unit(cv2.fisheye.undistortPoints(p, K, np.asarray(D, np.float64).reshape(4, 1),
                                             criteria=ITC).reshape(-1, 2))


def pinhole_rays(K, D, pix):
    p = np.ascontiguousarray(pix, np.float64).reshape(-1, 1, 2)
    return _unit(cv2.undistortPoints(p, K, np.asarray(D, np.float64).reshape(1, -1),
                                     criteria=ITC).reshape(-1, 2))


def equidistant_rays(f, cx, cy, pix):
    """Pure r = f*theta, fx = fy, no polynomial."""
    x, y = (pix[:, 0] - cx) / f, (pix[:, 1] - cy) / f
    th = np.hypot(x, y)
    s = np.where(th > 1e-12, np.sin(th) / np.maximum(th, 1e-12), 1.0)
    return np.c_[x * s, y * s, np.cos(th)]


def kabsch(A, B):
    """Rotation R minimising |A @ R - B|. A principal-point shift is ~a camera rotation,
    which extrinsics absorb, so remove it before calling two ray maps different."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def angle(A, B, align=False):
    """Per-pixel angle in degrees between two ray maps."""
    if align:
        A = A @ kabsch(A, B)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(A, B), axis=1), (A * B).sum(1)))


def stats(e):
    return {"mean": float(e.mean()), "p95": float(np.percentile(e, 95)), "max": float(e.max())}
