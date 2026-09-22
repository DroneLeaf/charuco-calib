#!/usr/bin/env python3
"""Filled-contour maps of mean reprojection error over the image plane, with a colourbar,
once in pixels and once in degrees. pipeline.py runs this at the end of every pass; the
CLI is for re-plotting from saved artefacts.

    venv/bin/python error_contour.py --tag 8mm --square 0.033 --marker 0.024

The angular error is the angle between the ray through the detected corner (unprojected
through the model) and the ray to the board corner itself (R*X + t). It is not px times a
constant: deg/px grows toward the edges on a lens with barrel distortion.

Every detected frame is re-posed with solvePnP against the calibration, per-point errors
are binned on a coarse grid, and the sum and count grids are smoothed together so sparse
cells don't dominate. Cells with too few observations are left blank rather than
extrapolated. Plotting needs matplotlib.
"""
import cv2, numpy as np, argparse
import rays


def residuals(recs, K, D, OBJP, fisheye, min_corners=12, max_frame_mean=None):
    """Per-corner reprojection error (px and deg) against a fresh solvePnP pose, plus the
    board tilt from fronto-parallel and distance for every frame that posed."""
    K = np.asarray(K, np.float64); D = np.asarray(D, np.float64).ravel()
    D = D[:4].reshape(4, 1) if fisheye else D[:5].reshape(1, -1)
    px, er, ea, tilt, dist = [], [], [], [], []
    for r in recs:
        if len(r[2]) < min_corners:
            continue
        o = OBJP[r[2]].reshape(-1, 1, 3)
        ip = r[1].reshape(-1, 1, 2).astype(np.float64)
        try:
            if fisheye:
                ok, rv, tv = cv2.fisheye.solvePnP(o.reshape(1, -1, 3), ip.reshape(1, -1, 2), K, D)
                if not ok: continue
                pr, _ = cv2.fisheye.projectPoints(o.reshape(1, -1, 3), rv, tv, K, D)
            else:
                ok, rv, tv = cv2.solvePnP(o.astype(np.float32), ip.astype(np.float32), K, D)
                if not ok: continue
                pr, _ = cv2.projectPoints(o, rv, tv, K, D)
            obs = (rays.fisheye_rays if fisheye else rays.pinhole_rays)(K, D, ip)
        except cv2.error:
            continue
        e = np.linalg.norm(pr.reshape(-1, 2) - ip.reshape(-1, 2), axis=1)
        R, _ = cv2.Rodrigues(np.asarray(rv, np.float64))
        prd = o.reshape(-1, 3) @ R.T + np.asarray(tv, np.float64).reshape(1, 3)
        ang = rays.angle(obs, prd / np.linalg.norm(prd, axis=1, keepdims=True))
        if not (np.all(np.isfinite(e)) and np.all(np.isfinite(ang))):
            continue
        if max_frame_mean is not None and e.mean() > max_frame_mean:
            continue
        px.append(ip.reshape(-1, 2)); er.append(e); ea.append(ang)
        tilt.append(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))); dist.append(float(np.linalg.norm(tv)))
    return dict(px=np.concatenate(px), e_px=np.concatenate(er), e_deg=np.concatenate(ea),
                tilt=np.array(tilt), dist=np.array(dist), K=K, frames=len(tilt))


def _grid(px, e, W, H, cell):
    gx, gy = int(np.ceil(W / cell)), int(np.ceil(H / cell))
    xi = np.clip((px[:, 0] / cell).astype(int), 0, gx - 1)
    yi = np.clip((px[:, 1] / cell).astype(int), 0, gy - 1)
    acc = np.zeros((gy, gx), np.float32); cnt = np.zeros((gy, gx), np.float32)
    np.add.at(acc, (yi, xi), e); np.add.at(cnt, (yi, xi), 1)
    acc = cv2.GaussianBlur(acc, (0, 0), 1.5); cnt = cv2.GaussianBlur(cnt, (0, 0), 1.5)
    mean = np.ma.masked_where(cnt < 15, acc / np.maximum(cnt, 1e-6))
    return (np.arange(gx) + 0.5) * cell, (np.arange(gy) + 0.5) * cell, mean


def plot_maps(res, W, H, label, out_tag, cell=20, clip=5.0):
    """res: {model name: residuals()}. One panel per model on a shared colourbar, written
    once in px and once in deg. Corners above `clip` px are misdetections, not lens error,
    and are dropped from both. Returns the two file names."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

    # one hue, light -> dark: magnitude, not category
    INK, MUTED, SURFACE, NODATA = "#1f1f1e", "#6b6a66", "#ffffff", "#ecebe7"
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    models, n = list(res), len(res)
    keep = {m: res[m]["e_px"] <= clip for m in models}
    files = []
    for unit, key, min_step in (("px", "e_px", 0.1), ("deg", "e_deg", 0.0025)):
        grids = {m: _grid(res[m]["px"][keep[m]], res[m][key][keep[m]], W, H, cell) for m in models}
        # shared levels so the panels read against one colourbar
        allv = np.concatenate([g[2].compressed() for g in grids.values()])
        p_lo, p_hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
        step = [s for s in (0.0025, 0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5)
                if s >= min_step and (p_hi - p_lo) / s <= 14][0]
        fmt = f"%.{max(1, len(f'{step:.4f}'.rstrip('0').split('.')[1]))}f"
        levels = np.round(np.arange(np.floor(p_lo / step) * step,
                                    np.ceil(p_hi / step) * step + step / 2, step), 6)
        cmap = LinearSegmentedColormap.from_list("seq_blue", ramp, N=len(levels) + 1)
        cmap.set_under(ramp[0]); cmap.set_over("#082548")
        norm = BoundaryNorm(levels, cmap.N, extend="both")
        mid = 0.5 * (levels[0] + levels[-1])     # dark ink on the light half of the ramp
        f3 = "%.4f" if unit == "deg" else "%.3f"

        fig, axes = plt.subplots(n, 1, figsize=(10, 1.0 + 5.3 * n), facecolor=SURFACE, sharex=True)
        for ax, m in zip(np.atleast_1d(axes), models):
            xs, ys, mean = grids[m]
            e, K = res[m][key][keep[m]], res[m]["K"]
            ax.set_facecolor(NODATA)
            cf = ax.contourf(xs, ys, mean, levels=levels, cmap=cmap, norm=norm, extend="both")
            cl = ax.contour(xs, ys, mean, levels=levels[::2], colors=SURFACE, linewidths=0.6, alpha=0.7)
            lab = cl.levels[::2]                 # every 4th level: corners have steep gradients
            ax.clabel(cl, levels=lab, fmt=fmt, fontsize=7.5, inline_spacing=2,
                      colors=[INK if v < mid else SURFACE for v in lab])
            ax.plot(K[0, 2], K[1, 2], "+", color=INK, ms=11, mew=1.6)
            ax.annotate("principal point", (K[0, 2], K[1, 2]), xytext=(9, -13),
                        textcoords="offset points", fontsize=8.5, color=INK)
            ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal")
            ax.set_ylabel("v (px)")
            ax.set_title(m, loc="left", fontsize=12, fontweight="bold", color=INK, pad=20)
            ax.text(0, 1.025, f"mean {f3 % e.mean()} {unit}   median {f3 % np.median(e)} {unit}   "
                    f"RMS {f3 % np.sqrt((e**2).mean())} {unit}   "
                    f"({len(e):,} corners, {int((~keep[m]).sum())} above {clip:g} px dropped)",
                    transform=ax.transAxes, fontsize=9, color=MUTED)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        np.atleast_1d(axes)[-1].set_xlabel("u (px)")
        what = "Reprojection error" if unit == "px" else "Angular reprojection error"
        fig.suptitle(f"{what} across the image — {label} ({W}×{H})",
                     x=0.075, ha="left", fontsize=14, fontweight="bold", color=INK)
        fig.subplots_adjust(left=0.075, right=0.86, top=1 - 1.05 / (1.0 + 5.3 * n),
                            bottom=0.7 / (1.0 + 5.3 * n), hspace=0.2)
        cb = fig.colorbar(cf, cax=fig.add_axes([0.885, 0.2, 0.018, 0.6]), ticks=levels, format=fmt)
        cb.set_label(f"mean {what.lower()} ({unit})", color=INK)
        cb.outline.set_visible(False)
        fig.text(0.075, 0.015, "grey = fewer than 15 observations nearby; not extrapolated",
                 fontsize=8.5, color=MUTED)
        out = f"errcontour_{out_tag}.png" if unit == "px" else f"errcontour_{out_tag}_deg.png"
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        files.append(out)
        print(f"  wrote {out}  (levels {fmt % levels[0]}..{fmt % levels[-1]} {unit})")
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--square", type=float, default=0.015)
    ap.add_argument("--marker", type=float, default=0.011)
    ap.add_argument("--calib-tag", default=None,
                    help="read calib_<model>_<calib-tag>.npz instead of the detections' tag, "
                         "e.g. 8mm-equidist from equidistant_check.py")
    ap.add_argument("--models", default="fisheye,pinhole")
    ap.add_argument("--min-corners", type=int, default=12)
    ap.add_argument("--cell", type=int, default=20, help="bin size in px")
    ap.add_argument("--clip", type=float, default=5.0,
                    help="drop gross outliers above this many px (misdetections, not lens error)")
    a = ap.parse_args()
    ctag = a.calib_tag or a.tag

    recs = np.load(f"detections_{a.tag}.npy", allow_pickle=True)
    board = cv2.aruco.CharucoBoard((11, 8), a.square, a.marker,
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    board.setLegacyPattern(True)
    OBJP = board.getChessboardCorners().astype(np.float64)
    res = {}
    for m in a.models.split(","):
        for suffix in ("_all", ""):
            try:
                C = np.load(f"calib_{m}_{ctag}{suffix}.npz")
                break
            except FileNotFoundError:
                continue
        W, H = [int(v) for v in C["size"]]
        res[m if ctag == a.tag else f"{m}  ({ctag})"] = residuals(
            recs, C["K"], C["D"], OBJP, m == "fisheye", a.min_corners)
    plot_maps(res, W, H, a.tag, ctag, a.cell, a.clip)


if __name__ == "__main__":
    main()
