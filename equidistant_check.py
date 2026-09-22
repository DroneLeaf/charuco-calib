#!/usr/bin/env python3
"""Can this lens be treated as a pure equidistant camera -- r = f*theta, fx = fy, no polynomial?
Standalone; not part of the pipeline (the tracker's own model is azel_model_check.py).

    venv/bin/python equidistant_check.py --tag 8mm --square 0.033 --marker 0.024

Reprojection error alone cannot answer this. With free board poses the solver buys a low
RMS for a wrong projection law by moving the focal length and the board distances
together, so a model can reproject to a pixel or two while pointing degrees away from the
truth. This fits the pure model, then compares its pixel -> ray map against the full
fisheye calibration, which is the number that matters for a spherical model. Three
variants:

    fitted             f, cx, cy as the solver chooses them with D = 0
    paraxial           f and principal point from the full calibration (right at the centre)
    best_single_focal  the single f that minimises angular error over the whole image

Writes calib_fisheye_<tag>-equidist.npz (the fitted variant, so error_contour.py
--calib-tag can map its reprojection error) and equidist_truth_<tag>.png (needs
matplotlib); the CLI also writes equidistant_<tag>.json.
"""
import cv2, numpy as np, argparse, json
import rays

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 300, 1e-9)
BASE = (cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4)


def run(recs, K0, D0, W, H, OBJP, tag, views=470, plot=True):
    K0 = np.asarray(K0, np.float64); D0 = np.asarray(D0, np.float64).reshape(4, 1)
    ncor = np.array([len(r[2]) for r in recs]); sharp = np.array([r[3] for r in recs])
    cand = np.flatnonzero((ncor >= 40) & (sharp >= np.percentile(sharp, 40)))
    idx = cand[:: max(1, len(cand) // views)]
    opf = [OBJP[recs[i][2]].reshape(1, -1, 3) for i in idx]
    ipf = [recs[i][1].reshape(1, -1, 2).astype(np.float64) for i in idx]

    def fit(f=None, K_init=None):
        """D pinned to zero. cv2.fisheye.calibrate ignores CALIB_FIX_ASPECT_RATIO, so
        fx = fy is enforced by pinning the focal length and searching over it from outside."""
        K = (K_init if K_init is not None else K0).copy()
        fl = BASE
        if f is not None:
            K[0, 0] = K[1, 1] = f; fl |= cv2.CALIB_FIX_FOCAL_LENGTH
        rms, K, _, _, _ = cv2.fisheye.calibrate(opf, ipf, (W, H), K, np.zeros((4, 1)), flags=fl, criteria=CRIT)
        return rms, K

    print(f"  pure equidistant fit on {len(idx)} views")
    rms_free, Kfree = fit()
    fs = 0.5 * (Kfree[0, 0] + Kfree[1, 1]) * np.array([0.98, 0.99, 1.0, 1.01, 1.02])
    rs = np.array([fit(f, Kfree)[0] for f in fs])
    c = np.polyfit(fs, rs ** 2, 2)
    f_iso = float(np.clip(-c[1] / (2 * c[0]), fs[0], fs[-1])) if c[0] > 0 else float(fs[np.argmin(rs)])
    rms_iso, Kiso = fit(f_iso, Kfree)
    print(f"    fitted     : RMS {rms_iso:.4f} px  f {f_iso:.1f}  cx {Kiso[0,2]:.1f}  cy {Kiso[1,2]:.1f}"
          f"   ({100*(f_iso/K0[0,0]-1):+.0f}% vs full-model fx {K0[0,0]:.1f})")
    np.savez(f"calib_fisheye_{tag}-equidist.npz", K=Kiso, D=np.zeros((4, 1)), size=np.array([W, H]))

    us, vs, pix = rays.pixel_grid(W, H)
    ref = rays.fisheye_rays(K0, D0, pix)
    cx0, cy0 = K0[0, 2], K0[1, 2]
    fgrid = np.linspace(0.85, 1.05, 401) * K0[0, 0]
    f_best = float(fgrid[np.argmin([np.sqrt((rays.angle(rays.equidistant_rays(f, cx0, cy0, pix), ref) ** 2).mean())
                                    for f in fgrid])])
    variants = [("fitted", f"as the solver fits it  (f = {f_iso:.0f}, boresight removed)",
                 rays.angle(rays.equidistant_rays(f_iso, Kiso[0, 2], Kiso[1, 2], pix), ref, align=True)),
                ("paraxial", f"paraxial focal from the full model  (f = {K0[0,0]:.0f})",
                 rays.angle(rays.equidistant_rays(K0[0, 0], cx0, cy0, pix), ref)),
                ("best_single_focal", f"best single focal over the image  (f = {f_best:.0f})",
                 rays.angle(rays.equidistant_rays(f_best, cx0, cy0, pix), ref))]
    print(f"    true pointing error vs the full calibration, whole image (deg):")
    for name, _, e in variants:
        print(f"      {name:18s} mean {e.mean():6.3f}  p95 {np.percentile(e,95):6.3f}  max {e.max():6.3f}"
              f"   (~{np.radians(e.mean())*K0[0,0]:.0f} px mean, {np.radians(e.max())*K0[0,0]:.0f} px max)")

    out = {
        "model": "pure equidistant, r = f*theta, fx = fy, no distortion polynomial",
        "reference": "this file's fisheye model (equidistant + k1..k4)",
        "n_fit_views": int(len(idx)),
        "fitted": {"f": f_iso, "cx": Kiso[0, 2], "cy": Kiso[1, 2], "rms_calib_px": rms_iso,
                   "note": "what a pure-equidistant calibration returns; low RMS, wrong focal -- do not use for angles",
                   "pointing_error_deg": rays.stats(variants[0][2])},
        "fitted_free_aspect": {"fx": Kfree[0, 0], "fy": Kfree[1, 1], "cx": Kfree[0, 2], "cy": Kfree[1, 2],
                               "rms_calib_px": rms_free},
        "paraxial": {"f": K0[0, 0], "cx": cx0, "cy": cy0, "pointing_error_deg": rays.stats(variants[1][2])},
        "best_single_focal": {"f": f_best, "cx": cx0, "cy": cy0,
                              "note": "the pure equidistant model to use if one must be used",
                              "pointing_error_deg": rays.stats(variants[2][2])},
        "files": [f"calib_fisheye_{tag}-equidist.npz"],
    }
    if plot:
        try:
            out["files"].append(_plot(variants, us, vs, W, H, K0, f"equidist_truth_{tag}.png",
                                      f"True angular error of a pure equidistant model — {tag} ({W}×{H})"))
        except ImportError:
            print("    (matplotlib not installed -- equidist_truth plot skipped)")
    return out


def _plot(variants, us, vs, W, H, K0, out, title, prefix="pure equidistant, "):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    INK, MUTED, SURFACE = "#1f1f1e", "#6b6a66", "#ffffff"
    ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
                         "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    n = len(variants)
    fig, axes = plt.subplots(n, 1, figsize=(10, 0.6 + 5.2 * n), facecolor=SURFACE, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (name, desc, e) in zip(axes, variants):
        g = e.reshape(us.shape)
        step = [s for s in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5) if g.max() / s <= 12][0]
        levels = np.round(np.arange(0, np.ceil(g.max() / step) * step + step / 2, step), 6)
        cmap = LinearSegmentedColormap.from_list("seq_blue", ramp, N=len(levels) - 1)
        norm = BoundaryNorm(levels, cmap.N)
        mid = 0.5 * levels[-1]
        fmt = "%.2f" if step < 0.1 else "%.1f" if step < 1 else "%.0f"
        cf = ax.contourf(us, vs, g, levels=levels, cmap=cmap, norm=norm)
        cl = ax.contour(us, vs, g, levels=levels[1::2], colors=SURFACE, linewidths=0.6, alpha=0.7)
        ax.clabel(cl, fmt=fmt + "°", fontsize=7.5, inline_spacing=2,
                  colors=[INK if v < mid else SURFACE for v in cl.levels])
        ax.plot(K0[0, 2], K0[1, 2], "+", color=INK, ms=11, mew=1.6)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_aspect("equal"); ax.set_ylabel("v (px)")
        ax.set_title(f"{prefix}{desc}", loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=20)
        ax.text(0, 1.025, f"pointing error  mean {e.mean():.2f}°   p95 {np.percentile(e,95):.2f}°   max {e.max():.2f}°"
                f"   ≈ {np.radians(e.mean())*K0[0,0]:.0f} px mean, {np.radians(e.max())*K0[0,0]:.0f} px max",
                transform=ax.transAxes, fontsize=9, color=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        cb = fig.colorbar(cf, ax=ax, ticks=levels, format=fmt, fraction=0.025, pad=0.02)
        cb.set_label("pointing error (deg)", color=INK); cb.outline.set_visible(False)
    axes[-1].set_xlabel("u (px)")
    fig.suptitle(title, x=0.07, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.07, 0.2 / (0.6 + 5.2 * n), "angle between the ray each model assigns to a pixel and the ray from the full "
             "calibration; each panel has its own scale", fontsize=8.5, color=MUTED)
    fig.subplots_adjust(left=0.07, right=0.93, top=1 - 1.15 / (0.6 + 5.2 * n),
                        bottom=0.8 / (0.6 + 5.2 * n), hspace=0.22)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"    wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--square", type=float, default=0.015)
    ap.add_argument("--marker", type=float, default=0.011)
    ap.add_argument("--views", type=int, default=470)
    a = ap.parse_args()
    recs = np.load(f"detections_{a.tag}.npy", allow_pickle=True)
    F = np.load(f"calib_fisheye_{a.tag}.npz")
    W, H = [int(v) for v in F["size"]]
    board = cv2.aruco.CharucoBoard((11, 8), a.square, a.marker,
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    board.setLegacyPattern(True)
    OBJP = board.getChessboardCorners().astype(np.float64)
    print(f"[{a.tag}] {W}x{H}")
    out = run(recs, F["K"], F["D"], W, H, OBJP, a.tag, a.views)
    json.dump(out, open(f"equidistant_{a.tag}.json", "w"), indent=2, default=float)
    print(f"wrote equidistant_{a.tag}.json")


if __name__ == "__main__":
    main()
