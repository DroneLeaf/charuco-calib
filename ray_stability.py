#!/usr/bin/env python3
"""How far can the pixel -> ray map be trusted? Standalone; writes stability_<tag>.json.

    venv/bin/python ray_stability.py --tag 8mm --square 0.033 --marker 0.024

Reprojection RMS does not answer this: fits whose RMS agrees to 0.001 px have disagreed
on focal length by 1-1.5% here, because focal length trades against board distance and
the distortion terms unless the board is filmed at strong tilt. So calibrate K interleaved
folds independently (uniformly sampled views, each fold spanning the whole clip) and
compare ray maps, fold against fold and fold against the main fit.

Read the two numbers together. Fold-to-fold is the statistical noise. Fold-vs-main is
what a different, equally defensible choice of views does to the answer; when it is much
larger than fold-to-fold the uncertainty is systematic, and more footage of the same kind
will not shrink it.
"""
import cv2, numpy as np, argparse, json
import rays

CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 300, 1e-9)


def fov(K, D, W, H):
    r = rays.fisheye_rays(K, D, np.array([[0, K[1, 2]], [W - 1, K[1, 2]], [K[0, 2], 0], [K[0, 2], H - 1]]))
    a = lambda A, B: float(np.degrees(np.arccos(np.clip(A @ B, -1, 1))))
    return a(r[0], r[1]), a(r[2], r[3])


def run(recs, Kf, Df, W, H, OBJP, folds=4, max_views=800):
    Kf = np.asarray(Kf, np.float64); Df = np.asarray(Df, np.float64).reshape(4, 1)
    ncor = np.array([len(r[2]) for r in recs]); sharp = np.array([r[3] for r in recs])
    cand = np.flatnonzero((ncor >= 40) & (sharp >= np.percentile(sharp, 40)))
    cand = cand[:: max(1, len(cand) // max_views)]
    _, _, pix = rays.pixel_grid(W, H, 40.0)
    main = rays.fisheye_rays(Kf, Df, pix)
    print(f"  ray-map stability: {len(cand)} uniformly sampled views -> {folds} interleaved folds")
    per, R = [], []
    for k in range(folds):
        idx = cand[k::folds]
        opf = [OBJP[recs[i][2]].reshape(1, -1, 3) for i in idx]
        ipf = [recs[i][1].reshape(1, -1, 2).astype(np.float64) for i in idx]
        try:
            rms, K, D, _, _ = cv2.fisheye.calibrate(
                opf, ipf, (W, H), Kf.copy(), np.zeros((4, 1)),
                flags=cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | cv2.CALIB_USE_INTRINSIC_GUESS,
                criteria=CRIT)
        except cv2.error:
            print(f"    fold {k}: fisheye calibration failed, skipped")
            continue
        r = rays.fisheye_rays(K, D, pix); R.append(r)
        h, v = fov(K, D, W, H)
        e = rays.angle(r, main, align=True)
        per.append({"n_views": int(len(idx)), "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                    "dist": [float(x) for x in D.ravel()], "rms_calib_px": float(rms),
                    "fov_h": h, "fov_v": v, "vs_main_deg": rays.stats(e)})
        print(f"    fold {k}: n={len(idx):4d}  fx {K[0,0]:8.2f}  cx {K[0,2]:8.2f}  cy {K[1,2]:7.2f}  RMS {rms:.4f}"
              f"  H FOV {h:.2f}  vs main: mean {e.mean():.3f} max {e.max():.3f} deg")
    if len(per) < 2:
        return {"folds": per}
    pairs = [rays.angle(R[i], R[j], align=True) for i in range(len(R)) for j in range(i + 1, len(R))]
    fx = np.array([p["fx"] for p in per]); hf = np.array([p["fov_h"] for p in per])
    f2f = float(np.mean([p.mean() for p in pairs])); f2m = float(np.mean([p["vs_main_deg"]["mean"] for p in per]))
    h_main, v_main = fov(Kf, Df, W, H)
    out = {
        "n_views": int(len(cand)),
        "fx": {"main": Kf[0, 0], "fold_mean": float(fx.mean()), "fold_std": float(fx.std(ddof=1)),
               "fold_vs_main_pct": float(100 * (fx.mean() / Kf[0, 0] - 1))},
        "fov_h": {"main": h_main, "fold_min": float(hf.min()), "fold_max": float(hf.max())},
        "fold_to_fold_deg": {"mean": f2f, "max": float(max(p.max() for p in pairs))},
        "fold_vs_main_deg": {"mean": f2m, "max": float(max(p["vs_main_deg"]["max"] for p in per))},
        "systematic": bool(f2m > 3 * f2f),
        "note": "boresight rotation removed before comparing; fold_vs_main >> fold_to_fold means the "
                "answer depends on which views are used (systematic), not on how many",
        "folds": per,
    }
    print(f"    fold-to-fold mean {f2f:.3f} deg | fold-vs-main mean {f2m:.3f} deg "
          f"(max {out['fold_vs_main_deg']['max']:.3f}) | fx folds {fx.mean():.1f} +- {fx.std(ddof=1):.1f} "
          f"vs main {Kf[0,0]:.1f} ({out['fx']['fold_vs_main_pct']:+.2f}%)")
    if out["systematic"]:
        print("    *** view-selection dependence dominates: treat the angular scale as known only to "
              f"~{abs(out['fx']['fold_vs_main_pct']):.1f}%. More board tilt / a flatter board, not more frames. ***")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--square", type=float, default=0.015)
    ap.add_argument("--marker", type=float, default=0.011)
    a = ap.parse_args()
    recs = np.load(f"detections_{a.tag}.npy", allow_pickle=True)
    F = np.load(f"calib_fisheye_{a.tag}.npz")
    W, H = [int(v) for v in F["size"]]
    board = cv2.aruco.CharucoBoard((11, 8), a.square, a.marker,
                                   cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    board.setLegacyPattern(True)
    print(f"[{a.tag}] {W}x{H}")
    out = run(recs, F["K"], F["D"], W, H, board.getChessboardCorners().astype(np.float64), a.folds)
    json.dump(out, open(f"stability_{a.tag}.json", "w"), indent=2, default=float)
    print(f"wrote stability_{a.tag}.json")


if __name__ == "__main__":
    main()
