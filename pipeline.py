#!/usr/bin/env python3
"""ChArUco intrinsics pipeline: detect -> pinhole + fisheye calibration -> FOV -> report.

    venv/bin/python pipeline.py --video X.mp4 --tag 1440p

Everything lands in intrinsics_<tag>.json: both models, FOV, validation in px and deg,
board-pose spread, and the leaf-tracker bearing model fitted to the lens.
--no-report stops after the fit and validation.

Board: calib.io 8x11, checker 8 mm, marker 6 mm, DICT_4X4_50.
NOTE the board is (squaresX=11, squaresY=8) with setLegacyPattern(True) -- the
naive (8,11)/non-legacy config detects almost nothing.
"""
import cv2, numpy as np, os, sys, time, json, argparse
from multiprocessing import Pool
import error_contour, azel_model_check

SQX, SQY = 11, 8
# Physical sizes. These do NOT affect the intrinsics (K, D) -- they only set the
# scale of the extrinsics. What does matter for detection is the marker/square
# RATIO, which the detector uses when refining. Overridable via --square/--marker;
# set as globals in main() before the Pool forks so workers inherit them.
SQUARE_LEN, MARKER_LEN = 0.008, 0.006
MIN_CORNERS_DET = 6

ARGS = None


def make_board():
    ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    b = cv2.aruco.CharucoBoard((SQX, SQY), SQUARE_LEN, MARKER_LEN, ad)
    b.setLegacyPattern(True)
    return b


def make_detector(W):
    """Detector params, tuned at 1280 and used unscaled at every resolution.

    An earlier version scaled these by round(W/1280). Measured on real clips that
    was strictly worse: at 4K it gave mean 43.1 corners/frame vs 47.6 unscaled,
    with 3 total detection failures vs 0. It also left adaptiveThreshWinSizeMin
    pinned at 3 while max/step scaled (window sweep 3,27,51,... at 4K -- a 3 px
    adaptive-threshold window is noise), and produced even window sizes at s=2
    (43*2=86) where the block size must be odd. Do not reintroduce the scaling
    without re-running detector_test.py.
    """
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dp.cornerRefinementWinSize = 5
    dp.adaptiveThreshWinSizeMin = 3
    dp.adaptiveThreshWinSizeMax = 43
    dp.adaptiveThreshWinSizeStep = 8
    dp.minMarkerPerimeterRate = 0.005      # relative, no scaling needed
    dp.polygonalApproxAccuracyRate = 0.05
    cp = cv2.aruco.CharucoParameters()
    cp.minMarkers = 1
    cp.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(make_board(), cp, dp)


def worker(a):
    wid, start, end, video, outdir, detdir, W, H = a
    cv2.setNumThreads(1)
    cd = make_detector(W)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    recs = []
    for fi in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci, mc, mi = cd.detectBoard(g)
        if ci is None or len(ci) < MIN_CORNERS_DET:
            continue
        cc = cc.reshape(-1, 2).astype(np.float32)
        ci = ci.reshape(-1).astype(np.int32)
        x0, y0 = np.clip(cc.min(0) - 5, 0, [W - 1, H - 1]).astype(int)
        x1, y1 = np.clip(cc.max(0) + 5, 0, [W - 1, H - 1]).astype(int)
        roi = g[y0:y1 + 1, x0:x1 + 1]
        sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size > 100 else 0.0
        cv2.imwrite(f"{outdir}/f{fi:06d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        recs.append((fi, cc, ci, sharp))
    cap.release()
    np.save(f"{detdir}/part_{wid:02d}.npy", np.array(recs, dtype=object), allow_pickle=True)
    return wid, len(recs)


def stage_detect(video, tag, nproc):
    outdir, detdir = f"frames_{tag}", f".det_{tag}"
    det_file = f"detections_{tag}.npy"
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"video {video}: {n} frames @ {W}x{H}", flush=True)
    if os.path.exists(det_file):
        print("  detections cached, skipping detect stage")
        return np.load(det_file, allow_pickle=True), W, H
    os.makedirs(outdir, exist_ok=True); os.makedirs(detdir, exist_ok=True)
    b = np.linspace(0, n, nproc + 1).astype(int)
    jobs = [(i, int(b[i]), int(b[i + 1]), video, outdir, detdir, W, H) for i in range(nproc)]
    t0 = time.time()
    with Pool(nproc) as p:
        for wid, c in p.imap_unordered(worker, jobs):
            print(f"  worker {wid:2d}: {c} hits ({time.time()-t0:.0f}s)", flush=True)
    parts = [np.load(f"{detdir}/part_{i:02d}.npy", allow_pickle=True) for i in range(nproc)]
    allr = sorted([r for p_ in parts if len(p_) for r in p_], key=lambda r: r[0])
    np.save(det_file, np.array(allr, dtype=object), allow_pickle=True)
    for f in os.listdir(detdir):
        os.remove(os.path.join(detdir, f))
    os.rmdir(detdir)
    nc = np.array([len(r[2]) for r in allr])
    print(f"  detect done in {time.time()-t0:.0f}s: {len(allr)}/{n} frames ({100*len(allr)/n:.1f}%), "
          f"corners mean {nc.mean():.1f} median {np.median(nc):.0f} max {nc.max()}")
    return np.array(allr, dtype=object), W, H


def coverage_report(recs, W, H, tag, save=True):
    """Where in the image plane did we actually observe corners? Distortion is only
    constrained where there is data, so this gates whether FOV is trustworthy."""
    heat = np.zeros((H, W), np.float32)
    rr = max(6, W // 140)
    for r in recs:
        for x, y in r[1]:
            cv2.circle(heat, (int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))), rr, 1, -1)
    occ = heat > 0
    if save:
        cv2.imwrite(f"coverage_{tag}.jpg", cv2.applyColorMap(
            (np.clip(heat / max(heat.max(), 1) * 3, 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO))
    eh, ew = int(0.15 * H), int(0.15 * W)
    print(f"  coverage: {occ.mean()*100:.1f}% of image area")
    stats = {"total": float(occ.mean() * 100)}
    for n, sl in [("top", occ[:eh]), ("bottom", occ[-eh:]),
                  ("left", occ[:, :ew]), ("right", occ[:, -ew:])]:
        stats[n] = float(sl.mean() * 100)
        print(f"    {n:7s} 15%: {stats[n]:5.1f}%")
    q = [occ[:H//2, :W//2], occ[:H//2, W//2:], occ[H//2:, :W//2], occ[H//2:, W//2:]]
    print("    quadrants  : " + "  ".join(f"{x.mean()*100:.0f}%" for x in q))
    # how close to the frame corners did any corner observation get?
    allp = np.concatenate([r[1] for r in recs])
    for name, (u, v) in [("TL", (0, 0)), ("TR", (W-1, 0)), ("BL", (0, H-1)), ("BR", (W-1, H-1))]:
        dmin = np.hypot(allp[:, 0] - u, allp[:, 1] - v).min()
        print(f"    nearest observation to {name}: {dmin:6.0f} px")
    return stats


def select_views(recs, W, H, n_views, min_corners):
    ncor = np.array([len(r[2]) for r in recs])
    sharp = np.array([r[3] for r in recs])
    fidx = np.array([r[0] for r in recs])
    thr = np.percentile(sharp, 35)
    cand = np.flatnonzero((ncor >= min_corners) & (sharp >= thr))
    if len(cand) < 30:                       # relax if the clip is short
        cand = np.flatnonzero(ncor >= max(12, min_corners // 2))
    GX, GY = 32, 18
    cells = []
    for i in cand:
        c = recs[i][1]
        cx = np.clip((c[:, 0] / W * GX).astype(int), 0, GX - 1)
        cy = np.clip((c[:, 1] / H * GY).astype(int), 0, GY - 1)
        cells.append(np.unique(cy * GX + cx))
    M = np.zeros((len(cand), GX * GY), np.float32)
    for k, cl in enumerate(cells):
        M[k, cl] = 1.0
    w = np.ones(GX * GY, np.float32)
    avail = np.ones(len(cand), bool)
    chosen = []
    for _ in range(n_views):
        sc = M @ w; sc[~avail] = -1
        k = int(np.argmax(sc))
        if sc[k] <= 0:
            break
        chosen.append(cand[k])
        avail &= np.abs(fidx[cand] - fidx[cand[k]]) >= 3
        avail[k] = False
        w[cells[k]] *= 0.55
    rows = [np.flatnonzero(cand == c)[0] for c in chosen]
    cov = (M[rows].sum(0) > 0).mean()
    print(f"  selected {len(chosen)} views, cell coverage {cov*100:.1f}% "
          f"(candidates {len(cand)}, sharp_thr {thr:.0f})")
    return np.array(chosen)


def validate(recs, K, D, OBJP, fisheye):
    """solvePnP against every detected frame. Returns the summary for the JSON and the
    per-corner residuals, which the error maps reuse rather than re-posing every frame."""
    r = error_contour.residuals(recs, K, D, OBJP, fisheye, min_corners=8, max_frame_mean=20)
    a, d = r["e_px"], r["e_deg"]
    return dict(rms=float(np.sqrt((a ** 2).mean())), mean=float(a.mean()),
                median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                rms_deg=float(np.sqrt((d ** 2).mean())), mean_deg=float(d.mean()),
                median_deg=float(np.median(d)), p95_deg=float(np.percentile(d, 95)),
                frames=r["frames"], points=int(len(a))), r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nproc", type=int, default=14)
    ap.add_argument("--views", type=int, default=500)
    ap.add_argument("--min-corners", type=int, default=30)
    ap.add_argument("--detect-only", action="store_true",
                    help="detect + coverage report only, skip calibration")
    ap.add_argument("--no-report", action="store_true",
                    help="skip the bearing-model fit and error maps")
    ap.add_argument("--square", type=float, default=0.008, help="checker size in metres")
    ap.add_argument("--marker", type=float, default=0.006, help="aruco marker size in metres")
    a = ap.parse_args()

    global SQUARE_LEN, MARKER_LEN
    SQUARE_LEN, MARKER_LEN = a.square, a.marker
    print(f"board: {SQX}x{SQY} squares, checker {SQUARE_LEN*1000:.1f} mm, "
          f"marker {MARKER_LEN*1000:.1f} mm (ratio {MARKER_LEN/SQUARE_LEN:.4f}), DICT_4X4_50, legacy")

    recs, W, H = stage_detect(a.video, a.tag, a.nproc)
    print(f"\n=== coverage {a.tag} ({W}x{H}) ===")
    coverage_report(recs, W, H, a.tag)
    if a.detect_only:
        return
    OBJP = make_board().getChessboardCorners().astype(np.float64)
    print(f"\n=== calibrating {a.tag} ({W}x{H}) ===")
    sel = select_views(recs, W, H, a.views, a.min_corners)

    def build(idxs, f64=False):
        op = [OBJP[recs[i][2]].reshape(-1, 1, 3).astype(np.float64 if f64 else np.float32) for i in idxs]
        ip = [recs[i][1].reshape(-1, 1, 2).astype(np.float64 if f64 else np.float32) for i in idxs]
        return op, ip

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 300, 1e-9)

    def pinhole_staged(op, ip, K_init=None):
        """Release parameters gradually. Fitting all 5 distortion terms plus a free
        principal point from a cold start lets the polynomial run away when edge
        coverage is thin (observed: fx diverging to 2496 on the merged 1440p set).

        A stage that frees the principal point can still walk it off the sensor, and
        OpenCV then raises "Principal point must be within the image" and kills the run
        (observed on the 12 mm clip, whose top edge is thinly covered). Each stage is
        therefore retried with the principal point pinned, and failing that the previous
        stage's result stands -- a slightly under-released fit beats no fit at all."""
        K = (K_init.copy() if K_init is not None
             else np.array([[0.5 * W, 0, W / 2], [0, 0.5 * W, H / 2], [0, 0, 1]], float))
        D = np.zeros(14)
        G = cv2.CALIB_USE_INTRINSIC_GUESS
        margin = 0.15                      # keep the principal point this far inside the frame
        inside = lambda M: (margin * W < M[0, 2] < (1 - margin) * W
                            and margin * H < M[1, 2] < (1 - margin) * H)
        for fl in (G | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST
                   | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
                   G | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
                   G | cv2.CALIB_FIX_K3,
                   G):
            for attempt in (fl, fl | cv2.CALIB_FIX_PRINCIPAL_POINT):
                try:
                    _, Kn, Dn, *_ = cv2.calibrateCamera(op, ip, (W, H), K.copy(), D.copy(),
                                                        flags=attempt, criteria=crit)
                except cv2.error:
                    continue
                if not (np.all(np.isfinite(Kn)) and np.all(np.isfinite(Dn)) and inside(Kn)):
                    continue
                K, D = Kn, Dn
                break
            else:
                print(f"    (pinhole stage {fl:#x} unusable -- keeping the previous stage)")
        return K, D

    COLD = np.array([[0.5 * W, 0, W / 2], [0, 0.5 * W, H / 2], [0, 0, 1]], float)

    def fisheye_fit(idxs, K_init=None, min_corners=0):
        """cv2.fisheye.calibrate asserts (InitExtrinsics: fabs(norm_u1) > 0) on
        near-degenerate views. Retry on progressively cleaner subsets, then give up
        gracefully rather than killing the run."""
        for mc in (min_corners, 25, 40, 55):
            use = [i for i in idxs if len(recs[i][2]) >= mc]
            if len(use) < 20:
                continue
            opf = [OBJP[recs[i][2]].reshape(1, -1, 3) for i in use]
            ipf = [recs[i][1].reshape(1, -1, 2).astype(np.float64) for i in use]
            Kf = (K_init.copy() if K_init is not None else COLD.copy())
            try:
                r = cv2.fisheye.calibrate(
                    opf, ipf, (W, H), Kf, np.zeros((4, 1)),
                    flags=cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW
                    | cv2.CALIB_USE_INTRINSIC_GUESS, criteria=crit)
                if mc != min_corners:
                    print(f"    (fisheye needed min_corners>={mc}: {len(use)} views)")
                return r
            except cv2.error:
                continue
        print("    *** fisheye calibration failed on all subsets ***")
        return float("nan"), COLD.copy(), np.zeros((4, 1)), None, None

    def drop_outlier_views(idxs, K, D, pct=92):
        """Views whose own solvePnP pose does not fit the provisional intrinsics. View
        selection is greedy for image coverage, so it actively favours frames where the
        board reaches the frame edge at a steep angle -- the most informative views and
        also the most often mis-detected. On the 12 mm clip a handful of them (10-25
        corners, degenerate poses, per-frame RMS up to 4.8e5 px) dragged the seed fit to
        RMS 3.9 while uniformly sampled halves of the same clip fitted to 0.7-0.9."""
        e = []
        for i in idxs:
            o = OBJP[recs[i][2]].reshape(1, -1, 3)
            p = recs[i][1].reshape(1, -1, 2).astype(np.float64)
            try:
                ok, rv, tv = cv2.fisheye.solvePnP(o, p, K, np.asarray(D, np.float64).reshape(4, 1))
                pr, _ = cv2.fisheye.projectPoints(o, rv, tv, K, np.asarray(D, np.float64).reshape(4, 1))
                r = np.sqrt(((pr.reshape(-1, 2) - p.reshape(-1, 2)) ** 2).sum(1).mean()) if ok else np.inf
            except cv2.error:
                r = np.inf
            e.append(r if np.isfinite(r) else np.inf)
        e = np.asarray(e)
        fin = e[np.isfinite(e)]
        if len(fin) < 20:
            return np.asarray(idxs), e
        # a fixed percentile alone cannot cope with a long tail; also cut anything far
        # above the bulk, so a good clip loses nothing and a spiky one loses the spikes
        thr = max(np.percentile(fin, pct), 3.0 * np.median(fin))
        keep = np.asarray(idxs)[e <= thr]
        return (keep if len(keep) >= 20 else np.asarray(idxs)), e

    # ---- fisheye FIRST: it stays well conditioned at wide FOV even with coverage
    # holes, so its K is a far better seed for the polynomial fit than 0.5*W.
    # (Seeding the pinhole cold made fx run away to 419 on a clip missing the top
    # of frame -- 23.9 deg angular residual, silently reported as a normal result.)
    rms_f0, Kf0, Df0, _, _ = fisheye_fit(sel)
    # ...then re-seed on the views that fit it, so a few bad detections cannot poison
    # every stage downstream (the pinhole rounds below reject outliers of their own,
    # but only after the staged fit has already been built on the bad seed).
    clean, ev = drop_outlier_views(sel, Kf0, Df0)
    if len(clean) < len(sel):
        rms_f1, Kf1, Df1, _, _ = fisheye_fit(clean)
        print(f"  seed: dropped {len(sel)-len(clean)}/{len(sel)} views as outliers "
              f"(worst {np.max(ev[np.isfinite(ev)]):.1f} px), RMS {rms_f0:.4f} -> {rms_f1:.4f}")
        if np.isfinite(rms_f1) and rms_f1 < rms_f0:
            sel, Kf0, Df0 = clean, Kf1, Df1

    # ---- pinhole, 2 rounds with outlier rejection, seeded from fisheye ----
    op, ip = build(sel)
    Kp, Dp = pinhole_staged(op, ip, K_init=Kf0)
    _, Kp, Dp, _, _, _, _, pve = cv2.calibrateCameraExtended(
        op, ip, (W, H), Kp, Dp, flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=crit)
    pve = np.asarray(pve).ravel()
    keep = sel[pve <= np.percentile(pve, 92)]
    op, ip = build(keep)
    Kp, Dp = pinhole_staged(op, ip, K_init=Kp)
    rms_p, Kp, Dp, _, _, sdi, _, pve2 = cv2.calibrateCameraExtended(
        op, ip, (W, H), Kp, Dp, flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=crit)
    Dp = Dp.ravel()[:5]
    print(f"  pinhole  RMS {rms_p:.4f} px on {len(keep)} views (dropped {len(sel)-len(keep)})")

    rms_f, Kf, Df, _, _ = fisheye_fit(keep, K_init=Kf0)
    print(f"  fisheye  RMS {rms_f:.4f} px on {len(keep)} views")

    # ---- divergence guard: the two models must agree on focal length ----
    dev = abs(Kp[0, 0] - Kf[0, 0]) / Kf[0, 0]
    if dev > 0.15:
        print(f"  *** WARNING: pinhole fx {Kp[0,0]:.1f} vs fisheye fx {Kf[0,0]:.1f} "
              f"({dev*100:.0f}% apart) -- the polynomial fit has diverged. "
              f"Trust the fisheye result; check the coverage report for holes. ***")

    vp, res_p = validate(recs, Kp, Dp.reshape(1, 5), OBJP, False)
    vf, res_f = validate(recs, Kf, Df, OBJP, True)
    print(f"  validation over all detected frames:")
    for n, v in (("pinhole", vp), ("fisheye", vf)):
        print(f"    {n} RMS {v['rms']:.4f} px  mean {v['mean']:.4f}  median {v['median']:.4f}"
              f"  | RMS {v['rms_deg']:.4f} deg  median {v['median_deg']:.4f}  ({v['frames']} fr, {v['points']} pts)")
    tilt, dist = res_f["tilt"], res_f["dist"]
    print(f"  board tilt from fronto-parallel: median {np.median(tilt):.1f}  p90 {np.percentile(tilt,90):.1f}  "
          f"max {tilt.max():.1f} deg, {(tilt > 30).mean()*100:.0f}% of frames above 30"
          f" | distance {dist.min():.2f}..{dist.max():.2f} m")
    if np.percentile(tilt, 90) < 30:
        print("  *** little board tilt: focal length is weakly constrained -- film 30-45 deg tilts. ***")

    # ---- FOV ----
    fx, fy, cx, cy = Kf[0, 0], Kf[1, 1], Kf[0, 2], Kf[1, 2]
    k = Df.ravel()
    t = np.linspace(0, np.radians(89.9), 400000)
    td = t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6 + k[3] * t**8)
    m = np.diff(td) <= 0
    iend = int(np.argmax(m)) if m.any() else len(t) - 1

    def ray(u, v):
        xd, yd = (u - cx) / fx, (v - cy) / fy
        rd = np.hypot(xd, yd)
        if rd < 1e-12:
            return np.array([0., 0., 1.]), 0.0
        if rd > td[iend]:
            return None, np.nan
        th = float(np.interp(rd, td[:iend], t[:iend]))
        d = np.array([xd / rd * np.sin(th), yd / rd * np.sin(th), np.cos(th)])
        return d / np.linalg.norm(d), np.degrees(th)

    ang = lambda A, B: float(np.degrees(np.arccos(np.clip(float(A @ B), -1, 1))))
    L, R = ray(0, cy)[0], ray(W - 1, cy)[0]
    T, B = ray(cx, 0)[0], ray(cx, H - 1)[0]
    corners = [ray(0, 0), ray(W - 1, 0), ray(0, H - 1), ray(W - 1, H - 1)]
    hf, vf_ = ang(L, R), ang(T, B)
    if all(c[0] is not None for c in corners):
        df = max(ang(corners[0][0], corners[3][0]), ang(corners[1][0], corners[2][0]))
    else:
        df = float("nan")
    # centred convention (comparable across resolutions)
    hc = 2 * ray(cx + W / 2, cy)[1]
    vc = 2 * ray(cx, cy + H / 2)[1]
    # naive rectified-pinhole numbers, for comparison
    hp = float(np.degrees(2 * np.arctan(W / 2 / Kp[0, 0])))
    vp_ = float(np.degrees(2 * np.arctan(H / 2 / Kp[1, 1])))

    print(f"\n  FOV (fisheye, measured):  H {hf:.2f}  V {vf_:.2f}  D {df:.2f} deg")
    print(f"  FOV centred convention :  H {hc:.2f}  V {vc:.2f}")
    print(f"  naive rectified pinhole:  H {hp:.2f}  V {vp_:.2f}  (naive convention)")
    print(f"  corner off-axis angles : " + "  ".join(f"{c[1]:.1f}" for c in corners) + " deg")
    print(f"\n  pinhole K: fx {Kp[0,0]:.3f} fy {Kp[1,1]:.3f} cx {Kp[0,2]:.3f} cy {Kp[1,2]:.3f}")
    print(f"  pinhole D: [{', '.join(f'{v:+.6f}' for v in Dp)}]")
    print(f"  fisheye K: fx {fx:.3f} fy {fy:.3f} cx {cx:.3f} cy {cy:.3f}")
    print(f"  fisheye D: [{', '.join(f'{v:+.6f}' for v in k)}]")

    # pinhole model validity at the corners
    rdc = [np.hypot((u - Kp[0, 2]) / Kp[0, 0], (v - Kp[1, 2]) / Kp[1, 1])
           for u, v in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]]
    kk = Dp
    rr = np.linspace(0, 4, 400001)
    fr = rr * (1 + kk[0] * rr**2 + kk[1] * rr**4 + kk[4] * rr**6)
    mm = np.diff(fr) <= 0
    rd_max = float(fr[int(np.argmax(mm))]) if mm.any() else float(fr[-1])
    print(f"  pinhole poly invertible to rd={rd_max:.4f}; corners need "
          f"{min(rdc):.4f}..{max(rdc):.4f} -> {'OK' if max(rdc) <= rd_max else 'CORNERS OUTSIDE MODEL'}")

    out = {
        "video": a.video, "image_size": [W, H], "n_frames_with_board": len(recs),
        "n_calib_views": int(len(keep)),
        "pinhole": {"fx": Kp[0, 0], "fy": Kp[1, 1], "cx": Kp[0, 2], "cy": Kp[1, 2],
                    "dist": [float(v) for v in Dp], "rms_calib_px": float(rms_p),
                    "validation": vp, "poly_invertible_to_rd": rd_max,
                    "corner_rd_min": float(min(rdc)), "corner_rd_max": float(max(rdc)),
                    "corners_within_model": bool(max(rdc) <= rd_max)},
        "fisheye": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "dist": [float(v) for v in k], "rms_calib_px": float(rms_f),
                    "validation": vf},
        "fov_deg": {"h": hf, "v": vf_, "d": df, "h_centred": hc, "v_centred": vc,
                    "h_naive_pinhole": hp, "v_naive_pinhole": vp_},
        "board_pose": {"tilt_deg": {"median": float(np.median(tilt)), "p90": float(np.percentile(tilt, 90)),
                                    "max": float(tilt.max()), "frac_above_30": float((tilt > 30).mean())},
                       "distance_m": {"min": float(dist.min()), "median": float(np.median(dist)),
                                      "max": float(dist.max())}},
    }
    dump = lambda: json.dump(out, open(f"intrinsics_{a.tag}.json", "w"), indent=2, default=float)
    dump()                                   # the fit survives even if the report stage dies
    np.savez(f"calib_pinhole_{a.tag}.npz", K=Kp, D=Dp, size=np.array([W, H]))
    np.savez(f"calib_fisheye_{a.tag}.npz", K=Kf, D=Df, size=np.array([W, H]))
    print(f"\nwrote intrinsics_{a.tag}.json, calib_pinhole_{a.tag}.npz, calib_fisheye_{a.tag}.npz")
    if a.no_report:
        return

    # ---- report: the tracker's bearing model fitted to this lens, and where the error sits ----
    print(f"\n=== report {a.tag} ===")
    out["bearing_model"] = azel_model_check.run(Kf, Df, W, H, a.tag)
    try:
        out["error_maps"] = error_contour.plot_maps({"fisheye": res_f, "pinhole": res_p}, W, H, a.tag, a.tag)
    except ImportError:
        print("  (matplotlib not installed -- error maps skipped)")
    dump()
    print(f"updated intrinsics_{a.tag}.json with bearing_model, error_maps")


if __name__ == "__main__":
    main()
