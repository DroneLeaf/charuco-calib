#!/usr/bin/env python3
"""The leaf-tracker bearing model fitted to this lens: the hfov / vfov / rot_x / rot_y to put
in the tracker config, and the pointing error that model is left with. pipeline.py runs this
at the end of every pass and stores it under "bearing_model" in intrinsics_<tag>.json; the
CLI re-runs it from a saved calibration and updates that JSON in place.

    venv/bin/python azel_model_check.py --tag 8mm

The model (leaf_tracker angles.compute_los + apply_compensating_angles), twist 0, optical-
centre offsets left at zero. Frame: x = image UP, y = image RIGHT, z = forward.

    az = (u - W/2) / W * hfov     el = -(v - H/2) / H * vfov      each LINEAR in its pixel offset
    b  = (sin el, cos el sin az, cos el cos az)
    b  <- Ry(rot_y) @ Rx(rot_x) @ b

so the only knobs are hfov_deg, vfov_deg and the two compensating angles, which here also
have to absorb the off-centre principal point. Right-handed, so +rot_x steers LEFT and
+rot_y steers UP: the top-left pixel is the most positive in both. This is NOT the radial
equidistant model (theta = r / f): it is separable per axis.

hfov and vfov are fitted independently, and come out anisotropic on a lens that compresses
toward the edges even though the pixels are square: the long axis reaches further into the
compressed region, so its best single slope is coarser. The isotropic alternative (one
deg/px on both axes) is reported alongside for consumers that need it.

The error is a comparison of pixel -> ray maps against the full fisheye calibration, not a
reprojection residual. Needs matplotlib for the plot.
"""
import numpy as np, argparse, json, os
import rays, equidistant_check

TO_XUP = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])    # CV (X right, Y down, Z fwd) -> x up, y right, z fwd


def chain(pix, W, H, hfov, vfov, rx, ry):
    az = (pix[:, 0] / W - 0.5) * np.radians(hfov)
    el = -(pix[:, 1] / H - 0.5) * np.radians(vfov)
    b = np.c_[np.sin(el), np.cos(el) * np.sin(az), np.cos(el) * np.cos(az)]
    cx_, sx_, cy_, sy_ = np.cos(np.radians(rx)), np.sin(np.radians(rx)), np.cos(np.radians(ry)), np.sin(np.radians(ry))
    Rx = np.array([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]])
    Ry = np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
    return b @ (Ry @ Rx).T


def run(K0, D0, W, H, tag, plot=True):
    K0 = np.asarray(K0, np.float64); D0 = np.asarray(D0, np.float64).reshape(4, 1)
    us, vs, pix = rays.pixel_grid(W, H)
    ref = rays.fisheye_rays(K0, D0, pix) @ TO_XUP.T

    def fit(unpack, p):
        """Gauss-Newton; `unpack` maps the free parameters to (hfov, vfov, rot_x, rot_y) in deg."""
        p = np.array(p, float)
        r = lambda q: (chain(pix, W, H, *unpack(q)) - ref).ravel()
        for _ in range(40):
            r0 = r(p)
            J = np.stack([(r(p + 1e-4 * np.eye(len(p))[k]) - r0) / 1e-4 for k in range(len(p))], 1)
            step = np.linalg.lstsq(J, -r0, rcond=None)[0]
            p += step
            if np.abs(step).max() < 1e-8:
                break
        return unpack(p)

    h0 = np.degrees(W / K0[0, 0])                       # start from the paraxial scale
    best = fit(lambda q: (q[0], q[1], q[2], q[3]), [h0, h0 * H / W, 0, 0])
    iso = fit(lambda q: (q[0], q[0] * H / W, q[1], q[2]), [h0, 0, 0])
    e, e_iso = (rays.angle(chain(pix, W, H, *p), ref) for p in (best, iso))

    print(f"  leaf-tracker bearing model (twist 0, offsets 0; x up, y right, +rot toward top-left):")
    print(f"    hfov_deg: {best[0]:.2f}\n    vfov_deg: {best[1]:.2f}")
    print(f"    rot_around_x_compensating_angle_deg: {best[2]:+.2f}\n    rot_around_y_compensating_angle_deg: {best[3]:+.2f}")
    print(f"    pointing error vs full calibration: mean {e.mean():.3f}  p95 {np.percentile(e,95):.3f}  max {e.max():.3f} deg"
          f"   ({best[0]/W:.5f} / {best[1]/H:.5f} deg/px H / V)")
    print(f"    isotropic alternative: hfov {iso[0]:.2f}  vfov {iso[1]:.2f}  -> mean {e_iso.mean():.3f}  max {e_iso.max():.3f} deg")
    out = {
        "model": "leaf-tracker az/el: az, el linear in the pixel offset; b = Ry(rot_y) Rx(rot_x) "
                 "(sin el, cos el sin az, cos el cos az); x up, y right, z fwd; twist 0; oc offsets 0",
        "hfov_deg": best[0], "vfov_deg": best[1],
        "rot_around_x_compensating_angle_deg": best[2], "rot_around_y_compensating_angle_deg": best[3],
        "deg_per_px": {"h": best[0] / W, "v": best[1] / H},
        "pointing_error_deg": rays.stats(e),
        "isotropic_alternative": {"hfov_deg": iso[0], "vfov_deg": iso[1],
                                  "rot_around_x_compensating_angle_deg": iso[2],
                                  "rot_around_y_compensating_angle_deg": iso[3],
                                  "pointing_error_deg": rays.stats(e_iso)},
    }
    if plot:
        try:
            desc = f"hfov {best[0]:.2f}, vfov {best[1]:.2f}, rot_x {best[2]:+.2f}, rot_y {best[3]:+.2f}"
            out["plot"] = equidistant_check._plot(
                [("best_fit", desc, e)], us, vs, W, H, K0, f"azel_truth_{tag}.png",
                f"True angular error of the leaf-tracker bearing model — {tag} ({W}×{H})", prefix="")
        except ImportError:
            print("    (matplotlib not installed -- azel_truth plot skipped)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    F = np.load(f"calib_fisheye_{a.tag}.npz")
    W, H = [int(v) for v in F["size"]]
    print(f"[{a.tag}] {W}x{H}")
    out = run(F["K"], F["D"], W, H, a.tag)
    jf = f"intrinsics_{a.tag}.json"
    d = json.load(open(jf)) if os.path.exists(jf) else {}
    d["bearing_model"] = out
    json.dump(d, open(jf, "w"), indent=2, default=float)
    print(f"updated {jf} [bearing_model]")


if __name__ == "__main__":
    main()
