# charuco-calib

Camera intrinsics from a video of a ChArUco board. Detects the board across the whole
clip, picks a well-spread subset of views, fits **both** a pinhole (Brown–Conrady) and a
fisheye (equidistant) model, and reports field of view with the caveats that actually
matter on wide lenses.

The reason it fits both: past roughly 120° the naive `2*atan(W/2/fx)` number from a
pinhole fit is wrong by tens of degrees. On one 4K clip the pinhole fit implies H 98.9°
while the fisheye model measures H 135.6° — and the pinhole distortion polynomial isn't
even invertible out as far as the image corners, so no amount of rectification recovers
them.

## The board

Generate one with the [calib.io pattern generator](https://calib.io/pages/camera-calibration-pattern-generator):

- **Target type:** ChArUco
- **Rows 8 × columns 11**, i.e. an 8×11 board
- **Checker size** and **marker size** — anything, as long as you know them. The defaults
  here are 8 mm / 6 mm; the larger boards used with `--square 0.015 --marker 0.011` are
  15 mm / 11 mm.
- **Dictionary:** 4×4 (50 markers) → `DICT_4X4_50`

Download the PDF and print it **at 100% scale** — no "fit to page", no margin scaling.
Then measure one printed square edge-to-edge with calipers and pass the real number to
`--square` / `--marker`, not the nominal one; consumer printers are routinely off by a
percent or two. Mount it on something rigid and flat (foam board, aluminium composite).
Any bow in the board shows up as distortion the fit will happily absorb. calib.io also
sells flat printed/anodised boards if the paper route isn't accurate enough.

Two things about how the board is constructed in code:

```python
board = cv2.aruco.CharucoBoard((11, 8), square, marker, dict_4x4_50)
board.setLegacyPattern(True)
```

It is `(squaresX=11, squaresY=8)` **with `setLegacyPattern(True)`** — that matches the
calib.io marker layout. The naive `(8, 11)` / non-legacy configuration detects almost
nothing, which looks exactly like a bad clip rather than a config error.

The physical square/marker sizes do not affect `K`/`D` at all; they only set the scale of
the extrinsics. The *ratio* between them does matter, because the detector uses it when
refining corners.

## Install

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Needs OpenCV ≥ 4.7 for the current `CharucoDetector` API (developed against 5.0).
matplotlib is only used for the report plots; without it the run still completes and
skips them.

## Run

```bash
venv/bin/python pipeline.py --video clip_1440p.mp4 --tag 1440p --square 0.008 --marker 0.006
```

For a larger board, e.g. 33 mm squares with 24 mm markers, filmed at 720p:

```bash
venv/bin/python pipeline.py --video clip_720p.mp4 --tag 720p --square 0.033 --marker 0.024
```

Everything is keyed by `--tag`, so one tag carries through the whole toolchain.
A pass writes:

| file | contents |
|---|---|
| `detections_<tag>.npy` | per-frame corners, ids, sharpness (cached — reruns skip detection) |
| `frames_<tag>/` | JPEGs of every frame the board was found in |
| `coverage_<tag>.jpg` | where in the image plane corners were actually observed |
| `calib_pinhole_<tag>.npz`, `calib_fisheye_<tag>.npz` | `K`, `D`, `size` |
| `intrinsics_<tag>.json` | **everything**: both models, FOV, validation in px and deg, board-pose spread, the leaf-tracker bearing model fitted to the lens, plot file names |
| `errcontour_<tag>.png`, `errcontour_<tag>_deg.png` | reprojection error across the image, in px and in degrees |
| `azel_truth_<tag>.png` | pointing error left by the leaf-tracker bearing model once fitted to this lens (see below) |

Useful flags: `--detect-only` (detection + coverage report, no fit), `--no-report` (stop
after fit + validation; skips the bearing-model fit and plots), `--views`
(how many views to select, default 500), `--nproc`, `--min-corners`.

**Several clips of the same camera:** concatenate them and calibrate the result as one
video. Same codec/resolution/fps concatenates without re-encoding:

```bash
printf "file '%s'\n" "$PWD"/clip_a.mp4 "$PWD"/clip_b.mp4 > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy clip_merged.mp4
```

### What is in `intrinsics_<tag>.json`

| key | contents |
|---|---|
| `pinhole`, `fisheye` | `fx fy cx cy`, `dist`, calibration RMS, and `validation` over every detected frame — `rms/mean/median/p95` in px and the same with a `_deg` suffix in degrees. `fisheye` is OpenCV's model: equidistant **plus** a `k1..k4` polynomial, not a pure `r = f·θ` lens. |
| `fov_deg` | measured, centred-convention and naive-pinhole FOV |
| `board_pose` | board tilt from fronto-parallel and distance across the clip. Little tilt (p90 under ~30°) leaves focal length weakly constrained; the run warns. |
| `bearing_model` | the leaf-tracker az/el model (`angles.compute_los` + compensating angles; twist 0, optical-centre offsets 0; x up, y right, +rot toward the top-left) fitted to this lens: `hfov_deg`, `vfov_deg`, `rot_around_x_compensating_angle_deg`, `rot_around_y_compensating_angle_deg` to paste into the tracker config, the `deg_per_px` they imply, and the **true pointing error** that model is left with against the full calibration. hfov / vfov are fitted independently and come out anisotropic on a lens that compresses toward its edges; `isotropic_alternative` is the best single-deg/px pair. |
| `error_maps` | the two contour plot file names |

Reprojection RMS is not a pointing accuracy. With free board poses the solver can buy a
low RMS for a wrong projection law by moving focal length and board distance together —
on one 57° lens a pure equidistant fit reprojected to 1.6 px while pointing 7° off. The
`bearing_model` error is therefore a comparison of pixel→ray maps, not a residual.

### Filming the clip

Almost every bad calibration traces back to the capture, not the fit. The board is only
a few hundred corners per frame, and the solver cannot invent constraints that were never
filmed. Four things matter, roughly in order:

**Keep the board in focus.** Check focus before you start recording, not after — on a
fixed-focus lens that means confirming the board is beyond the hyperfocal distance, and on
anything autofocusing it means locking focus so it can't hunt mid-clip. A soft board still
detects, so nothing warns you; the corners just land a fraction of a pixel off, and that
error goes straight into `D`. If the printed squares don't look crisp on playback, refilm.

**Move slowly.** Motion blur is the single biggest killer of corner accuracy, and it is
worst exactly where you need the data most — near the frame edges, where you tend to swing
the board fastest. Glide, don't sweep; pause for a beat at each position. The pipeline
scores every frame with a Laplacian-variance sharpness metric and discards the bottom 35%,
but that only picks the best of what you gave it. It cannot recover detail that was never
captured. Short exposure helps, so film bright.

**Cover as much of the frame as you can.** Get the board into all four corners and along
every edge, not just the comfortable centre region, and tilt it to a range of angles —
30–45° in both axes, not just fronto-parallel. Distortion is only constrained where there
is data, so an unvisited corner means the model is extrapolating there, and on a wide lens
that is precisely where distortion is strongest. A board that only ever appears flat and
central leaves focal length and the distortion terms fighting each other, and the fit can
diverge without ever looking wrong. Aim for a clip a few minutes long.

**Check the coverage image after every run.** `coverage_<tag>.jpg` is written before the
fit, and reading it takes five seconds. Dark regions are places the model is guessing.
The console report next to it gives edge/quadrant percentages and the distance from each
frame corner to the nearest observation — if those corner distances are large, refilm
rather than trusting the FOV number. Use `--detect-only` to get the coverage report without
waiting for the calibration.

### What a good result looks like

Aim for **calibration RMS below 0.8 px** on both models. Both numbers are printed at the
fit stage:

```
  pinhole  RMS 0.7106 px on 552 views (dropped 48)
  fisheye  RMS 0.5669 px on 552 views
```

Above ~1 px, don't reach for solver settings — go back to the capture. It is nearly always
blur or thin edge coverage, and both are cheaper to fix by refilming than to paper over.
Note that the separate validation RMS, measured over *every* detected frame rather than the
selected views, runs higher because it includes marginal frames the fit deliberately
excluded; the two aren't comparable, so hold the 0.8 px target against the calibration
number. Also check that pinhole and fisheye `fx` agree — the run warns if they don't — and
that the coverage report shows data near all four corners.

## Pipeline stages

1. **Detect** — `nproc` workers each take a contiguous slice of the video, run
   `CharucoDetector`, and record corners plus a Laplacian-variance sharpness score over
   the board ROI. Results merge into one cached `.npy`.
2. **Coverage report** — heatmap plus edge/quadrant/corner statistics. Distortion is only
   constrained where there is data, so this gates whether the FOV number is trustworthy at
   all. Read it before believing anything downstream.
3. **View selection** — greedy max-coverage over a 32×18 image grid, restricted to frames
   above the 35th sharpness percentile, with a ±3-frame spacing rule so near-duplicate
   frames can't stack.
4. **Fit** — fisheye *first*, because it stays well conditioned at wide FOV even with
   coverage holes, then use its `K` to seed the pinhole fit. The pinhole fit releases
   parameters in stages (fixed principal point + `k1` only → tangential → `k3`) and runs
   two rounds with 92nd-percentile outlier rejection.
5. **Validate** — solvePnP against *every* detected frame, not just the ones used to fit,
   reporting RMS / mean / median / p95.
6. **FOV** — inverts the fisheye θ(r) numerically and measures the angle between edge rays.
   Also prints the centred convention and the naive pinhole number for comparison.
7. **Report** — the leaf-tracker bearing model fitted to the lens (seconds: it fits ray
   maps, no calibration reruns) and the error contour maps (reusing the validation
   residuals, so no frame is posed twice). `--no-report` skips it.

### Guards worth knowing about

- **Divergence guard.** If pinhole `fx` and fisheye `fx` disagree by more than 15%, the run
  warns loudly. This is a real failure mode: seeding the pinhole cold on a clip missing the
  top of frame drove `fx` to 419 and produced a 23.9° angular residual that otherwise
  reported as a perfectly normal result.
- **Outlier views poison the seed.** View selection is greedy for image coverage, so it
  favours frames where the board reaches the frame edge at a steep angle — the most
  informative views, and the most often mis-detected. The fisheye seed fit is therefore
  re-run on the views that fit it (percentile *and* a multiple-of-median cut, so a clean
  clip loses nothing). On the 12 mm clip 35 of 500 views, worst 1.1e2 px, took the seed
  from RMS 3.93 to 0.69 and the final fit to 0.59; uniformly sampled halves of the same
  clip had always fitted to 0.7–0.9, which is what gave the cause away.
- **Principal point walking off the sensor.** A staged pinhole fit that frees the
  principal point can push it outside the image, and OpenCV then raises
  `Principal point must be within the image` and kills the run. Each stage is retried
  with the principal point pinned, and failing that the previous stage's result stands.
- **Polynomial invertibility.** The run checks how far out in normalised radius the pinhole
  distortion polynomial stays monotonic and compares that to what the image corners need.
  `CORNERS OUTSIDE MODEL` means the pinhole model is undefined there.
- **Fisheye degeneracy.** `cv2.fisheye.calibrate` asserts on near-degenerate views, so the
  fit retries on progressively cleaner subsets (min corners 25 / 40 / 55) before giving up
  rather than killing the run.
- **Detector parameters are not resolution-scaled.** An earlier version scaled them by
  `round(W/1280)`. Measured at 4K that was strictly worse — 43.1 vs 47.6 mean corners per
  frame, 3 detection failures vs 0. Don't reintroduce it without rerunning
  `detector_test.py`.

## Analysis and verification tools

Each answers one specific question that came up while calibrating.

| script | question |
|---|---|
| `refine_all.py` | Refine intrinsics against **every** detected frame, not 500 views. Joint LM over 4000 views is a 24018² system and runs for days; poses are conditionally independent given the intrinsics, so it alternates pose-solve / Gauss–Newton on the 9 intrinsics instead. Same optimum, linear in frames. Writes `*_all.npz`. |
| `verify.py` | Coverage map, empirical θ(r) vs both models, undistorted sample frames. |
| `error_contour.py` | Re-plot the px / deg error contour maps from saved artefacts; `--calib-tag <tag>-equidist --models fisheye` maps the pure-equidistant fit instead. Run by the pipeline. |
| `azel_model_check.py` | Fit the leaf-tracker bearing model to a saved calibration and update `bearing_model` in `intrinsics_<tag>.json`. Run by the pipeline. |
| `equidistant_check.py` | Can the lens be treated as pure radial `r = f·θ`? Not part of the pipeline; writes `equidistant_<tag>.json`. |
| `ray_stability.py` | Do independent folds agree on the pixel→ray map, and with the main fit? Fold-to-fold is statistical noise; fold-vs-main is what a different choice of views does to the answer. Not part of the pipeline; writes `stability_<tag>.json`. |
| `error_distribution.py` | Reprojection error magnitude *and* where in the frame it lands, pinhole vs fisheye. |
| `pp_uncertainty.py` | Is the principal-point offset real or fit noise? K interleaved folds calibrated independently; compare the spread to the measured offset. |
| `angle_consistency.py` | Do two calibrations describe the same optics? The angle between two 3D rays is rotation-invariant, so matched features across two clips must give the same angular separation through either calibration. A wrong focal length shows up as a fixed factor. |
| `fov_ratio.py` | Empirical FOV ratio between two capture modes from SIFT correspondences alone, no calibration involved — near the axis `r_b ≈ (f_b/f_a)·r_a`. |
| `fov_visual.py` | Remaps every mode into one shared equidistant projection so FOVs can be compared by eye regardless of where the camera was pointing. |
| `crop_experiment/crop_experiment.py` | Ground truth for the crop-factor argument: apply *known* centre crops to one video, calibrate each, check `fx' = s·fx` and `cx' = s·(cx - x0)`. |
| `detector_test.py` | Measures whether detector-parameter scaling helps at 4K. It doesn't. |
| `diag_detection_rate.py` | Splits a low-detection-rate clip into causes: marker detection vs ChArUco interpolation vs board-config mismatch. |

Scripts other than `pipeline.py` expect to run in the directory holding the `detections_*`
and `calib_*` artefacts, and several default to `--square 0.015 --marker 0.011` rather than
the 8 mm / 6 mm default in `pipeline.py` — pass them explicitly.

## Not included

Videos, extracted frames, and `.npy` / `.npz` artefacts are gitignored; a single 4K run is
several GB of JPEGs.
