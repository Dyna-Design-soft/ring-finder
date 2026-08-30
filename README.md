# ring-finder — camera calibration

`calibrate.py` computes a camera's **intrinsics** (focal length + the true
optical centre) and **lens-distortion** coefficients from photos of a ChArUco
calibration board, and writes them to `calibration.json` / `calibration.npz`.

## Why this exists

When you detect a ring and read its pixel position, converting that to
millimetres only works if you know two things the raw image can't give you:

1. **The true optical centre `(cx, cy)`.** A webcam's optical centre is *not*
   the image centre `(W/2, H/2)`; it is usually offset by several pixels. If
   you measure distances from the image centre, everything is biased.
2. **The lens distortion.** Cheap lenses bend straight lines and push
   off-centre points outward (barrel distortion). The further a ring is from
   the middle of the frame, the more its measured position drifts from the
   truth.

Assuming the image centre and ignoring distortion is exactly what makes a
ring's position look correct in the middle of the frame but wrong toward the
edges. Calibration measures both so you can *undistort* a pixel coordinate
before scaling it to mm.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python calibrate.py Cali/                       # uses the folder's images
python calibrate.py Cali/ --square-mm 20        # set your printed square size
python calibrate.py Cali/ --square-mm 25 --marker-mm 17
```

Outputs (written next to the images, or use `--out DIR`):

- `calibration.json` — human-readable intrinsics, distortion, and a **quality
  verdict**.
- `calibration.npz` — the same `camera_matrix` and `distortion_coefficients`
  for numpy/OpenCV.
- `undistorted/` — before/after previews (skip with `--no-previews`).

Board defaults: **10×7 squares, `DICT_4X4_50`, marker ≈ 0.68 × square**
(matches the sample `Cali` set). Override with `--squares-x/-y` and `--dict`.

> The physical `--square-mm` only scales real-world distances. The intrinsics
> and distortion — the part that fixes the "position drifts off-centre"
> problem — do not depend on it, so undistortion works even before you know
> the exact square size. Set it correctly when you need absolute millimetres.

## Check a set before you trust it (`--check`)

Run this first on any new set — it reports each image's out-of-plane board
tilt and writes an annotated `tilt_check.png`, without calibrating:

```bash
python calibrate.py yourfolder/ --check
```

```
board tilt (deg): min 4.7  mean 23.6  max 38.1  (90th 31.5)
VERDICT: good tilt variety.
```

If it says **NOT ENOUGH TILT**, re-shoot before going further (see below).

## Read the quality verdict — this is the important part

A tiny reprojection error does **not** mean the calibration is usable. The
tool checks the geometry and prints one of:

- **GOOD** — geometry and error both look sane.
- **QUESTIONABLE** — usable but could be better (e.g. limited board tilt).
- **UNRELIABLE** — the numbers are not trustworthy; do not use them.

### The `Cali/` sample set is UNRELIABLE — and shows the classic mistake

Every image in the provided `Cali/` set has the board **flat and facing the
camera** (measured tilt ≈ 0–3°). Camera calibration is mathematically
*unobservable* from head-on views only: you cannot separate focal length from
distance, and the optical centre is barely constrained. The optimiser then
lands on nonsense (an optical centre outside the sensor, an impossible field
of view) while still reporting a small reprojection error. **A calibration
built from that set is why pixel→mm and the centre position come out wrong.**

### How to shoot a set that calibrates well

Take ~15–25 photos of the board and **vary the pose a lot**:

- **Tilt the board 20–45°** — left, right, up, down, and diagonally. This is
  the single most important thing; flat-on shots add almost nothing.
- **Move the board around the frame** — centre, all four corners, edges — so
  distortion is sampled across the whole sensor.
- **Vary the distance** — some near (board filling most of the frame), some
  far.
- Keep the board rigid and flat, well lit, and in focus; avoid motion blur.

Re-run until the verdict is **GOOD**.

## Convert pixel → mm (`pixel_to_mm.py`)

The calibration maps a pixel to a **direction**, not a distance. To get
millimetres you must add the **scale** of the plane the rings lie on. Two
ways, both provided by `pixel_to_mm.py` (each undistorts the pixel first):

### A) Homography — recommended, most accurate

Put the ChArUco board **flat in the same plane as your rings**, take one
reference photo, and build a pixel→mm map from it. Exact even if the camera
is not perfectly perpendicular; no distance measurement needed. Ideal for a
fixed camera + plane rig — do it once, reuse the map for every ring image.

```bash
# one-time: build the map (square-mm = your printed square size)
python pixel_to_mm.py make-map calibration.json board_in_plane.bmp \
       --square-mm 20 --out plane_map.json

# convert a ring's pixel position
python pixel_to_mm.py to-mm calibration.json --map plane_map.json --u 210 --v 128
```

`make-map` prints a fit residual; on the sample data it is ~0.17 mm, and
detected board corners land on their true mm grid to that accuracy. If the
residual is large, re-take the board photo flatter and sharper.

### B) Perpendicular pinhole — quick, needs a known distance

If the camera looks straight down at the plane from a known distance `Z` mm:

```bash
python pixel_to_mm.py to-mm calibration.json --Z 300 --u 210 --v 128
```

`X_mm = (u − cx)·Z/fx`, `Y_mm = (v − cy)·Z/fy`. Only correct when the camera
is truly perpendicular and `Z` is known; the homography handles tilt for you.

### In your own code

```python
import pixel_to_mm as p2m
K, dist = p2m.load_calibration("calibration.json")

# mode A:
import json; H = json.load(open("plane_map.json"))["H"]
x_mm, y_mm = p2m.px_to_mm_homography(u, v, K, dist, H)

# mode B:
x_mm, y_mm = p2m.px_to_mm_perpendicular(u, v, K, dist, Z_mm=300)
```

> The common mistake that makes positions "drift away from the centre" is
> measuring from the image centre `(W/2, H/2)` with a single mm-per-pixel
> constant. Measure from the calibrated optical centre `(cx, cy)`, undistort
> first, and use a homography (not one global scale) whenever the plane is
> tilted or the distance varies.
