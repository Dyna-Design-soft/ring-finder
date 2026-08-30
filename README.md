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

## Using the result to fix pixel → mm

Once you have a trustworthy `calibration.json`, undistort a ring's pixel
coordinate before converting to millimetres:

```python
import json, numpy as np, cv2

cal = json.load(open("calibration.json"))
K = np.array(cal["camera_matrix"])
dist = np.array(cal["distortion_coefficients"])

def undistort_px(x, y):
    """Map a raw pixel (x, y) to an undistorted pixel in the same frame."""
    pt = np.array([[[float(x), float(y)]]], dtype=np.float64)
    out = cv2.undistortPoints(pt, K, dist, P=K)
    return float(out[0, 0, 0]), float(out[0, 0, 1])
```

To get **millimetres**, you still need the scale of the plane the rings lie
on. If that plane is the calibration board's plane at a known distance, use
its pose (a homography from board mm ↔ image px) to map undistorted pixels to
mm. Measuring positions from the calibrated optical centre `(cx, cy)` — not
the image centre — and undistorting first removes the off-centre bias you are
seeing.
