# ring-finder

Detect rings on the conveyor and output each one's **robot XY (mm)** for
pick-and-place — robust to the textured mesh background (FastSAM
segmentation, not HoughCircles).

## Project structure

```
ring-finder/
├── scripts/               # all runnable tools (keep them together)
│   ├── gui.py             #  GUI: watch folder, live image + results table
│   ├── ringworker.py      #  background detection worker used by the GUI
│   ├── ring_finder.py     #  MAIN (CLI): detect + robot XY, one run
│   ├── watch_folder.py    #  live CLI: watch a folder, measure each new image
│   ├── robot_map.py       #  fit the pixel -> robot XY homography
│   ├── ai.py              #  detection only (pixels)
│   ├── calibrate.py       #  camera calibration from a ChArUco board
│   ├── pixel_to_mm.py     #  single pixel -> mm helpers
│   └── rings_to_mm.py     #  batch *_rings.json -> mm
├── config/
│   └── robot_map.json     # fitted pixel->robot homography (auto-loaded)
├── data/
│   ├── incoming/          # camera drops .bmp here  (watched)
│   ├── calibration/       # ChArUco board photos for calibrate.py
│   └── robot_xy.csv       # robot coords used to fit robot_map.json
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt          # first run downloads FastSAM (~138 MB)
```

## Better accuracy: a custom model

FastSAM is generic and works out of the box, but a model trained on your
washers is more accurate and far faster. See `training/README_train.md`:
bootstrap labels from FastSAM (`training/pseudolabel.py`), correct them,
train (`training/train_yolo.py`), then set MODEL = your `best.pt` and Model
type = `yolo` in the app.

## Build a Windows .exe (ring_app)

On a Windows machine with Python and the deps installed, from the project root:

```bat
build_exe.bat
```

(or `pip install pyinstaller` then `pyinstaller ring_app.spec`). Output is
`dist\ring_app\ring_app.exe` — copy the whole `dist\ring_app` folder to the
target PC. When frozen, the app keeps `config\` and `data\` **next to the
exe**, so settings (`config\app_config.json`) and results persist and stay
editable. The FastSAM weight isn't bundled: on first run it downloads, or drop
`FastSAM-x.pt` beside the exe, or use the **Download** button on the
Configuration → Detection tab (fetches the model named in MODEL into the app
folder — handy for an offline PC: download once on a connected machine and
copy the `.pt` over).

**Run `dist\ring_app\ring_app.exe`, not the copy under `build\`** (`build\`
is only PyInstaller's scratch folder). Keep the exe next to its `_internal\`
folder. `ring_app.spec` ships with `console=True` so a terminal shows any
startup error on the first build; flip it to `False` for the final windowed
build once it runs.

**"Failed to load Python DLL python3xx.dll — the specified module could not
be found"** means the target PC is missing the Microsoft Visual C++
Redistributable (x64) — install it and retry:
https://aka.ms/vs/17/release/vc_redist.x64.exe

## Quick start — GUI (easiest)

```bash
python scripts/gui.py            # or open scripts/gui.py in PyCharm and press Run
```

The window: set the **Watch folder** (defaults to `data/incoming/`), press
**Start**, and every new `.bmp` the camera drops in is detected and shown
with its ring(s) circled, a results table (pixel + robot XY), and a log.
**Open image...** runs a single file on demand. The model loads once on a
background thread, so the window stays responsive.

## Standalone configurable app (`scripts/ring_app.py`)

A self-contained operator app (independent of the other scripts) with three
tabs:

- **Live** — Start/Stop watching, annotated image (**zoomable**: +/-, Fit,
  100%, mouse wheel; scrollbars when enlarged), results table, log, a live
  **TCP connection status**, and a banner that turns red **CONVEYOR EMPTY**
  when no ring is found. Shows the **processing time** per image (detection
  ms and total ms). Click a table column header to re-sort the view.

  > Timing note: with the tuned default `imgsz=640`, `FastSAM-x` is
  > ~0.8–0.9 s/image on CPU (vs ~2.5 s at 1024) and detections are actually
  > more repeatable, because the source is only ~320×240 - larger imgsz just
  > adds interpolation noise. Tune `imgsz` on the Detection tab.
- **Configuration** — sub-tabs (General / Detection / Output / TCP / Robot)
  with a fixed Save bar; **starts watching automatically on open** (toggle in
  General). All settings saved to `config/app_config.json`:
  watch folder, image extension, **filename pattern (glob)**, poll time,
  output CSV (append log) and a **latest CSV** (overwritten each image with
  only the current readings), **XY offset (mm)**,
  **auto-delete images older than N min**, calibration (homography) file, the
  the detection **model** (FastSAM, a YOLO-seg, or a custom-trained `.pt` — a
  **Model type** selector picks the right loader) with a **sub-pixel circle
  fit** toggle for tighter diameters, plus `conf, iou, min/max_area_frac,
  min_circ, min_radius_frac`, **frame averaging** (mean of N frames of a
  stationary part, to cut noise), an optional **inner-diameter (hole)**
  measurement (best-effort, image-based), a **TCP server** (host/port/line
  format), and
  **output ordering** (sort rings by `y`, `x`, or `diameter`, asc/desc — sets
  the ring-id order used in the table, CSV and TCP).
- **Calibration** — build the pixel→robot homography in-app: **Grab latest
  from folder** or **Load image…** → the ring pixel is detected → set the
  **Robot X/Y** for that ring (type it manually, or tick **Read robot position
  (TCP)** to continuously read the robot's live position — the app sends the
  request command like `STP` and parses `X=xx.xx,Y=yy.yy`, showing it live —
  then press **Update** to copy the live value into the fields) → **Add
  point**; collect ≥4 points, then
  **Fit & Save** writes the map (with in-fit and leave-one-out RMS, and
  RANSAC-flagged bad points). Or **Batch (folder + Excel)**: point it at a
  folder of calibration images plus an Excel/CSV table (columns for **X**,
  **Y**, **image name**) and it detects each ring, pairs it with the robot XY,
  fits, and writes the calibration file automatically. A calibration image
  should hold **one** ring; if any has several, an **if many rings** selector
  chooses `largest` / `center` (nearest image centre) / `skip`, and the run
  reports which images had multiples. Choose the **Model**: `homography` (perspective,
  8 DOF), `affine` (6 DOF), or `similarity` (rotation+uniform scale+translation
  via sin/cos, 4 DOF) - similarity is most robust when the camera views a flat
  plane squarely and often gives the lowest leave-one-out error.

  **Coverage map** shows where the current calibration is strong (green,
  near points) vs weak (red, far from any point) with each point coloured by
  its error - so you know where to add points. **Load points from map** pulls
  an existing map's points back into the table so you can add more and re-fit
  (**Fit & Save** overwrites the same file to *improve it*, or **Save as...**
  makes a new one).

  Buttons: **Save settings**, **Save & Restart** / **Restart app** (relaunch
  so a new model reloads), and **Export config… / Import config…** (share a
  full settings file between machines; import keeps only known keys).

```bash
python scripts/ring_app.py
```

Behaviour notes:
- **Filename pattern** (glob) limits which files are processed:
  `*` = any image; `202*_*.bmp` or `????????_??????.bmp` = only timestamp
  names like `20260830_154619.bmp`; or an exact name like `image.bmp`.
- **Watcher re-fires on a changed file too**, not only new names — so a camera
  that overwrites one fixed filename still triggers each frame (detects new
  name **or** changed modification time; waits for the file to stop changing
  first).
- **TCP server**: enable it and the robot connects as a client; each detection
  is sent as `{id},{x},{y}` lines (format configurable with
  `{id} {x} {y} {dia} {image}`). When no ring is found, the configurable
  **empty message** (default `EMPTY`) is sent instead, so the controller knows
  the conveyor is empty.
- **Robot XY = homography(pixel) + (offset_x, offset_y)**; written to the CSV
  and broadcast over TCP.

## Quick start — command line

```bash
# one folder / one image (auto-loads config/robot_map.json)
python scripts/ring_finder.py data/incoming/
python scripts/ring_finder.py path/to/image.bmp
python scripts/ring_finder.py data/incoming/ --no-robot   # pixels only

# live: watch data/incoming/ and measure every new .bmp automatically
python scripts/watch_folder.py                 # uses data/incoming by default
python scripts/watch_folder.py /some/other/dir --poll 0.5
```

Each image produces `<image>_rings.png` (annotated) and `<image>_rings.json`
(pixel + `robot_x`/`robot_y`); the watcher also appends a timestamped row per
ring to `results.csv`. Paths are anchored to the project root, so it works
whether you Run from PyCharm or the command line. A file is processed only
after its size stops growing, so half-written images aren't read early.

## Recalibrate after any camera or robot move

```bash
# 1. jog the robot to a few rings, record id,x,y in data/robot_xy.csv
# 2. capture one image per ring (1.bmp, 2.bmp, ...) and detect them
python scripts/ai.py calib_shots/
# 3. fit the new map
python scripts/robot_map.py fit --rings calib_shots/ --robot data/robot_xy.csv \
       --out config/robot_map.json
```

`fit` prints an in-fit and a leave-one-out RMS, and RANSAC-flags any point
whose recorded robot XY doesn't fit. `ring_finder.py`/`watch_folder.py`
pick up `config/robot_map.json` automatically on the next run.

---

## camera calibration

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

### Batch: convert ai.py detections to mm (`rings_to_mm.py`)

`rings_to_mm.py` reads the `*_rings.json` files from `ai.py`, converts every
ring's centre and outer diameter to mm, and writes `x_mm`, `y_mm`,
`diameter_mm` back into each JSON. Pick one scale source:

```bash
# best: homography from a board photo taken flat on the belt
python rings_to_mm.py calibration.json parts/ --map belt_map.json

# perpendicular camera at a known working distance (mm)
python rings_to_mm.py calibration.json parts/ --Z 300

# calibrate the scale from one ring whose true outer diameter you know
python rings_to_mm.py calibration.json parts/ \
       --ref-mm 24 --ref-json parts/first_rings.json --ref-id 1

# measured field of view: the real WIDTH x HEIGHT (mm) the image covers
# at the belt surface (origin defaults to image centre)
python rings_to_mm.py calibration.json parts/ --frame-mm 200 150
python rings_to_mm.py calibration.json parts/ --frame-mm 200 150 --origin topleft
```

`--frame-mm` assumes the camera looks straight at a flat belt (uniform
mm/px); the calibration is still used to undistort each point first.

## Pixel → robot XY for pick-and-place (`robot_map.py`)

If a robot picks the rings, map the pixel straight to the robot's XY frame.
The camera views a flat plane, so one homography does it — no intrinsics
file needed (it absorbs scale, perspective and the tiny distortion).

Calibrate once from a few rings whose robot XY you jogged to and recorded
(one ring per numbered image, `1.bmp`, `2.bmp`, …; `robot_xy.csv` is
`id,x,y`), then apply to every new detection:

```bash
python ai.py calib_shots/                 # detect -> <id>_rings.json
python robot_map.py fit --rings calib_shots/ --robot robot_xy.csv --out robot_map.json
python robot_map.py apply robot_map.json --rings newparts/   # adds robot_x/y
python robot_map.py apply robot_map.json --u 210 --v 128     # one pixel
```

`fit` reports the in-fit RMS **and a leave-one-out RMS** (the honest error
on an unseen ring), and RANSAC-flags any point whose recorded robot XY
doesn't fit — printing what the map expected, so a mistyped coordinate is
caught instead of poisoning the whole map.

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
