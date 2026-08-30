# ============================================================
# ring_finder.py  -  detect rings and output robot XY
# ------------------------------------------------------------
# One-step production script for pick-and-place:
#   1. finds every ring by FastSAM segmentation (robust to the textured
#      mesh background that defeats HoughCircles), then
#   2. maps each ring's pixel centre straight to the robot's XY frame (mm)
#      using a hand-eye homography.
#
# Install once:  pip install ultralytics opencv-python numpy
# Run:           python ring_finder.py image.bmp
#                python ring_finder.py folder/
#                python ring_finder.py folder/ --map robot_map.json
#
# Outputs per image:
#   <image>_rings.png    annotated picture (pixel centre + robot XY)
#   <image>_rings.json   per-ring pixel + robot XY, for the robot/LabVIEW
#
# The robot map:
#   A default homography (fitted from the 9-point robot_xy calibration) is
#   built in, so this runs as-is. Re-calibrate whenever the camera or robot
#   moves, with robot_map.py, and pass the new file via --map:
#       python robot_map.py fit --rings calib/ --robot robot_xy.csv \
#              --out robot_map.json
#       python ring_finder.py parts/ --map robot_map.json
#   Pass --no-robot to output pixels only.
#
# Detection tuning (CONFIG below): CONF/IOU sensitivity, MIN/MAX_AREA_FRAC
# size gate, MIN_CIRC roundness, MIN_RADIUS_FRAC smallest ring.
# ============================================================

import cv2
import numpy as np
import sys
import os
import glob
import json
import argparse
from ultralytics import FastSAM

# -------- CONFIG --------
MODEL           = "FastSAM-x.pt"     # or "FastSAM-s.pt" (faster, lighter)
CONF, IOU       = 0.20, 0.7
MIN_AREA_FRAC   = 0.004
MAX_AREA_FRAC   = 0.25
MIN_CIRC        = 0.75
MIN_RADIUS_FRAC = 0.03
# ------------------------

# Default pixel -> robot XY homography (fitted from the 9-point robot_xy
# calibration; leave-one-out accuracy ~2.5 mm). Regenerate with robot_map.py
# and override with --map after any camera/robot move.
DEFAULT_ROBOT_H = [
    [1.0434843407911647,  -0.01545154100766795, -86.26829179471179],
    [0.0038693571824632523, 1.0846467959620896,  220.35843122700368],
    [-4.50499731203967e-05, 7.468832723876367e-05, 1.0],
]

_model = None

# project root = folder above this scripts/ dir
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MAP_FILE = os.path.join(ROOT, "config", "robot_map.json")


def get_model():
    global _model
    if _model is None:
        _model = FastSAM(MODEL)
    return _model


def load_homography(map_path):
    # explicit path wins; else config/robot_map.json; else the built-in map
    if map_path is None and os.path.exists(DEFAULT_MAP_FILE):
        map_path = DEFAULT_MAP_FILE
    if map_path is None:
        return np.array(DEFAULT_ROBOT_H, dtype=np.float64)
    return np.array(json.load(open(map_path))["H"], dtype=np.float64)


def px_to_robot(x, y, H):
    q = cv2.perspectiveTransform(
        np.array([[x, y]], np.float64).reshape(-1, 1, 2), H).reshape(2)
    return float(q[0]), float(q[1])


def detect_rings(path, H=None):
    img = cv2.imread(path)
    if img is None:
        print("cannot read " + path)
        return []
    Hh, W = img.shape[:2]
    min_radius = max(5, (Hh * W) ** 0.5 * MIN_RADIUS_FRAC)

    res = get_model()(path, device="cpu", retina_masks=True,
                      imgsz=1024, conf=CONF, iou=IOU, verbose=False)[0]
    if res.masks is None:
        print(os.path.basename(path) + ": no segments")
        _write_json(path, W, Hh, [], H)
        return []
    masks = res.masks.data.cpu().numpy()

    rings = []
    for m in masks:
        mask = (m > 0.5).astype(np.uint8)
        a = int(mask.sum())
        if a < MIN_AREA_FRAC * Hh * W or a > MAX_AREA_FRAC * Hh * W:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * cv2.contourArea(c) / (peri * peri) if peri else 0
        if circ < MIN_CIRC or r < min_radius:
            continue
        if any((x - rx) ** 2 + (y - ry) ** 2 < (0.5 * max(r, rr)) ** 2
               for rx, ry, rr in rings):
            continue
        rings.append((x, y, r))

    rings = sorted(rings, key=lambda t: (t[1], t[0]))   # row-by-row order

    # annotated image
    vis = img.copy()
    for i, (x, y, r) in enumerate(rings):
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        label = str(i + 1)
        if H is not None:
            rx, ry = px_to_robot(x, y, H)
            label += " (%.1f,%.1f)" % (rx, ry)
        cv2.putText(vis, label, (int(x) - 10, int(y) - int(r) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    out_png = os.path.splitext(path)[0] + "_rings.png"
    cv2.imwrite(out_png, vis)

    _write_json(path, W, Hh, rings, H)

    # console
    print("")
    print(os.path.basename(path) + " - " + str(len(rings)) + " rings")
    for i, (x, y, r) in enumerate(rings):
        line = ("  #%d  center=(%d,%d) px  radius=%d  diameter=%d"
                % (i + 1, int(x), int(y), int(r), int(2 * r)))
        if H is not None:
            rx, ry = px_to_robot(x, y, H)
            line += "  robot=(%.2f,%.2f) mm" % (rx, ry)
        print(line)
    print("  -> " + out_png)
    print("  -> " + os.path.splitext(path)[0] + "_rings.json")
    return rings


def _write_json(path, W, Hh, rings, H):
    out_rings = []
    for i, (x, y, r) in enumerate(rings):
        d = {
            "id": i + 1,
            "x": round(float(x), 1),
            "y": round(float(y), 1),
            "radius": round(float(r), 1),
            "diameter": round(float(2 * r), 1),
        }
        if H is not None:
            rx, ry = px_to_robot(x, y, H)
            d["robot_x"] = round(rx, 3)
            d["robot_y"] = round(ry, 3)
        out_rings.append(d)
    data = {
        "image": os.path.basename(path),
        "width": int(W),
        "height": int(Hh),
        "count": len(rings),
        "frame": "robot_xy_mm" if H is not None else "pixels",
        "rings": out_rings,
    }
    with open(os.path.splitext(path)[0] + "_rings.json", "w") as f:
        json.dump(data, f, indent=2)


def main():
    ap = argparse.ArgumentParser(
        description="Detect rings and output robot XY.")
    ap.add_argument("input", nargs="?", default=".",
                    help="image file or folder (default: current dir)")
    ap.add_argument("--map", default=None,
                    help="robot_map.json from robot_map.py "
                         "(default: built-in homography)")
    ap.add_argument("--no-robot", action="store_true",
                    help="output pixels only, skip robot XY")
    args = ap.parse_args()

    H = None if args.no_robot else load_homography(args.map)

    if os.path.isdir(args.input):
        exts = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")
        seen = set()
        for ext in exts:
            for p in sorted(glob.glob(os.path.join(args.input, "*." + ext))):
                # skip our own annotated outputs
                if p.endswith("_rings.png") or p in seen:
                    continue
                seen.add(p)
                detect_rings(p, H)
    else:
        detect_rings(args.input, H)


if __name__ == "__main__":
    main()
