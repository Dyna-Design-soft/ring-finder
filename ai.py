# ============================================================
# AI Ring Detection with FastSAM  (segment-anything)
# ------------------------------------------------------------
# Finds every ring by neural-network segmentation, then filters
# the masks for ring-shaped ones. Robust to textured/mesh
# backgrounds that defeat HoughCircles.
#
# Install once:   pip install ultralytics opencv-python numpy
# Run:            python ai.py image.bmp
#                 python ai.py folder/
#
# Outputs per image:
#   <image>_ai_rings.png    annotated picture
#   <image>_rings.json      ring positions (for LabVIEW / other software)
#
# Weights (FastSAM-x.pt, ~138 MB) auto-download on first run.
# Use "FastSAM-s.pt" (~23 MB) for a faster, lighter model.
#
# Tuning (edit the CONFIG values below):
#   CONF / IOU        segmentation sensitivity (lower CONF = more masks)
#   MIN_AREA_FRAC     ignore masks smaller than this fraction of image
#   MAX_AREA_FRAC     ignore masks larger than this
#   MIN_CIRC          minimum circularity to count as a ring (1.0 = perfect)
#   MIN_RADIUS_FRAC   min ring radius as fraction of image diagonal
# ============================================================

import cv2, numpy as np, sys, os, glob, json
from ultralytics import FastSAM

# -------- CONFIG --------
MODEL           = "FastSAM-x.pt"     # or "FastSAM-s.pt" (faster)
CONF, IOU       = 0.20, 0.7
MIN_AREA_FRAC   = 0.004
MAX_AREA_FRAC   = 0.25
MIN_CIRC        = 0.75
MIN_RADIUS_FRAC = 0.03               # auto-scales to any resolution
# ------------------------

_model = None
def get_model():
    global _model
    if _model is None:
        _model = FastSAM(MODEL)
    return _model


def detect_rings(path):
    img = cv2.imread(path)
    if img is None:
        print("cannot read " + path)
        return []
    H, W = img.shape[:2]
    min_radius = max(5, (H * W) ** 0.5 * MIN_RADIUS_FRAC)

    res = get_model()(path, device="cpu", retina_masks=True,
                      imgsz=1024, conf=CONF, iou=IOU, verbose=False)[0]
    if res.masks is None:
        print(os.path.basename(path) + ": no segments")
        _write_json(path, W, H, [])
        return []
    masks = res.masks.data.cpu().numpy()

    rings = []
    for m in masks:
        mask = (m > 0.5).astype(np.uint8)
        a = int(mask.sum())
        if a < MIN_AREA_FRAC * H * W or a > MAX_AREA_FRAC * H * W:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * cv2.contourArea(c) / (peri * peri) if peri else 0
        if circ < MIN_CIRC or r < min_radius:
            continue
        if any((x - rx) ** 2 + (y - ry) ** 2 < (0.5 * max(r, rr)) ** 2 for rx, ry, rr in rings):
            continue
        rings.append((x, y, r))

    rings = sorted(rings, key=lambda t: (t[1], t[0]))   # row-by-row order

    # annotated image
    vis = img.copy()
    for i, (x, y, r) in enumerate(rings):
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
        cv2.putText(vis, str(i + 1), (int(x) - 10, int(y) - int(r) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    out_png = os.path.splitext(path)[0] + "_ai_rings.png"
    cv2.imwrite(out_png, vis)

    # JSON output
    _write_json(path, W, H, rings)

    # console
    print("")
    print(os.path.basename(path) + " - " + str(len(rings)) + " rings")
    for i, (x, y, r) in enumerate(rings):
        print("  #" + str(i + 1) + " center=(" + str(int(x)) + "," + str(int(y)) +
              ") px  radius=" + str(int(r)) + "  diameter=" + str(int(2 * r)))
    print("  -> " + out_png)
    print("  -> " + os.path.splitext(path)[0] + "_rings.json")
    return rings


def _write_json(path, W, H, rings):
    data = {
        "image": os.path.basename(path),
        "width": int(W),
        "height": int(H),
        "count": len(rings),
        "rings": [
            {
                "id": i + 1,
                "x": round(float(x), 1),
                "y": round(float(y), 1),
                "radius": round(float(r), 1),
                "diameter": round(float(2 * r), 1)
            }
            for i, (x, y, r) in enumerate(rings)
        ]
    }
    out_json = os.path.splitext(path)[0] + "_rings.json"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "."
    if os.path.isdir(arg):
        for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff"):
            for p in sorted(glob.glob(os.path.join(arg, "*." + ext))):
                detect_rings(p)
    else:
        detect_rings(arg)
