# ============================================================
# Convert ai.py ring detections (pixels) to millimetres
# ------------------------------------------------------------
# Reads the *_rings.json files written by ai.py, converts each ring's
# centre and diameter to millimetres using the camera calibration plus a
# SCALE source, and writes the mm values back into the JSON (and prints
# them). ai.py is not modified.
#
# The calibration alone cannot give mm - it maps a pixel to a direction,
# not a distance. You must also tell it how big a pixel is at the belt
# surface, in ONE of three ways:
#
#   --map plane_map.json   (BEST) homography from a ChArUco board photo
#                          taken flat on the belt (see pixel_to_mm.py
#                          make-map). Exact, handles camera tilt.
#
#   --Z 300                perpendicular pinhole: camera looks straight
#                          down at the belt from 300 mm. Simple; only
#                          correct if the camera really is perpendicular.
#
#   --ref-mm 24.0 --ref-json img_rings.json --ref-id 1
#                          calibrate the scale from ONE ring whose true
#                          outer diameter you know (24.0 mm here). Assumes
#                          a flat plane at one distance (uniform mm/px).
#
# Usage:
#   python rings_to_mm.py calibration.json parts_folder/ --Z 300
#   python rings_to_mm.py calibration.json parts_folder/ --map belt_map.json
#   python rings_to_mm.py calibration.json parts_folder/ \
#          --ref-mm 24 --ref-json parts_folder/20260724_120716_rings.json --ref-id 1
# ============================================================

import cv2
import numpy as np
import json
import os
import glob
import argparse


def load_calibration(path):
    with open(path) as f:
        cal = json.load(f)
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    dist = np.array(cal["distortion_coefficients"], dtype=np.float64)
    return K, dist


def undistort_px(pts, K, dist):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)


def _rim_points(x, y, r):
    """Four rim points (right, left, bottom, top) of a circle."""
    return [(x + r, y), (x - r, y), (x, y + r), (x, y - r)]


# ---- scale backends: each returns (center_mm, outer_diameter_mm) ----

def convert_homography(x, y, r, K, dist, H):
    H = np.asarray(H, dtype=np.float64)
    pts = np.array([(x, y)] + _rim_points(x, y, r), dtype=np.float64)
    und = undistort_px(pts, K, dist).reshape(-1, 1, 2)
    mm = cv2.perspectiveTransform(und, H).reshape(-1, 2)
    c = mm[0]
    dia = 0.5 * (np.linalg.norm(mm[1] - mm[2]) + np.linalg.norm(mm[3] - mm[4]))
    return (float(c[0]), float(c[1])), float(dia)


def convert_perpendicular(x, y, r, K, dist, Z):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    pts = np.array([(x, y)] + _rim_points(x, y, r), dtype=np.float64)
    und = undistort_px(pts, K, dist)
    XY = np.column_stack([(und[:, 0] - cx) * Z / fx, (und[:, 1] - cy) * Z / fy])
    c = XY[0]
    dia = 0.5 * (np.linalg.norm(XY[1] - XY[2]) + np.linalg.norm(XY[3] - XY[4]))
    return (float(c[0]), float(c[1])), float(dia)


def convert_scale(x, y, r, K, dist, mm_per_px, cx0, cy0):
    """Uniform-scale fallback: mm measured from the optical centre."""
    und = undistort_px([(x, y)], K, dist)[0]
    c = ((und[0] - cx0) * mm_per_px, (und[1] - cy0) * mm_per_px)
    return (float(c[0]), float(c[1])), float(2 * r * mm_per_px)


def convert_frame(x, y, r, K, dist, sx, sy, ox, oy):
    """Frame-extent scale: separate mm/px in x and y from a measured
    field-of-view, mm measured from the origin pixel (ox, oy)."""
    und = undistort_px([(x, y)], K, dist)[0]
    c = ((und[0] - ox) * sx, (und[1] - oy) * sy)
    dia = r * (sx + sy)              # 2*r * mean(sx, sy)
    return (float(c[0]), float(c[1])), float(dia)


def process(calib, folder, mode, H=None, Z=None, mm_per_px=None, frame=None):
    K, dist = load_calibration(calib)
    cx0, cy0 = K[0, 2], K[1, 2]
    jsons = sorted(glob.glob(os.path.join(folder, "*_rings.json")))
    if not jsons:
        print("no *_rings.json files in " + folder)
        return
    for jp in jsons:
        data = json.load(open(jp))
        for ring in data.get("rings", []):
            x, y, r = ring["x"], ring["y"], ring["radius"]
            if mode == "map":
                (mx, my), dmm = convert_homography(x, y, r, K, dist, H)
            elif mode == "Z":
                (mx, my), dmm = convert_perpendicular(x, y, r, K, dist, Z)
            elif mode == "frame":
                (mx, my), dmm = convert_frame(x, y, r, K, dist, *frame)
            else:
                (mx, my), dmm = convert_scale(x, y, r, K, dist, mm_per_px, cx0, cy0)
            ring["x_mm"] = round(mx, 3)
            ring["y_mm"] = round(my, 3)
            ring["diameter_mm"] = round(dmm, 3)
        data["units"] = {"mode": mode}
        with open(jp, "w") as f:
            json.dump(data, f, indent=2)
        print(os.path.basename(jp) + ":")
        for ring in data.get("rings", []):
            print("  #%d  centre=(%.2f, %.2f) mm  outer_dia=%.2f mm"
                  % (ring["id"], ring["x_mm"], ring["y_mm"], ring["diameter_mm"]))


def main():
    ap = argparse.ArgumentParser(
        description="Convert ai.py ring pixels to mm.")
    ap.add_argument("calibration")
    ap.add_argument("folder", help="folder of *_rings.json from ai.py")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--map", help="plane_map.json (homography) - most accurate")
    g.add_argument("--Z", type=float, help="working distance mm (perpendicular)")
    g.add_argument("--ref-mm", type=float,
                   help="true outer diameter (mm) of a known reference ring")
    g.add_argument("--frame-mm", type=float, nargs=2, metavar=("W", "H"),
                   help="measured real WIDTH and HEIGHT (mm) that the whole "
                        "image covers at the belt surface")
    ap.add_argument("--ref-json", help="the *_rings.json holding the reference ring")
    ap.add_argument("--ref-id", type=int, help="id of the reference ring")
    ap.add_argument("--origin", choices=["center", "topleft", "optical"],
                    default="center",
                    help="where mm (0,0) is: image center (default), "
                         "top-left corner, or the calibrated optical centre")
    args = ap.parse_args()

    if args.map:
        H = json.load(open(args.map))["H"]
        process(args.calibration, args.folder, "map", H=H)
    elif args.Z is not None:
        process(args.calibration, args.folder, "Z", Z=args.Z)
    elif args.frame_mm is not None:
        W_mm, H_mm = args.frame_mm
        K, _ = load_calibration(args.calibration)
        # image size: read from any rings.json (has width/height)
        anyj = sorted(glob.glob(os.path.join(args.folder, "*_rings.json")))
        if not anyj:
            ap.error("no *_rings.json in " + args.folder)
        meta = json.load(open(anyj[0]))
        wpx, hpx = meta["width"], meta["height"]
        sx, sy = W_mm / wpx, H_mm / hpx
        if args.origin == "topleft":
            ox, oy = 0.0, 0.0
        elif args.origin == "optical":
            ox, oy = float(K[0, 2]), float(K[1, 2])
        else:
            ox, oy = (wpx - 1) / 2.0, (hpx - 1) / 2.0
        print("frame scale: %.4f mm/px (x), %.4f mm/px (y)  from %g x %g mm "
              "over %d x %d px" % (sx, sy, W_mm, H_mm, wpx, hpx))
        print("origin (%s) at pixel (%.1f, %.1f)" % (args.origin, ox, oy))
        process(args.calibration, args.folder, "frame",
                frame=(sx, sy, ox, oy))
    else:
        if not (args.ref_json and args.ref_id):
            ap.error("--ref-mm requires --ref-json and --ref-id")
        ref = json.load(open(args.ref_json))
        rr = next((r for r in ref["rings"] if r["id"] == args.ref_id), None)
        if rr is None:
            ap.error("ref-id %d not found in %s" % (args.ref_id, args.ref_json))
        mm_per_px = args.ref_mm / (2.0 * rr["radius"])
        print("scale from reference: %.4f mm/px "
              "(ring #%d = %.2f mm outer over %.1f px radius)"
              % (mm_per_px, args.ref_id, args.ref_mm, rr["radius"]))
        process(args.calibration, args.folder, "scale", mm_per_px=mm_per_px)


if __name__ == "__main__":
    main()
