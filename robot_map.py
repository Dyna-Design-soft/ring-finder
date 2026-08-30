# ============================================================
# Pixel -> robot XY mapping (hand-eye homography)
# ------------------------------------------------------------
# Turns a detected ring's pixel position into the robot's XY coordinate
# (mm) so a pick-and-place can go straight to it. The camera views a flat
# plane, so the pixel <-> robot relationship is a single homography. You
# calibrate it once from a handful of rings whose robot XY you recorded,
# then apply it to every new detection.
#
# This maps pixels DIRECTLY to robot mm, so it needs no camera-intrinsics
# file: the homography absorbs scale, perspective and the (negligible)
# lens distortion in one step.
#
#   fit    build the map from numbered ring images + their robot XY
#          python robot_map.py fit --rings parts/ --robot robot_xy.csv \
#                 --out robot_map.json
#
#   apply  convert a pixel, or write robot XY into ai.py's *_rings.json
#          python robot_map.py apply robot_map.json --u 210 --v 128
#          python robot_map.py apply robot_map.json --rings newparts/
#
# robot_xy.csv has a header row then id,x,y (mm), one row per image; the
# ring pixel for image <id> is read from <id>_rings.json (run ai.py first).
# ============================================================

import cv2
import numpy as np
import json
import os
import csv
import glob
import argparse


def _read_robot_csv(path):
    """id -> (x, y) from a csv with a header line then id,x,y."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or len(row) < 3:
                continue
            try:
                out[int(float(row[0]))] = (float(row[1]), float(row[2]))
            except ValueError:
                continue  # header / blank
    if not out:
        raise RuntimeError("no id,x,y rows parsed from " + path)
    return out


def _ring_pixel(rings_dir, i):
    """Centre pixel of the (first) ring in <i>_rings.json."""
    jp = os.path.join(rings_dir, "%d_rings.json" % i)
    if not os.path.exists(jp):
        return None
    rings = json.load(open(jp)).get("rings", [])
    if not rings:
        return None
    return (rings[0]["x"], rings[0]["y"])


def fit(rings_dir, robot_csv, out_path, ransac_mm=6.0):
    robot = _read_robot_csv(robot_csv)
    ids, P, R, missing = [], [], [], []
    for i in sorted(robot):
        p = _ring_pixel(rings_dir, i)
        if p is None:
            missing.append(i)
            continue
        ids.append(i)
        P.append(p)
        R.append(robot[i])
    if missing:
        print("warning: no ring found for image id(s): %s "
              "(run ai.py on them first)" % missing)
    if len(ids) < 4:
        raise RuntimeError("need at least 4 paired points, have %d" % len(ids))
    P = np.array(P, np.float64)
    R = np.array(R, np.float64)

    # robust fit: flag gross data errors (e.g. a mistyped robot coordinate)
    H, mask = cv2.findHomography(P, R, cv2.RANSAC, ransac_mm)
    inl = [ids[k] for k in range(len(ids)) if mask[k]]
    out = [ids[k] for k in range(len(ids)) if not mask[k]]

    # refit with least-squares on the inliers only
    Pi = np.array([P[ids.index(i)] for i in inl], np.float64)
    Ri = np.array([R[ids.index(i)] for i in inl], np.float64)
    H, _ = cv2.findHomography(Pi, Ri, 0)

    proj = cv2.perspectiveTransform(Pi.reshape(-1, 1, 2), H).reshape(-1, 2)
    res = np.linalg.norm(proj - Ri, axis=1)

    # leave-one-out: honest accuracy on an unseen ring
    loo = []
    for hold in inl:
        tr = [i for i in inl if i != hold]
        Pt = np.array([P[ids.index(i)] for i in tr], np.float64)
        Rt = np.array([R[ids.index(i)] for i in tr], np.float64)
        Hh, _ = cv2.findHomography(Pt, Rt, 0)
        q = cv2.perspectiveTransform(
            np.array([P[ids.index(hold)]], np.float64).reshape(-1, 1, 2),
            Hh).reshape(2)
        loo.append(float(np.linalg.norm(q - R[ids.index(hold)])))

    print("Fitted pixel -> robot homography on %d point(s): %s" % (len(inl), inl))
    print("  in-fit RMS      : %.3f mm   (max %.3f mm)"
          % (np.sqrt((res ** 2).mean()), res.max()))
    print("  leave-one-out RMS: %.3f mm  (expected error on a new ring)"
          % np.sqrt(np.mean(np.square(loo))))
    if out:
        print("  EXCLUDED as outliers (check these robot XY rows): %s" % out)
        for i in out:
            q = cv2.perspectiveTransform(
                np.array([P[ids.index(i)]], np.float64).reshape(-1, 1, 2),
                H).reshape(2)
            print("     id %d: sheet (%.2f, %.2f)  but map predicts (%.2f, %.2f)"
                  % (i, robot[i][0], robot[i][1], q[0], q[1]))

    data = {
        "type": "homography_px_to_robot_mm",
        "H": H.tolist(),
        "points_used": inl,
        "excluded": out,
        "rms_mm": float(np.sqrt((res ** 2).mean())),
        "loo_rms_mm": float(np.sqrt(np.mean(np.square(loo)))),
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print("  -> " + out_path)
    return data


def px_to_robot(u, v, H):
    q = cv2.perspectiveTransform(
        np.array([[u, v]], np.float64).reshape(-1, 1, 2), np.asarray(H)).reshape(2)
    return float(q[0]), float(q[1])


def apply(map_path, u=None, v=None, rings_dir=None):
    H = json.load(open(map_path))["H"]
    if rings_dir:
        for jp in sorted(glob.glob(os.path.join(rings_dir, "*_rings.json"))):
            data = json.load(open(jp))
            for ring in data.get("rings", []):
                rx, ry = px_to_robot(ring["x"], ring["y"], H)
                ring["robot_x"] = round(rx, 3)
                ring["robot_y"] = round(ry, 3)
            json.dump(data, open(jp, "w"), indent=2)
            print(os.path.basename(jp) + ":")
            for ring in data.get("rings", []):
                print("  #%d  pixel (%.1f, %.1f) -> robot (%.2f, %.2f) mm"
                      % (ring["id"], ring["x"], ring["y"],
                         ring["robot_x"], ring["robot_y"]))
    else:
        rx, ry = px_to_robot(u, v, H)
        print("pixel (%.1f, %.1f) -> robot (%.2f, %.2f) mm" % (u, v, rx, ry))


def main():
    ap = argparse.ArgumentParser(description="Pixel <-> robot XY homography.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="build the map from images + robot XY")
    f.add_argument("--rings", required=True,
                   help="folder with <id>_rings.json (run ai.py first)")
    f.add_argument("--robot", required=True, help="csv: id,x,y (mm)")
    f.add_argument("--out", default="robot_map.json")
    f.add_argument("--ransac-mm", type=float, default=6.0,
                   help="outlier threshold in mm (default %(default)s)")

    a = sub.add_parser("apply", help="convert pixel(s) to robot XY")
    a.add_argument("map")
    a.add_argument("--u", type=float)
    a.add_argument("--v", type=float)
    a.add_argument("--rings", help="folder of *_rings.json to annotate")

    args = ap.parse_args()
    if args.cmd == "fit":
        fit(args.rings, args.robot, args.out, args.ransac_mm)
    else:
        if args.rings:
            apply(args.map, rings_dir=args.rings)
        elif args.u is not None and args.v is not None:
            apply(args.map, u=args.u, v=args.v)
        else:
            ap.error("apply needs either --rings DIR or both --u and --v")


if __name__ == "__main__":
    main()
