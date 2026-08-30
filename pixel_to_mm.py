# ============================================================
# Convert ring pixel positions to millimetres using a calibration
# ------------------------------------------------------------
# The camera calibration (fx, fy, cx, cy, distortion) from calibrate.py
# maps a pixel to a *direction* from the camera - not to a distance. To
# get millimetres you also need the SCALE of the plane the rings lie on.
# This module gives you two ways to supply it:
#
#   A) Homography  (recommended, most accurate)
#      Put the ChArUco board flat in the SAME plane as your rings, take
#      one reference photo, and build a pixel -> mm map from it. This is
#      exact even if the camera is not perfectly perpendicular, and it
#      needs no distance measurement. Best for a fixed camera + plane rig.
#
#   B) Perpendicular pinhole
#      If the camera looks straight down at the plane from a known
#      distance Z (mm), then  X_mm = (u - cx) * Z / fx. Simple, but only
#      correct when the camera really is perpendicular and Z is known.
#
# In both cases the pixel is undistorted first using the calibration.
#
# Usage examples:
#   # one-time: make a pixel->mm map from a board photo in the ring plane
#   python pixel_to_mm.py make-map calibration.json board_in_plane.bmp \
#          --square-mm 20 --out plane_map.json
#
#   # convert a ring pixel with that map
#   python pixel_to_mm.py to-mm calibration.json --map plane_map.json --u 210 --v 128
#
#   # or the perpendicular shortcut with a known working distance
#   python pixel_to_mm.py to-mm calibration.json --Z 300 --u 210 --v 128
# ============================================================

import cv2
import numpy as np
import json
import argparse

# board defaults must match your calibration board (see calibrate.py)
SQUARES_X, SQUARES_Y = 10, 7
MARKER_RATIO = 0.68
DICTIONARY = "DICT_4X4_50"


# ---------------- calibration ----------------

def load_calibration(path):
    """Load calibration.json -> (K, dist)."""
    with open(path) as f:
        cal = json.load(f)
    K = np.array(cal["camera_matrix"], dtype=np.float64)
    dist = np.array(cal["distortion_coefficients"], dtype=np.float64)
    return K, dist


def undistort_px(pts, K, dist):
    """Undistort pixel coordinates, staying in pixel units (P=K).

    pts: (N,2) array-like. Returns (N,2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(pts, K, dist, P=K)
    return out.reshape(-1, 2)


# ---------------- A) homography (plane) ----------------

def make_plane_map(calib_path, board_image, square_mm,
                   squares_x=SQUARES_X, squares_y=SQUARES_Y,
                   marker_mm=None, dict_name=DICTIONARY):
    """Build a pixel -> mm homography from a photo of the ChArUco board
    lying in the measurement plane.

    Returns a dict you can save as JSON and reuse. The mm coordinate system
    has its origin at the board's first inner corner, x along the columns,
    y along the rows, in millimetres.
    """
    K, dist = load_calibration(calib_path)
    if marker_mm is None:
        marker_mm = square_mm * MARKER_RATIO
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_mm,
                                   marker_mm, dictionary)
    detector = cv2.aruco.CharucoDetector(board)

    img = cv2.imread(board_image)
    if img is None:
        raise FileNotFoundError(board_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ch_corners, ch_ids, _, _ = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 8:
        raise RuntimeError("Not enough board corners found in %s "
                           "(need >= 8, got %d)"
                           % (board_image, 0 if ch_ids is None else len(ch_ids)))

    # object points in mm (board plane, z=0) and matching image points
    obj_all = board.getChessboardCorners()          # (Ncorners,3) in mm
    ids = ch_ids.flatten()
    mm_pts = obj_all[ids][:, :2].astype(np.float64)   # (N,2) mm
    img_pts = undistort_px(ch_corners.reshape(-1, 2), K, dist)  # (N,2) px

    # homography: undistorted pixel -> mm
    H, mask = cv2.findHomography(img_pts, mm_pts, cv2.RANSAC, 3.0)
    if H is None:
        raise RuntimeError("homography fit failed")

    # residual check: map the image points through H, compare to true mm
    proj = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    resid = np.sqrt(((proj - mm_pts) ** 2).sum(axis=1))
    return {
        "type": "homography_px_to_mm",
        "calibration": calib_path,
        "square_mm": square_mm,
        "H": H.tolist(),
        "num_corners": int(len(ids)),
        "residual_mm_rms": float(np.sqrt((resid ** 2).mean())),
        "residual_mm_max": float(resid.max()),
    }


def px_to_mm_homography(u, v, K, dist, H):
    """Undistort pixel (u,v) then map to mm through homography H."""
    p = undistort_px([[u, v]], K, dist)
    mm = cv2.perspectiveTransform(p.reshape(-1, 1, 2), np.asarray(H)).reshape(2)
    return float(mm[0]), float(mm[1])


# ---------------- B) perpendicular pinhole ----------------

def px_to_mm_perpendicular(u, v, K, dist, Z_mm):
    """Undistort pixel (u,v) then convert to mm assuming the plane is
    perpendicular to the optical axis at distance Z_mm.

    Returns mm relative to the optical axis (0,0 = straight ahead)."""
    p = undistort_px([[u, v]], K, dist)[0]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (p[0] - cx) * Z_mm / fx
    Y = (p[1] - cy) * Z_mm / fy
    return float(X), float(Y)


# ---------------- CLI ----------------

def _cli():
    ap = argparse.ArgumentParser(description="Convert ring pixels to mm.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make-map",
                       help="build a pixel->mm map from a board photo taken "
                            "in the ring plane")
    m.add_argument("calibration")
    m.add_argument("board_image")
    m.add_argument("--square-mm", type=float, required=True,
                   help="printed square edge in mm")
    m.add_argument("--squares-x", type=int, default=SQUARES_X)
    m.add_argument("--squares-y", type=int, default=SQUARES_Y)
    m.add_argument("--dict", dest="dict_name", default=DICTIONARY)
    m.add_argument("--out", default="plane_map.json")

    t = sub.add_parser("to-mm", help="convert one pixel to mm")
    t.add_argument("calibration")
    t.add_argument("--u", type=float, required=True, help="pixel x")
    t.add_argument("--v", type=float, required=True, help="pixel y")
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--map", help="plane_map.json from 'make-map' (mode A)")
    g.add_argument("--Z", type=float, help="working distance in mm (mode B)")

    args = ap.parse_args()

    if args.cmd == "make-map":
        m = make_plane_map(args.calibration, args.board_image, args.square_mm,
                           args.squares_x, args.squares_y,
                           dict_name=args.dict_name)
        with open(args.out, "w") as f:
            json.dump(m, f, indent=2)
        print("pixel->mm map written to %s" % args.out)
        print("  corners used     : %d" % m["num_corners"])
        print("  fit residual RMS : %.3f mm  (max %.3f mm)"
              % (m["residual_mm_rms"], m["residual_mm_max"]))
        print("  -> if the residual is large, re-take the board photo flatter "
              "/ sharper.")

    elif args.cmd == "to-mm":
        K, dist = load_calibration(args.calibration)
        if args.map:
            mp = json.load(open(args.map))
            x, y = px_to_mm_homography(args.u, args.v, K, dist, mp["H"])
            print("pixel (%.1f, %.1f) -> (%.3f, %.3f) mm  [board-plane frame]"
                  % (args.u, args.v, x, y))
        else:
            x, y = px_to_mm_perpendicular(args.u, args.v, K, dist, args.Z)
            print("pixel (%.1f, %.1f) -> (%.3f, %.3f) mm  [from optical axis, "
                  "Z=%.1f mm]" % (args.u, args.v, x, y, args.Z))


if __name__ == "__main__":
    _cli()
