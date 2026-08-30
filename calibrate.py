# ============================================================
# Camera calibration from ChArUco board images
# ------------------------------------------------------------
# Computes the camera intrinsics (focal length + the TRUE optical
# centre) and lens-distortion coefficients from a folder of
# calibration photos of a ChArUco board.
#
# Why this matters for ring-finding:
#   Converting a ring's pixel position to millimetres is only
#   correct once you know
#     1. the real optical centre (cx, cy) - a webcam's optical
#        centre is NOT the image centre (W/2, H/2); it is usually
#        offset by several pixels, and
#     2. the lens distortion, which bends straight lines and pushes
#        off-centre points further from where they really are.
#   Assuming the image centre and ignoring distortion is exactly
#   what makes positions "drift away" the further they are from the
#   middle of the frame. This tool measures both so you can
#   undistort a pixel coordinate before scaling it to mm.
#
# Install once:  pip install opencv-contrib-python numpy
# Run:           python calibrate.py Cali/
#                python calibrate.py Cali/ --square-mm 20 --marker-mm 14
#
# Outputs (written next to the images by default):
#   calibration.json   human-readable intrinsics + distortion
#   calibration.npz    same data for numpy / OpenCV
#   undistorted/       side-by-side before/after previews
#
# Board geometry (override on the command line if yours differs):
#   10 x 7 squares, DICT_4X4_50 ArUco markers, marker/square ~= 0.68
# ============================================================

import cv2
import numpy as np
import sys
import os
import glob
import json
import argparse

# -------- DEFAULT BOARD CONFIG --------
SQUARES_X = 10          # number of chessboard squares across
SQUARES_Y = 7           # number of chessboard squares down
SQUARE_MM = 20.0        # printed size of one square (edge), in mm  *** set to YOUR board ***
MARKER_RATIO = 0.68     # marker edge / square edge (measured from the sample set)
DICTIONARY = "DICT_4X4_50"
# --------------------------------------

IMAGE_EXTS = ("png", "jpg", "jpeg", "bmp", "tif", "tiff")


def _get_dictionary(name):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def _make_board(squares_x, squares_y, square_mm, marker_mm, dictionary):
    # OpenCV >= 4.7 API
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm, marker_mm, dictionary
    )


def _list_images(folder):
    files = []
    for ext in IMAGE_EXTS:
        files += glob.glob(os.path.join(folder, "*." + ext))
        files += glob.glob(os.path.join(folder, "*." + ext.upper()))
    return sorted(set(files))


def _assess_quality(K, dist, rms, image_size, obj_points, img_points):
    """Judge whether the calibration is trustworthy.

    The biggest failure mode for pixel->mm work is a calibration built from
    board views that are all nearly head-on (fronto-parallel). Without tilt,
    focal length is not separable from distance and the principal point is
    barely observable, so the numbers look precise (tiny reprojection error)
    yet are physically wrong - which is exactly what makes off-centre points
    map to the wrong millimetres.
    """
    w, h = image_size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    # per-view board tilt (angle of the board plane away from fronto-parallel)
    tilts = []
    for op, ip in zip(obj_points, img_points):
        ok, rvec, tvec = cv2.solvePnP(op, ip, K, dist)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        tilts.append(float(np.degrees(np.arccos(min(1.0, abs(R[2, 2]))))))
    tilts = np.array(tilts) if tilts else np.array([0.0])
    tilt_p90 = float(np.percentile(tilts, 90))

    fov_x = float(np.degrees(2 * np.arctan(w / (2 * fx))))
    cx_frac = cx / w
    cy_frac = cy / h

    warnings = []
    # 1) tilt diversity - the decisive check
    if tilt_p90 < 8:
        warnings.append(
            "Board is nearly frontal in every view (90th-pct tilt %.1f deg). "
            "Focal length and optical centre are UNOBSERVABLE from this set - "
            "the numbers below are unreliable no matter how small the "
            "reprojection error looks. Re-shoot with the board tilted "
            "20-45 deg in several directions." % tilt_p90)
    elif tilt_p90 < 18:
        warnings.append(
            "Limited board tilt (90th-pct %.1f deg). More tilt (20-45 deg in "
            "varied directions) would make focal length / distortion more "
            "reliable." % tilt_p90)
    # 2) principal point should sit near the middle of the sensor
    if not (0.30 <= cx_frac <= 0.70 and 0.30 <= cy_frac <= 0.70):
        warnings.append(
            "Optical centre (%.0f, %.0f) is far from the image middle "
            "(%.0f, %.0f) - implausible for a normal lens, a sign the "
            "calibration did not converge to a real solution."
            % (cx, cy, (w - 1) / 2.0, (h - 1) / 2.0))
    # 3) field of view sanity
    if not (25 <= fov_x <= 120):
        warnings.append(
            "Implausible horizontal field of view %.1f deg (fx=%.0f). Typical "
            "webcams are 40-90 deg; this usually means focal length was not "
            "constrained by the data." % (fov_x, fx))

    if any("UNOBSERVABLE" in wtext or "implausible" in wtext.lower()
           for wtext in warnings):
        verdict = "UNRELIABLE"
    elif warnings:
        verdict = "QUESTIONABLE"
    else:
        verdict = "GOOD"

    return {
        "verdict": verdict,
        "warnings": warnings,
        "board_tilt_deg": {
            "min": round(float(tilts.min()), 2),
            "mean": round(float(tilts.mean()), 2),
            "max": round(float(tilts.max()), 2),
            "pct90": round(tilt_p90, 2),
        },
        "horizontal_fov_deg": round(fov_x, 1),
    }


def calibrate(folder, square_mm, marker_mm, squares_x, squares_y,
              dict_name, out_dir=None, save_previews=True):
    dictionary = _get_dictionary(dict_name)
    board = _make_board(squares_x, squares_y, square_mm, marker_mm, dictionary)

    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, aruco_params)
    charuco_detector = cv2.aruco.CharucoDetector(board)

    images = _list_images(folder)
    if not images:
        print("no images found in " + folder)
        return None

    all_corners = []      # charuco corner pixel coords, per image
    all_ids = []          # charuco corner ids, per image
    used = []             # image paths that contributed
    image_size = None

    print("Detecting ChArUco corners in %d images ..." % len(images))
    for path in images:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]  # (w, h)

        ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray)
        n = 0 if ch_ids is None else len(ch_ids)
        if n >= 6:  # need a reasonable number of corners for a stable view
            all_corners.append(ch_corners)
            all_ids.append(ch_ids)
            used.append(path)
        print("  %-28s %2d corners%s" % (
            os.path.basename(path), n, "" if n >= 6 else "  (skipped)"))

    if len(all_corners) < 3:
        print("Not enough usable views (%d). Need at least 3." % len(all_corners))
        return None

    # matched object/image points -> calibrateCamera
    obj_points = []
    img_points = []
    for corners, ids in zip(all_corners, all_ids):
        op, ip = board.matchImagePoints(corners, ids)
        if op is not None and len(op) >= 6:
            obj_points.append(op)
            img_points.append(ip)

    # Initial-guess + fixed k3 keeps a cheap webcam's calibration stable:
    # a free k3 on a small sensor tends to run away and drag fx/cx with it.
    w, h = image_size
    K_init = np.array([[float(w), 0, (w - 1) / 2.0],
                       [0, float(w), (h - 1) / 2.0],
                       [0, 0, 1.0]])
    dist_init = np.zeros((5, 1))
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, K_init, dist_init, flags=flags
    )

    # per-view reprojection error
    per_view = []
    total_err, total_pts = 0.0, 0
    for i, (op, ip) in enumerate(zip(obj_points, img_points)):
        proj, _ = cv2.projectPoints(op, rvecs[i], tvecs[i], K, dist)
        ip_f = np.asarray(ip, dtype=np.float64).reshape(-1, 2)
        proj_f = np.asarray(proj, dtype=np.float64).reshape(-1, 2)
        err = float(np.sqrt(((ip_f - proj_f) ** 2).sum()))
        n = len(op)
        per_view.append((used[i], err / np.sqrt(n)))
        total_err += err * err
        total_pts += n
    mean_err = np.sqrt(total_err / total_pts)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    quality = _assess_quality(K, dist, rms, image_size, obj_points, img_points)

    result = {
        "quality": quality,
        "image_width": int(w),
        "image_height": int(h),
        "num_images_used": len(used),
        "num_images_total": len(images),
        "rms_reprojection_error_px": round(float(rms), 4),
        "mean_reprojection_error_px": round(float(mean_err), 4),
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "fx": round(fx, 3),
        "fy": round(fy, 3),
        "cx": round(cx, 3),
        "cy": round(cy, 3),
        "image_centre_x": (w - 1) / 2.0,
        "image_centre_y": (h - 1) / 2.0,
        "optical_centre_offset_px": [round(cx - (w - 1) / 2.0, 2),
                                     round(cy - (h - 1) / 2.0, 2)],
        "board": {
            "squares_x": squares_x,
            "squares_y": squares_y,
            "square_mm": square_mm,
            "marker_mm": round(marker_mm, 4),
            "dictionary": dict_name,
        },
    }

    out_dir = out_dir or folder
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "calibration.json")
    npz_path = os.path.join(out_dir, "calibration.npz")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    np.savez(npz_path,
             camera_matrix=K,
             distortion_coefficients=dist,
             image_width=w, image_height=h,
             rms_reprojection_error_px=rms)

    # ---- report ----
    print("")
    print("=" * 52)
    print("Calibration complete  (%d / %d views used)" % (len(used), len(images)))
    print("  RMS reprojection error : %.4f px" % rms)
    print("  mean reprojection error: %.4f px" % mean_err)
    print("  image size             : %d x %d" % (w, h))
    print("  focal length  fx, fy   : %.2f, %.2f px" % (fx, fy))
    print("  optical centre cx, cy  : %.2f, %.2f px" % (cx, cy))
    print("  image centre           : %.1f, %.1f px" % ((w - 1) / 2.0, (h - 1) / 2.0))
    print("  --> optical centre is offset by (%.2f, %.2f) px from image centre"
          % (cx - (w - 1) / 2.0, cy - (h - 1) / 2.0))
    print("  distortion (k1 k2 p1 p2 k3 ...):")
    print("     " + "  ".join("%+.5f" % v for v in dist.ravel()))
    print("  board tilt (deg)       : min %.1f  mean %.1f  max %.1f  (90th %.1f)"
          % (quality["board_tilt_deg"]["min"], quality["board_tilt_deg"]["mean"],
             quality["board_tilt_deg"]["max"], quality["board_tilt_deg"]["pct90"]))
    print("  horizontal field of view: %.1f deg" % quality["horizontal_fov_deg"])
    worst = sorted(per_view, key=lambda t: -t[1])[:3]
    print("  worst views:")
    for p, e in worst:
        print("     %-28s %.3f px" % (os.path.basename(p), e))
    print("-" * 52)
    print("  QUALITY: " + quality["verdict"])
    for wtext in quality["warnings"]:
        # wrap warning text to ~66 cols for the console
        words, line = wtext.split(), ""
        print("   [!] ", end="")
        col = 0
        for word in words:
            if col + len(word) + 1 > 62:
                print("\n        ", end="")
                col = 0
            print(word + " ", end="")
            col += len(word) + 1
        print("")
    if not quality["warnings"]:
        print("   No problems detected. Reprojection error and geometry look sane.")
    print("=" * 52)
    print("  -> " + json_path)
    print("  -> " + npz_path)

    if save_previews:
        prev_dir = os.path.join(out_dir, "undistorted")
        os.makedirs(prev_dir, exist_ok=True)
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        for path in used[:8]:
            img = cv2.imread(path)
            und = cv2.undistort(img, K, dist, None, newK)
            combo = np.hstack([img, und])
            cv2.imwrite(os.path.join(prev_dir,
                        os.path.splitext(os.path.basename(path))[0] + "_ud.png"), combo)
        print("  -> " + prev_dir + "  (left = original, right = undistorted)")

    return result


def build_parser():
    p = argparse.ArgumentParser(
        description="Camera calibration from ChArUco board images.")
    p.add_argument("folder", help="folder containing the calibration images")
    p.add_argument("--square-mm", type=float, default=SQUARE_MM,
                   help="printed edge length of one square in mm "
                        "(default %(default)s) - set this to YOUR board")
    p.add_argument("--marker-mm", type=float, default=None,
                   help="printed edge length of one ArUco marker in mm "
                        "(default = %.2f x square)" % MARKER_RATIO)
    p.add_argument("--squares-x", type=int, default=SQUARES_X,
                   help="number of squares across (default %(default)s)")
    p.add_argument("--squares-y", type=int, default=SQUARES_Y,
                   help="number of squares down (default %(default)s)")
    p.add_argument("--dict", default=DICTIONARY, dest="dict_name",
                   help="ArUco dictionary (default %(default)s)")
    p.add_argument("--out", default=None,
                   help="output folder (default: the image folder)")
    p.add_argument("--no-previews", action="store_true",
                   help="do not write undistorted preview images")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    marker_mm = args.marker_mm if args.marker_mm is not None \
        else args.square_mm * MARKER_RATIO
    calibrate(args.folder, args.square_mm, marker_mm,
              args.squares_x, args.squares_y, args.dict_name,
              out_dir=args.out, save_previews=not args.no_previews)
