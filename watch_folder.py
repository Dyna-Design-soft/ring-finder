# ============================================================
# watch_folder.py  -  continuously watch a folder, compute XY per image
# ------------------------------------------------------------
# Runs forever. Whenever a NEW image file lands in WATCH_DIR, it detects
# the ring(s) and computes the robot XY, then writes:
#   <image>_rings.png    annotated picture
#   <image>_rings.json   pixel + robot XY
# and appends one row per ring to results.csv (a running log).
#
# Designed to run inside PyCharm: set the folders in CONFIG below (or pass
# them on the command line) and press Run. Stop with the red Stop button
# (or Ctrl+C).
#
# Install once:  pip install ultralytics opencv-python numpy
#
# How "new" is decided: the folder is polled every POLL_SECONDS; a file is
# processed once its size has stopped changing (so half-copied images are
# not read early) and it has not been processed before.
# ============================================================

import os
import sys
import csv
import time
import argparse

import ring_finder as rf   # reuse the detector + robot map from ring_finder.py

# -------- CONFIG (edit these, or override on the command line) --------
WATCH_DIR     = "./incoming"     # folder the camera drops images into
ROBOT_MAP     = None             # path to robot_map.json, or None = built-in map
RESULTS_CSV   = "results.csv"    # running log (created inside WATCH_DIR)
POLL_SECONDS  = 1.0              # how often to scan the folder
IMAGE_EXTS    = (".bmp",)         # file types to watch for (camera writes .bmp)
PROCESS_EXISTING = False         # True: also process images already there at start
# ---------------------------------------------------------------------


def _is_image(name):
    low = name.lower()
    return low.endswith(IMAGE_EXTS) and not low.endswith("_rings.png")


def _stable_size(path, prev):
    """Return current size; a file is 'stable' when this equals prev."""
    try:
        return os.path.getsize(path)
    except OSError:
        return -1


def _append_csv(csv_path, image, rings):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "image", "ring_id",
                        "x_px", "y_px", "diameter_px", "robot_x_mm", "robot_y_mm"])
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in rings:
            w.writerow([ts, image, r["id"], r["x"], r["y"], r["diameter"],
                        r.get("robot_x", ""), r.get("robot_y", "")])


def process(path, H, csv_path):
    import json
    try:
        rf.detect_rings(path, H)               # writes _rings.png + _rings.json
        jp = os.path.splitext(path)[0] + "_rings.json"
        rings = json.load(open(jp)).get("rings", []) if os.path.exists(jp) else []
        _append_csv(csv_path, os.path.basename(path), rings)
    except Exception as e:                      # never let one bad file kill the loop
        print("  ! error on %s: %s" % (os.path.basename(path), e))


def watch(watch_dir, robot_map, results_csv, poll, process_existing):
    watch_dir = os.path.abspath(watch_dir)
    os.makedirs(watch_dir, exist_ok=True)
    csv_path = os.path.join(watch_dir, results_csv)
    H = None if robot_map == "none" else rf.load_homography(robot_map)

    # Snapshot what is already there BEFORE the (slow) model load, so images
    # that arrive while the model is loading are still treated as new.
    done = set()          # basenames already processed
    sizes = {}            # basename -> last seen size (for stability)
    if not process_existing:
        for n in os.listdir(watch_dir):
            if _is_image(n):
                done.add(n)                     # ignore what was there at startup

    print("Loading FastSAM model ...")
    rf.get_model()                              # load once, up front
    print("Watching: %s" % watch_dir)
    print("Robot map: %s" % ("built-in" if robot_map is None else robot_map))
    print("Results  : %s" % csv_path)
    print("File type: %s" % ", ".join(IMAGE_EXTS))
    print("Poll     : every %.1fs   (Ctrl+C to stop)" % poll)
    if done:
        print("(ignoring %d file(s) already present; waiting for new ones)" % len(done))
    print("")

    while True:
        try:
            names = [n for n in os.listdir(watch_dir) if _is_image(n)]
        except FileNotFoundError:
            time.sleep(poll)
            continue
        for n in sorted(names):
            if n in done:
                continue
            path = os.path.join(watch_dir, n)
            size = _stable_size(path, sizes.get(n))
            if size < 0:
                continue
            if sizes.get(n) != size:            # still being written; check next round
                sizes[n] = size
                continue
            # size stable -> process
            print("NEW: %s" % n)
            process(path, H, csv_path)
            done.add(n)
            sizes.pop(n, None)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(
        description="Continuously watch a folder and compute ring XY.")
    ap.add_argument("watch_dir", nargs="?", default=WATCH_DIR,
                    help="folder to watch (default %(default)s)")
    ap.add_argument("--map", default=ROBOT_MAP,
                    help="robot_map.json (default: built-in); use 'none' for "
                         "pixels only")
    ap.add_argument("--csv", default=RESULTS_CSV,
                    help="results log filename (default %(default)s)")
    ap.add_argument("--poll", type=float, default=POLL_SECONDS,
                    help="seconds between scans (default %(default)s)")
    ap.add_argument("--all", action="store_true", default=PROCESS_EXISTING,
                    help="also process images already in the folder at start")
    args = ap.parse_args()
    try:
        watch(args.watch_dir, args.map, args.csv, args.poll, args.all)
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
