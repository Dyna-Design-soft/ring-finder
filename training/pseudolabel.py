# ============================================================
# pseudolabel.py  -  bootstrap YOLO-seg labels from FastSAM
# ------------------------------------------------------------
# Runs the current FastSAM detector over a folder of images and writes
# YOLO-segmentation label files (one polygon per ring, class 0 = "ring"),
# plus a dataset folder + dataset.yaml ready for training. You then only
# CORRECT the labels in a tool (LabelImg/Roboflow/CVAT) instead of drawing
# everything from scratch, and train (see train_yolo.py).
#
# Usage:
#   python training/pseudolabel.py <images_dir> [out_dir] [--val 0.2]
#
# Output (out_dir, default training/dataset):
#   images/train, images/val, labels/train, labels/val, dataset.yaml
# ============================================================

import os
import sys
import glob
import shutil
import random
import argparse

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "scripts"))
import ring_app as A   # reuse the exact detector/config


def mask_polygons(img, cfg, det):
    """Return list of normalized polygons [[x1,y1,x2,y2,...], ...] for rings."""
    H, W = img.shape[:2]
    min_af = float(cfg.get("min_area_frac", 0.004))
    max_af = float(cfg.get("max_area_frac", 0.25))
    min_circ = float(cfg.get("min_circ", 0.75))
    res = det._predict(img, float(cfg.get("conf", 0.20)),
                       float(cfg.get("iou", 0.7)), 1024)
    polys = []
    masks = getattr(res, "masks", None)
    if masks is None or masks.data is None:
        return polys
    for m in masks.data.cpu().numpy():
        mask = (m > 0.5).astype(np.uint8)
        a = int(mask.sum())
        if a < min_af * H * W or a > max_af * H * W:
            continue
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * cv2.contourArea(c) / (peri * peri) if peri else 0
        if circ < min_circ:
            continue
        # simplify the contour a little, normalize to 0..1
        eps = 0.005 * peri
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(approx) < 3:
            continue
        approx[:, 0] /= W
        approx[:, 1] /= H
        polys.append(approx.reshape(-1).tolist())
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images_dir")
    ap.add_argument("out_dir", nargs="?",
                    default=os.path.join(os.path.dirname(__file__), "dataset"))
    ap.add_argument("--val", type=float, default=0.2, help="val split fraction")
    ap.add_argument("--class-name", default="ring")
    args = ap.parse_args()

    cfg = A.load_config()
    det = A.Detector()
    det.load(cfg.get("model", "FastSAM-x.pt"), cfg.get("model_type", "auto"))

    files = []
    for ext in ("*.bmp", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        files += glob.glob(os.path.join(args.images_dir, ext))
    files = sorted(f for f in files if not f.lower().endswith("_rings.png"))
    if not files:
        print("no images in " + args.images_dir)
        return
    random.seed(0)
    random.shuffle(files)
    n_val = int(len(files) * args.val)
    split = {f: ("val" if i < n_val else "train") for i, f in enumerate(files)}

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(args.out_dir, sub), exist_ok=True)

    labeled = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        polys = mask_polygons(img, cfg, det)
        s = split[f]
        stem = os.path.splitext(os.path.basename(f))[0]
        shutil.copy(f, os.path.join(args.out_dir, "images", s,
                                    os.path.basename(f)))
        with open(os.path.join(args.out_dir, "labels", s, stem + ".txt"),
                  "w") as fh:
            for p in polys:
                fh.write("0 " + " ".join("%.6f" % v for v in p) + "\n")
        labeled += len(polys)
        print("%s -> %d ring(s) [%s]" % (os.path.basename(f), len(polys), s))

    with open(os.path.join(args.out_dir, "dataset.yaml"), "w") as fh:
        fh.write("path: %s\n" % os.path.abspath(args.out_dir))
        fh.write("train: images/train\n")
        fh.write("val: images/val\n")
        fh.write("names:\n  0: %s\n" % args.class_name)

    print("\n%d images, %d ring polygons written to %s"
          % (len(files), labeled, args.out_dir))
    print("dataset.yaml ready. Next: correct labels, then run train_yolo.py")


if __name__ == "__main__":
    main()
