# ============================================================
# train_yolo.py  -  train a custom YOLO-seg ring model
# ------------------------------------------------------------
# Trains an ultralytics YOLO segmentation model on the dataset produced by
# pseudolabel.py (after you correct the labels). The result is a fast,
# accurate model specialised to YOUR washers.
#
# Usage:
#   python training/train_yolo.py [--data training/dataset/dataset.yaml]
#                                 [--model yolo11n-seg.pt] [--epochs 100]
#                                 [--imgsz 640] [--device cpu]
#
# After training, the best weights are printed (runs/segment/train*/weights/
# best.pt). Point the app at it:  Configuration > Detection > MODEL = that
# path, Model type = yolo.  (Use --device 0 on a CUDA GPU for fast training.)
# ============================================================

import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--data", default=os.path.join(here, "dataset", "dataset.yaml"))
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="base seg model: yolo11n-seg.pt (fast) .. yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cpu", help="'cpu' or a GPU index like 0")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device)
    print("\nDone. Use the printed best.pt in the app "
          "(MODEL = .../best.pt, Model type = yolo).")


if __name__ == "__main__":
    main()
