# Train a custom ring model (YOLO-seg)

FastSAM works out of the box but is generic and slow (~2–3 s/image on CPU).
A small model trained on **your** washers is more accurate, rejects
belt/lighting false positives, gives a confidence score, and runs in
milliseconds. This folder makes that quick.

## 1. Collect images
Put 100–300 representative images (varied positions, sizes, lighting, some
empty belt) in a folder, e.g. `data/train_shots/`.

## 2. Bootstrap labels from FastSAM (saves most of the work)
```bash
python training/pseudolabel.py data/train_shots training/dataset --val 0.2
```
This runs FastSAM and writes YOLO-seg polygon labels + a dataset:
```
training/dataset/
  images/train  images/val
  labels/train  labels/val
  dataset.yaml
```

## 3. Correct the labels
Open the dataset in a labeling tool and fix mistakes (missed rings, false
positives, loose outlines):
- **Label Studio**, **Roboflow**, or **CVAT** (import YOLO format), or
- **labelImg** / **X-AnyLabeling** for local editing.
Keep the single class `ring` (id 0). Good labels = good model.

## 4. Train
```bash
python training/train_yolo.py --data training/dataset/dataset.yaml \
       --model yolo11n-seg.pt --epochs 100 --imgsz 640 --device cpu
```
Use `--device 0` on a CUDA GPU (much faster). `yolo11n-seg.pt` is the
smallest/fastest; step up to `yolo11s-seg.pt` / `yolo11m-seg.pt` for more
accuracy. Best weights land in `runs/segment/train*/weights/best.pt`.

## 5. Use it in the app
Configuration → **Detection**:
- **MODEL** = the path to `best.pt`
- **Model type** = `yolo`
- Save & Restart.

Everything else (mm mapping, diameter, TCP, CSV, inner diameter) works the
same — the app auto-handles YOLO masks. Retrain whenever the part or camera
changes.

> Tip: the app's **sub-pixel circle fit** (Detection tab) improves diameter
> precision on top of any model.
