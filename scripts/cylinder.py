# ============================================================
# cylinder.py  -  cylinder-type component detection + table
# ------------------------------------------------------------
# Separate, self-contained add-on for the second component type (cylinders),
# kept OUT of ring_app.py so the working ring/circle pipeline is untouched.
#
# Detection + table logic is faithfully ported from the reference project
# (github.com/Robokks/object-detection-, detect_service + report_service +
# schemas): a YOLO(-seg) model is run on the frame; each detected object gives
#   - a center (axis-aligned bounding-box center),
#   - an orientation angle from cv2.minAreaRect on the mask outline
#     (0 = horizontal, increasing clockwise, normalized to [0, 180)),
#   - left / right edge-midpoints of the bounding box.
# Training / server / labelling parts of that project are intentionally NOT
# included.
#
# On top of that we add robot mapping: left / right / center pixel points are
# converted to robot mm through the SAME calibration map the ring app already
# uses (homography / similarity / affine / thin-plate spline), and the angle is
# reported in the robot frame. The circle pipeline is unaffected - which
# component the watcher runs is chosen by a marker file in the watch folder.
# ============================================================

import os
import math

import cv2
import numpy as np


# ---- component-mode marker file -------------------------------------------

def read_mode(folder, default="circle"):
    """Return 'circle' or 'cylinder' based on a marker file in the watch folder.

    The marker is a small text file named 'mode.txt' whose contents start with
    'cyl' (-> cylinder) or 'circle'/'ring' (-> circle). No file, empty, or an
    unrecognised value -> `default` (circle), so the existing ring behaviour is
    the safe fallback and nothing changes until a marker is placed.
    """
    try:
        path = os.path.join(folder, "mode.txt")
        if not os.path.isfile(path):
            return default
        txt = open(path, "r", encoding="utf-8", errors="ignore").read().strip().lower()
    except Exception:
        return default
    if txt.startswith("cyl"):
        return "cylinder"
    if txt.startswith("circle") or txt.startswith("ring"):
        return "circle"
    return default


# ---- orientation angle (ported from schemas._orientation_angle) -----------

def orientation_angle(points):
    """Long-axis orientation in degrees, 0 = horizontal, increasing clockwise
    (image y-down), normalized to [0, 180). Needs an outline (>=3 points);
    an axis-aligned box carries no orientation, so returns 0."""
    if points is None or len(points) < 3:
        return 0.0
    pts = np.array(points, dtype=np.float32)
    (_, _), (rw, rh), angle = cv2.minAreaRect(pts)
    theta = angle if rw >= rh else angle + 90.0
    return float(theta % 180.0)


def _long_axis_dir(points):
    """Unit direction (dx, dy) of the object's long axis in image pixels, from
    minAreaRect. Falls back to horizontal when there is no outline."""
    a = math.radians(orientation_angle(points))
    return math.cos(a), math.sin(a)


# ---- detector --------------------------------------------------------------

class CylinderDetector:
    """Loads a YOLO / YOLO-seg checkpoint and returns cylinder detections.
    Use a *segment* model (masks) so the orientation angle is real; a plain
    box-only 'detect' model always reports angle 0."""

    def __init__(self):
        self.model = None
        self.weights = None
        self.task = "segment"

    def load(self, weights, task="segment"):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.weights = weights
        self.task = task
        return self

    def detect(self, img_bgr, conf=0.25):
        """img_bgr: HxWx3 BGR (OpenCV). Returns a list of detections, each a
        dict with pixel geometry: class_name, confidence, cx, cy, x, y, w, h,
        angle (deg, long-axis), points (outline or [])."""
        if self.model is None:
            raise RuntimeError("model not loaded - call load() first")
        res = self.model.predict(source=img_bgr[:, :, ::-1], conf=conf,
                                  verbose=False)[0]
        names = getattr(res, "names", {}) or {}
        boxes = getattr(res, "boxes", None)
        masks = getattr(res, "masks", None)
        out = []
        if masks is not None and masks.xy is not None:
            for i, poly in enumerate(masks.xy):
                points = np.asarray(poly, np.float32).tolist()
                if len(points) < 3:
                    continue
                if boxes is not None and i < len(boxes):
                    x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                    c = float(boxes.conf[i].item())
                    ci = int(boxes.cls[i].item())
                else:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    c, ci = None, None
                out.append(self._pack(names, ci, c, x1, y1, x2, y2, points))
        elif boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                c = float(boxes.conf[i].item())
                ci = int(boxes.cls[i].item())
                out.append(self._pack(names, ci, c, x1, y1, x2, y2, []))
        return out

    @staticmethod
    def _pack(names, ci, conf, x1, y1, x2, y2, points):
        w, h = x2 - x1, y2 - y1
        return {
            "class_name": names.get(ci, str(ci)) if ci is not None else "object",
            "confidence": conf,
            "x": x1, "y": y1, "w": w, "h": h,
            "cx": x1 + w / 2.0, "cy": y1 + h / 2.0,
            "angle": orientation_angle(points),
            "points": points,
        }


# ---- table / robot mapping -------------------------------------------------

def _robot_angle(mapper, map_point, cx, cy, points, span=20.0):
    """Orientation of the long axis expressed in the ROBOT frame: step a short
    distance along the pixel long-axis either side of the center, map both to
    robot mm, and take the angle between them. Normalized to [0, 180)."""
    dx, dy = _long_axis_dir(points)
    ax = map_point(mapper, cx - span * dx, cy - span * dy)
    bx = map_point(mapper, cx + span * dx, cy + span * dy)
    ang = math.degrees(math.atan2(bx[1] - ax[1], bx[0] - ax[0]))
    return float(ang % 180.0)


def cylinder_records(dets, cfg=None, mapper=None):
    """Build the cylinder result table. Each record has the pixel geometry and,
    when a calibration `mapper` is given, the robot-mm left/right/center points
    and the angle in the robot frame. `left`/`right` follow the reference
    project: the mid-points of the LEFT and RIGHT edges of the bounding box."""
    from ring_app import map_point                      # reuse the exact mapper
    ox = float(cfg.get("offset_x", 0.0)) if cfg else 0.0
    oy = float(cfg.get("offset_y", 0.0)) if cfg else 0.0
    recs = []
    for i, d in enumerate(dets):
        cx, cy, x, y, w, h = d["cx"], d["cy"], d["x"], d["y"], d["w"], d["h"]
        lpx, lpy = x, cy                 # left edge midpoint (pixel)
        rpx, rpy = x + w, cy             # right edge midpoint (pixel)
        rec = {
            "id": i + 1,
            "class_name": d["class_name"],
            "confidence": round(d["confidence"], 3) if d["confidence"] is not None else "",
            "cx_px": round(cx, 1), "cy_px": round(cy, 1),
            "left_x_px": round(lpx, 1), "left_y_px": round(lpy, 1),
            "right_x_px": round(rpx, 1), "right_y_px": round(rpy, 1),
            "angle_px_deg": round(d["angle"], 1),
            # robot-frame fields (filled when a map is available)
            "left_x": "", "left_y": "", "right_x": "", "right_y": "",
            "cx_mm": "", "cy_mm": "", "angle_deg": "",
        }
        if mapper is not None:
            lx, ly = map_point(mapper, lpx, lpy)
            rx, ry = map_point(mapper, rpx, rpy)
            cxm, cym = map_point(mapper, cx, cy)
            rec["left_x"] = round(lx + ox, 3)
            rec["left_y"] = round(ly + oy, 3)
            rec["right_x"] = round(rx + ox, 3)
            rec["right_y"] = round(ry + oy, 3)
            rec["cx_mm"] = round(cxm + ox, 3)
            rec["cy_mm"] = round(cym + oy, 3)
            rec["angle_deg"] = round(_robot_angle(mapper, map_point, cx, cy,
                                                  d["points"]), 1)
        recs.append(rec)
    return recs


# columns for the cylinder CSV / live table (robot-frame first, like the ring
# app's records; pixel values kept for traceability)
CYL_COLUMNS = ["id", "class_name", "confidence",
               "left_x", "left_y", "right_x", "right_y", "angle_deg",
               "cx_mm", "cy_mm",
               "cx_px", "cy_px", "angle_px_deg"]


def annotate_cylinders(img_bgr, dets, cfg=None, mapper=None):
    """Draw the cylinders (box, long axis, left/right dots) and return
    (vis, records)."""
    vis = img_bgr.copy()
    recs = cylinder_records(dets, cfg, mapper)
    for d, rec in zip(dets, recs):
        x, y, w, h = int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cx, cy = d["cx"], d["cy"]
        dx, dy = _long_axis_dir(d["points"])
        L = 0.5 * max(w, h)
        p1 = (int(cx - L * dx), int(cy - L * dy))
        p2 = (int(cx + L * dx), int(cy + L * dy))
        cv2.line(vis, p1, p2, (0, 165, 255), 2)          # long axis
        cv2.circle(vis, (x, int(cy)), 4, (255, 0, 0), -1)        # left mid
        cv2.circle(vis, (x + w, int(cy)), 4, (0, 0, 255), -1)    # right mid
        label = "%d %.0fdeg" % (rec["id"], rec["angle_deg"]
                                if rec["angle_deg"] != "" else d["angle"])
        cv2.putText(vis, label, (x, max(12, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return vis, recs
