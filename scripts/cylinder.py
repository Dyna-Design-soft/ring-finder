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

    def detect(self, img_bgr, conf=0.25, min_aspect=1.8):
        """img_bgr: HxWx3 BGR (OpenCV). Returns a list of detections, each a
        dict with pixel geometry: class_name, confidence, cx, cy, x, y, w, h,
        angle (deg, long-axis), aspect (rotated-rect long/short), points.

        min_aspect rejects detections that are not elongated enough to be a
        real cylinder/pin (a round washer or a belt patch scores ~1.0-1.6;
        genuine pins are ~2+). This is what keeps an EMPTY conveyor from
        false-detecting. Set to 0 to disable the shape gate."""
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
                d = self._pack(names, ci, c, x1, y1, x2, y2, points)
                if min_aspect and d["aspect"] < min_aspect:
                    continue                       # too round -> not a cylinder
                out.append(d)
        elif boxes is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = [float(v) for v in boxes.xyxy[i].tolist()]
                c = float(boxes.conf[i].item())
                ci = int(boxes.cls[i].item())
                d = self._pack(names, ci, c, x1, y1, x2, y2, [])
                if min_aspect and d["aspect"] < min_aspect:
                    continue
                out.append(d)
        return out

    @staticmethod
    def _pack(names, ci, conf, x1, y1, x2, y2, points):
        w, h = x2 - x1, y2 - y1
        bcx, bcy = x1 + w / 2.0, y1 + h / 2.0
        # Sub-pixel geometry from image moments of the whole mask outline (uses
        # every outline point, not just the extreme corners minAreaRect keys on),
        # so centre / angle / length / width are steadier at low resolution.
        pts = np.array(points, np.float32) if len(points) >= 3 else None
        M = cv2.moments(pts) if pts is not None else None
        if M is not None and abs(M["m00"]) > 1e-6:
            rcx = M["m10"] / M["m00"]
            rcy = M["m01"] / M["m00"]
            mu20 = M["mu20"] / M["m00"]
            mu02 = M["mu02"] / M["m00"]
            mu11 = M["mu11"] / M["m00"]
            a = 0.5 * math.atan2(2 * mu11, mu20 - mu02)     # principal axis
            ca, sa = math.cos(a), math.sin(a)
            pl = (pts[:, 0] - rcx) * ca + (pts[:, 1] - rcy) * sa   # along axis
            pw = -(pts[:, 0] - rcx) * sa + (pts[:, 1] - rcy) * ca  # across axis
            length = float(np.percentile(pl, 98) - np.percentile(pl, 2))
            width = float(np.percentile(pw, 98) - np.percentile(pw, 2))
            if width > length:                    # keep length = long side
                length, width = width, length
                a += math.pi / 2.0
        elif pts is not None:                     # degenerate polygon
            rect = cv2.minAreaRect(pts)
            (rcx, rcy), (rw, rh), ang = rect
            length, width = max(rw, rh), min(rw, rh)
            a = math.radians(ang if rw >= rh else ang + 90.0)
        else:                                     # box-only fallback
            rcx, rcy, length, width, a = bcx, bcy, max(w, h), min(w, h), 0.0
        aspect = length / max(1e-6, width)
        ldx, ldy = math.cos(a), math.sin(a)
        sdx, sdy = -ldy, ldx
        end1 = (rcx - length / 2 * ldx, rcy - length / 2 * ldy)
        end2 = (rcx + length / 2 * ldx, rcy + length / 2 * ldy)
        side1 = (rcx - width / 2 * sdx, rcy - width / 2 * sdy)
        side2 = (rcx + width / 2 * sdx, rcy + width / 2 * sdy)
        hx, hy = length / 2 * ldx, length / 2 * ldy
        wx, wy = width / 2 * sdx, width / 2 * sdy
        corners = [[rcx - hx - wx, rcy - hy - wy], [rcx + hx - wx, rcy + hy - wy],
                   [rcx + hx + wx, rcy + hy + wy], [rcx - hx + wx, rcy - hy + wy]]
        left, right = (end1, end2) if end1[0] <= end2[0] else (end2, end1)
        return {
            "class_name": names.get(ci, str(ci)) if ci is not None else "object",
            "confidence": conf,
            "x": x1, "y": y1, "w": w, "h": h,
            "cx": rcx, "cy": rcy,                 # sub-pixel centre
            "angle": orientation_angle(points),
            "aspect": float(aspect),
            "length_px": float(length), "width_px": float(width),
            "corners": corners,
            "left": left, "right": right, "side1": side1, "side2": side2,
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

    def dist_mm(pa, pb):
        a = map_point(mapper, pa[0], pa[1])
        b = map_point(mapper, pb[0], pb[1])
        return math.hypot(b[0] - a[0], b[1] - a[1])

    recs = []
    for i, d in enumerate(dets):
        cx, cy = d["cx"], d["cy"]
        (lpx, lpy), (rpx, rpy) = d["left"], d["right"]      # rotated ends
        rec = {
            "id": i + 1,
            "class_name": d["class_name"],
            "confidence": round(d["confidence"], 3) if d["confidence"] is not None else "",
            "cx_px": round(cx, 1), "cy_px": round(cy, 1),
            "left_x_px": round(lpx, 1), "left_y_px": round(lpy, 1),
            "right_x_px": round(rpx, 1), "right_y_px": round(rpy, 1),
            "width_px": round(d["width_px"], 1), "height_px": round(d["length_px"], 1),
            "angle_px_deg": round(d["angle"], 1),
            # robot-frame fields (filled when a map is available)
            "left_x": "", "left_y": "", "right_x": "", "right_y": "",
            "cx_mm": "", "cy_mm": "", "angle_deg": "",
            "width_mm": "", "height_mm": "",
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
            # height = length between the two ends; width = across the short axis
            rec["height_mm"] = round(dist_mm(d["left"], d["right"]), 3)
            rec["width_mm"] = round(dist_mm(d["side1"], d["side2"]), 3)
        recs.append(rec)
    return recs


# columns for the cylinder CSV / live table (robot-frame first, like the ring
# app's records; pixel values kept for traceability)
CYL_COLUMNS = ["id", "class_name", "confidence",
               "left_x", "left_y", "right_x", "right_y", "angle_deg",
               "width_mm", "height_mm", "cx_mm", "cy_mm",
               "width_px", "height_px", "cx_px", "cy_px", "angle_px_deg"]


def annotate_cylinders(img_bgr, dets, cfg=None, mapper=None):
    """Draw the cylinders (box, long axis, left/right dots) and return
    (vis, records)."""
    vis = img_bgr.copy()
    recs = cylinder_records(dets, cfg, mapper)
    for d, rec in zip(dets, recs):
        corners = np.array(d["corners"], np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [corners], True, (0, 200, 0), 2)    # rotated box hugs it
        (lx, ly), (rx, ry) = d["left"], d["right"]
        cv2.line(vis, (int(lx), int(ly)), (int(rx), int(ry)),
                 (0, 165, 255), 2)                              # long axis (ends)
        cv2.circle(vis, (int(lx), int(ly)), 4, (255, 0, 0), -1)   # left end
        cv2.circle(vis, (int(rx), int(ry)), 4, (0, 0, 255), -1)   # right end
        ang = rec["angle_deg"] if rec["angle_deg"] != "" else d["angle"]
        label = "%d %.0fdeg" % (rec["id"], ang)
        tx, ty = int(min(c[0][0] for c in corners)), int(min(c[0][1] for c in corners))
        cv2.putText(vis, label, (tx, max(12, ty - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return vis, recs
