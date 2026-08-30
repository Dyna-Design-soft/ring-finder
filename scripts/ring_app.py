# ============================================================
# ring_app.py  -  standalone live GUI with configuration + calibration
# ------------------------------------------------------------
# A self-contained operator application. It does NOT modify or depend on
# the other scripts - detection, homography, offset, CSV logging, the
# homography builder and the GUI all live here.
#
# Tabs:
#   Live           annotated image + results table + log, Start / Stop
#   Configuration  folder, image ext, poll time, output CSV, XY offset,
#                  calibration (homography) file, and the FastSAM
#                  detection parameters - saved to config/app_config.json
#   Calibration    build the pixel->robot homography from point pairs:
#                  grab an image (from the watch folder or a file), the
#                  ring pixel is detected, you type the robot XY for that
#                  ring and Add point; Fit & Save writes the homography.
#
# Robot XY = homography(ring pixel) + (offset_x, offset_y).
#
# Install once:  pip install ultralytics opencv-python numpy pillow
# Run:           python scripts/ring_app.py     (or press Run in PyCharm)
# ============================================================

import os
import csv
import glob
import json
import time
import queue
import socket
import threading

import cv2
import numpy as np
from ultralytics import FastSAM

# tkinter / Pillow imported lazily in main() so the worker stays testable.
tk = ttk = filedialog = messagebox = Image = ImageTk = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(ROOT, "config", "app_config.json")

DEFAULT_CONFIG = {
    "watch_folder": os.path.join(ROOT, "data", "incoming"),
    "image_ext": ".bmp",
    "poll_seconds": 0.5,
    "output_csv": os.path.join(ROOT, "data", "results.csv"),
    "offset_x": 0.0,
    "offset_y": 0.0,
    "homography_file": os.path.join(ROOT, "config", "robot_map.json"),
    # ---- FastSAM detection parameters ----
    "model": "FastSAM-x.pt",
    "conf": 0.20,
    "iou": 0.7,
    "min_area_frac": 0.004,
    "max_area_frac": 0.25,
    "min_circ": 0.75,
    "min_radius_frac": 0.03,
    # ---- TCP server (sends robot XY to the controller) ----
    "tcp_enabled": False,
    "tcp_host": "0.0.0.0",
    "tcp_port": 5000,
    "tcp_format": "{id},{x},{y}",   # one line per ring; \n appended
    # ---- watcher housekeeping ----
    "auto_delete_minutes": 10,      # delete images older than this (0 = off)
}

# detection parameter keys shown on the Configuration tab (numeric)
DET_PARAMS = ["conf", "iou", "min_area_frac", "max_area_frac",
              "min_circ", "min_radius_frac"]


def load_config():
    cfg = dict(DEFAULT_CONFIG)
    try:
        cfg.update(json.load(open(CONFIG_FILE)))
    except Exception:
        pass
    return cfg


def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    json.dump(cfg, open(CONFIG_FILE, "w"), indent=2)


def load_homography(path):
    try:
        return np.array(json.load(open(path))["H"], dtype=np.float64)
    except Exception:
        return None


def fit_homography(points, ransac_mm=6.0):
    """points = [(px, py, robot_x, robot_y), ...] -> dict with H + stats.
    Raises ValueError if fewer than 4 points."""
    if len(points) < 4:
        raise ValueError("need at least 4 points, have %d" % len(points))
    P = np.array([[p[0], p[1]] for p in points], np.float64)
    R = np.array([[p[2], p[3]] for p in points], np.float64)
    H, mask = cv2.findHomography(P, R, cv2.RANSAC, ransac_mm)
    if H is None:
        raise ValueError("homography fit failed")
    inl = [i for i in range(len(points)) if mask[i]]
    if len(inl) < 4:
        inl = list(range(len(points)))          # fall back to all
    Pi, Ri = P[inl], R[inl]
    H, _ = cv2.findHomography(Pi, Ri, 0)
    proj = cv2.perspectiveTransform(Pi.reshape(-1, 1, 2), H).reshape(-1, 2)
    res = np.linalg.norm(proj - Ri, axis=1)
    # leave-one-out only makes sense with >=5 points
    loo = None
    if len(inl) >= 5:
        errs = []
        for h in inl:
            tr = [i for i in inl if i != h]
            Hh, _ = cv2.findHomography(P[tr], R[tr], 0)
            q = cv2.perspectiveTransform(P[h].reshape(-1, 1, 2), Hh).reshape(2)
            errs.append(float(np.linalg.norm(q - R[h])))
        loo = float(np.sqrt(np.mean(np.square(errs))))
    return {
        "type": "homography_px_to_robot_mm",
        "H": H.tolist(),
        "points_used": len(inl),
        "excluded": [i + 1 for i in range(len(points)) if i not in inl],
        "rms_mm": float(np.sqrt((res ** 2).mean())),
        "loo_rms_mm": loo,
        "points": [{"id": i + 1, "px": p[0], "py": p[1],
                    "robot_x": p[2], "robot_y": p[3]}
                   for i, p in enumerate(points)],
    }


# ---------------- TCP server ----------------

class TcpServer:
    """Simple line-oriented TCP server. The robot controller connects as a
    client; each detection is broadcast to all connected clients."""

    def __init__(self, log):
        self.log = log
        self.sock = None
        self.clients = []
        self.lock = threading.Lock()
        self.running = False

    def start(self, host, port):
        self.stop()
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
            s.listen(5)
            s.settimeout(1.0)
        except OSError as e:
            self.log("TCP: cannot listen on %s:%s (%s)" % (host, port, e))
            return False
        self.sock = s
        self.running = True
        threading.Thread(target=self._accept, daemon=True).start()
        self.log("TCP server listening on %s:%s" % (host, port))
        return True

    def _accept(self):
        while self.running:
            try:
                c, addr = self.sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self.lock:
                self.clients.append(c)
            self.log("TCP client connected: %s:%s" % addr)

    def broadcast(self, text):
        if not self.running or not text:
            return
        data = text.encode("ascii", "replace")
        dead = []
        with self.lock:
            for c in self.clients:
                try:
                    c.sendall(data)
                except OSError:
                    dead.append(c)
            for c in dead:
                try:
                    c.close()
                except OSError:
                    pass
                self.clients.remove(c)
        if dead:
            self.log("TCP: dropped %d disconnected client(s)" % len(dead))

    def stop(self):
        self.running = False
        with self.lock:
            for c in self.clients:
                try:
                    c.close()
                except OSError:
                    pass
            self.clients = []
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


# ---------------- detection ----------------

class Detector:
    def __init__(self):
        self.model = None
        self.model_name = None

    def load(self, name):
        if self.model is None or name != self.model_name:
            self.model = FastSAM(name)
            self.model_name = name

    def find_rings(self, img, cfg):
        H, W = img.shape[:2]
        min_r = max(5, (H * W) ** 0.5 * float(cfg.get("min_radius_frac", 0.03)))
        conf = float(cfg.get("conf", 0.20))
        iou = float(cfg.get("iou", 0.7))
        min_af = float(cfg.get("min_area_frac", 0.004))
        max_af = float(cfg.get("max_area_frac", 0.25))
        min_circ = float(cfg.get("min_circ", 0.75))
        res = self.model(img, device="cpu", retina_masks=True, imgsz=1024,
                         conf=conf, iou=iou, verbose=False)[0]
        rings = []
        if res.masks is None:
            return rings
        for m in res.masks.data.cpu().numpy():
            mask = (m > 0.5).astype(np.uint8)
            a = int(mask.sum())
            if a < min_af * H * W or a > max_af * H * W:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(c)
            peri = cv2.arcLength(c, True)
            circ = 4 * np.pi * cv2.contourArea(c) / (peri * peri) if peri else 0
            if circ < min_circ or r < min_r:
                continue
            if any((x - rx) ** 2 + (y - ry) ** 2 < (0.5 * max(r, rr)) ** 2
                   for rx, ry, rr in rings):
                continue
            rings.append((x, y, r))
        return sorted(rings, key=lambda t: (t[1], t[0]))


def annotate(img, rings, cfg=None, H=None):
    """Draw rings; returns (vis, records)."""
    ox = float(cfg.get("offset_x", 0.0)) if cfg else 0.0
    oy = float(cfg.get("offset_y", 0.0)) if cfg else 0.0
    vis = img.copy()
    recs = []
    for i, (x, y, r) in enumerate(rings):
        rec = {"id": i + 1, "x": round(x, 1), "y": round(y, 1),
               "diameter": round(2 * r, 1), "robot_x": "", "robot_y": ""}
        label = str(i + 1)
        if H is not None:
            q = cv2.perspectiveTransform(
                np.array([[x, y]], np.float64).reshape(-1, 1, 2), H).reshape(2)
            rec["robot_x"] = round(float(q[0]) + ox, 3)
            rec["robot_y"] = round(float(q[1]) + oy, 3)
            label += " (%.1f,%.1f)" % (rec["robot_x"], rec["robot_y"])
        cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(vis, label, (int(x) - 12, int(y) - int(r) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        recs.append(rec)
    return vis, recs


# ---------------- background worker ----------------

class Worker(threading.Thread):
    def __init__(self, q, cfg, tcp=None):
        super().__init__(daemon=True)
        self.q = q
        self.cfg = cfg
        self.tcp = tcp
        self.det = Detector()
        self.watching = threading.Event()
        self.stop_flag = threading.Event()
        self.jobs = queue.Queue()          # (kind, path): "process" | "calib"
        self._seen = {}     # name -> (size, mtime) last observed (stability)
        self._proc = {}     # name -> mtime last processed (re-fire on change)
        self._last_cleanup = 0.0

    def log(self, m):
        self.q.put(("log", m))

    def run(self):
        try:
            self.log("Loading FastSAM model (first run downloads it)...")
            self.det.load(self.cfg.get("model", "FastSAM-x.pt"))
            self.log("Model ready.")
        except Exception as e:
            self.log("ERROR loading model: %s" % e)
        self.q.put(("ready", None))
        while not self.stop_flag.is_set():
            try:
                kind, path = self.jobs.get_nowait()
                if kind == "calib":
                    self._calib(path)
                else:
                    self._process(path)
                continue
            except queue.Empty:
                pass
            if self.watching.is_set():
                self._scan()
            time.sleep(max(0.05, float(self.cfg.get("poll_seconds", 0.5))))

    def start_watching(self):
        self._seen.clear()
        self._proc.clear()
        # mark pre-existing files as already processed (at their current mtime);
        # a later overwrite changes the mtime and re-fires them.
        folder = self.cfg["watch_folder"]
        try:
            for n in os.listdir(folder):
                if self._is_img(n):
                    try:
                        self._proc[n] = os.path.getmtime(os.path.join(folder, n))
                    except OSError:
                        pass
        except OSError:
            pass
        self.watching.set()
        self.log("Watching %s" % folder)

    def stop_watching(self):
        self.watching.clear()
        self.log("Stopped watching.")

    def submit(self, path, kind="process"):
        self.jobs.put((kind, path))

    def _is_img(self, n):
        ext = self.cfg.get("image_ext", ".bmp").lower()
        low = n.lower()
        return low.endswith(ext) and not low.endswith("_rings.png")

    def _scan(self):
        folder = self.cfg["watch_folder"]
        try:
            names = [n for n in os.listdir(folder) if self._is_img(n)]
        except OSError:
            return
        present = set()
        for n in sorted(names):
            present.add(n)
            path = os.path.join(folder, n)
            try:
                st = os.stat(path)
            except OSError:
                continue
            key = (st.st_size, st.st_mtime)
            if self._seen.get(n) != key:      # still being written -> wait a poll
                self._seen[n] = key
                continue
            # stable. process if new name OR the file was modified/overwritten
            if self._proc.get(n) == st.st_mtime:
                continue
            self._process(path)
            self._proc[n] = st.st_mtime
        # forget files that disappeared, so a reappearing name re-fires
        for gone in [n for n in self._seen if n not in present]:
            self._seen.pop(gone, None)
            self._proc.pop(gone, None)
        self._cleanup(folder)

    def _cleanup(self, folder):
        """Delete images (and their _rings outputs) older than N minutes."""
        try:
            mins = float(self.cfg.get("auto_delete_minutes", 0) or 0)
        except (TypeError, ValueError):
            mins = 0
        if mins <= 0:
            return
        now = time.time()
        if now - self._last_cleanup < 20:     # throttle: at most every 20 s
            return
        self._last_cleanup = now
        cutoff = now - mins * 60
        try:
            entries = os.listdir(folder)
        except OSError:
            return
        removed = 0
        for n in entries:
            if not self._is_img(n):
                continue
            path = os.path.join(folder, n)
            try:
                if os.path.getmtime(path) >= cutoff:
                    continue
                os.remove(path)
                removed += 1
                self._seen.pop(n, None)
                self._proc.pop(n, None)
                # remove matching annotated/JSON outputs if present
                stem = os.path.splitext(path)[0]
                for ext in ("_rings.png", "_rings.json"):
                    if os.path.exists(stem + ext):
                        os.remove(stem + ext)
            except OSError:
                pass
        if removed:
            self.log("housekeeping: deleted %d image(s) older than %g min"
                     % (removed, mins))

    def _detect(self, path):
        self.det.load(self.cfg.get("model", "FastSAM-x.pt"))
        img = cv2.imread(path)
        if img is None:
            self.log("cannot read %s" % os.path.basename(path))
            return None, None
        return img, self.det.find_rings(img, self.cfg)

    def _process(self, path):
        name = os.path.basename(path)
        try:
            img, rings = self._detect(path)
            if img is None:
                return
            H = load_homography(self.cfg.get("homography_file", ""))
            vis, recs = annotate(img, rings, self.cfg, H)
            self._write_csv(name, recs)
            self._tcp_send(name, recs)
            self.q.put(("result", {"name": name, "image": vis, "rings": recs,
                                   "has_map": H is not None}))
            self.log("%s -> %d ring(s)%s" % (name, len(recs),
                     "" if H is not None else "  (no homography loaded!)"))
        except Exception as e:
            self.log("ERROR on %s: %s" % (name, e))

    def _calib(self, path):
        name = os.path.basename(path)
        try:
            img, rings = self._detect(path)
            if img is None:
                return
            vis, _ = annotate(img, rings, self.cfg, None)
            self.q.put(("calib_result", {"name": name, "image": vis,
                                         "rings": [(round(x, 1), round(y, 1),
                                                    round(r, 1))
                                                   for x, y, r in rings]}))
            self.log("calib image %s -> %d ring(s)" % (name, len(rings)))
        except Exception as e:
            self.log("calib ERROR on %s: %s" % (name, e))

    def _tcp_send(self, image, rings):
        if not self.tcp:
            return
        fmt = self.cfg.get("tcp_format", "{id},{x},{y}")
        lines = []
        for r in rings:
            if r["robot_x"] == "":       # only send rings with a robot XY
                continue
            try:
                lines.append(fmt.format(id=r["id"], x=r["robot_x"],
                                        y=r["robot_y"], dia=r["diameter"],
                                        image=image))
            except Exception:
                lines.append("%s,%s,%s" % (r["id"], r["robot_x"], r["robot_y"]))
        if lines:
            self.tcp.broadcast("\n".join(lines) + "\n")

    def _write_csv(self, image, rings):
        path = self.cfg.get("output_csv", "")
        if not path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp", "image", "ring_id", "x_px",
                                "y_px", "diameter_px", "robot_x", "robot_y"])
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                for r in rings:
                    w.writerow([ts, image, r["id"], r["x"], r["y"],
                                r["diameter"], r["robot_x"], r["robot_y"]])
        except Exception as e:
            self.log("CSV write failed: %s" % e)


# ---------------- GUI ----------------

class App:
    def __init__(self, root):
        self.root = root
        root.title("Ring Finder - live / configuration / calibration")
        root.geometry("1040x720")
        self.cfg = load_config()
        self.q = queue.Queue()
        self.tcp = TcpServer(lambda m: self.q.put(("log", m)))
        self.worker = Worker(self.q, self.cfg, self.tcp)
        self.worker.start()
        if self.cfg.get("tcp_enabled"):
            self.tcp.start(self.cfg.get("tcp_host", "0.0.0.0"),
                           self.cfg.get("tcp_port", 5000))
        self._imgtk = None
        self._calib_imgtk = None
        self._calib_rings = []        # detected rings in current calib image
        self.points = []              # collected (px,py,rx,ry)

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)
        self.live = ttk.Frame(nb)
        self.conf = ttk.Frame(nb)
        self.calib = ttk.Frame(nb)
        nb.add(self.live, text="  Live  ")
        nb.add(self.conf, text="  Configuration  ")
        nb.add(self.calib, text="  Calibration  ")
        self._build_live(self.live)
        self._build_config(self.conf)
        self._build_calib(self.calib)

        self.root.after(150, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---- Live tab ----
    def _build_live(self, p):
        bar = ttk.Frame(p, padding=6)
        bar.pack(side=tk.TOP, fill=tk.X)
        self.start_btn = ttk.Button(bar, text="Start", command=self.start,
                                    state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Open image...", command=self.open_image).pack(
            side=tk.LEFT, padx=(10, 0))
        self.status = ttk.Label(bar, text="Loading model...",
                                foreground="#b26b00")
        self.status.pack(side=tk.RIGHT)

        mid = ttk.Frame(p)
        mid.pack(fill=tk.BOTH, expand=True)
        left = ttk.LabelFrame(mid, text="Latest image", padding=6)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas = tk.Label(left, background="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        right = ttk.LabelFrame(mid, text="Rings (pixel + robot mm)", padding=6)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("id", "x_px", "y_px", "dia_px", "robot_x", "robot_y")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (34, 58, 58, 62, 92, 92)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        logf = ttk.LabelFrame(p, text="Log", padding=4)
        logf.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(0, 6))
        self.logbox = tk.Text(logf, height=6, state=tk.DISABLED,
                              background="#111", foreground="#ddd")
        self.logbox.pack(fill=tk.X)

    # ---- Configuration tab ----
    def _build_config(self, p):
        self.vars = {}
        general = [
            ("watch_folder", "Watch folder", "dir"),
            ("image_ext", "Image extension", "text"),
            ("poll_seconds", "Poll time (seconds)", "text"),
            ("output_csv", "Output CSV file", "savefile"),
            ("offset_x", "Offset X (mm)", "text"),
            ("offset_y", "Offset Y (mm)", "text"),
            ("auto_delete_minutes", "Auto-delete images older than (min)", "text"),
            ("homography_file", "Calibration (homography) file", "openfile"),
        ]
        frm = ttk.LabelFrame(p, text="General", padding=12)
        frm.pack(fill=tk.X, padx=10, pady=(10, 6))
        for i, (key, label, kind) in enumerate(general):
            self._config_row(frm, i, key, label, kind)
        self.map_info = ttk.Label(frm, text="", foreground="#555")
        self.map_info.grid(row=len(general), column=1, sticky=tk.W, pady=(2, 4))
        self._refresh_map_info()

        det = ttk.LabelFrame(p, text="Detection parameters (FastSAM)", padding=12)
        det.pack(fill=tk.X, padx=10, pady=6)
        self._config_row(det, 0, "model", "MODEL", "text")
        for i, key in enumerate(DET_PARAMS):
            self._config_row(det, i + 1, key, key.upper(), "text")
        ttk.Label(det, text="MODEL change takes effect on the next image "
                            "(model reloads).", foreground="#777").grid(
            row=len(DET_PARAMS) + 1, column=1, sticky=tk.W, pady=(2, 0))

        tcp = ttk.LabelFrame(p, text="TCP server (sends robot XY to controller)",
                             padding=12)
        tcp.pack(fill=tk.X, padx=10, pady=6)
        self.tcp_enabled_var = tk.BooleanVar(value=bool(self.cfg.get("tcp_enabled")))
        ttk.Checkbutton(tcp, text="Enable TCP server",
                        variable=self.tcp_enabled_var).grid(
            row=0, column=1, sticky=tk.W, pady=4)
        self._config_row(tcp, 1, "tcp_host", "Host / bind address", "text")
        self._config_row(tcp, 2, "tcp_port", "Port", "text")
        self._config_row(tcp, 3, "tcp_format", "Line format", "text")
        ttk.Label(tcp, text="Placeholders: {id} {x} {y} {dia} {image}. "
                            "One line per ring; newline appended.",
                  foreground="#777").grid(row=4, column=1, sticky=tk.W)
        self.tcp_info = ttk.Label(tcp, text="", foreground="#555")
        self.tcp_info.grid(row=5, column=1, sticky=tk.W, pady=(2, 0))

        btns = ttk.Frame(p, padding=(10, 4))
        btns.pack(fill=tk.X)
        ttk.Button(btns, text="Save settings", command=self.save).pack(
            side=tk.LEFT)
        ttk.Button(btns, text="Reload calibration",
                   command=self._refresh_map_info).pack(side=tk.LEFT, padx=8)
        ttk.Label(btns, text="Robot XY = homography(pixel) + (offset X, offset Y). "
                            "Folder/poll apply on next Start.",
                  foreground="#777").pack(side=tk.LEFT, padx=10)

    def _config_row(self, parent, i, key, label, kind):
        ttk.Label(parent, text=label, width=26).grid(row=i, column=0,
                                                     sticky=tk.W, pady=4)
        var = tk.StringVar(value=str(self.cfg.get(key, "")))
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var, width=54).grid(row=i, column=1, pady=4)
        if kind == "dir":
            ttk.Button(parent, text="Browse", command=lambda k=key:
                       self._pick_dir(k)).grid(row=i, column=2, padx=6)
        elif kind == "openfile":
            ttk.Button(parent, text="Browse", command=lambda k=key:
                       self._pick_file(k, save=False)).grid(row=i, column=2, padx=6)
        elif kind == "savefile":
            ttk.Button(parent, text="Browse", command=lambda k=key:
                       self._pick_file(k, save=True)).grid(row=i, column=2, padx=6)

    # ---- Calibration tab ----
    def _build_calib(self, p):
        top = ttk.Frame(p, padding=8)
        top.pack(fill=tk.X)
        ttk.Button(top, text="Grab latest from folder",
                   command=self.calib_grab).pack(side=tk.LEFT)
        ttk.Button(top, text="Load image...",
                   command=self.calib_load).pack(side=tk.LEFT, padx=6)
        ttk.Label(top, text="Detected ring:").pack(side=tk.LEFT, padx=(14, 2))
        self.calib_ring_sel = tk.Spinbox(top, from_=1, to=1, width=4)
        self.calib_ring_sel.pack(side=tk.LEFT)
        self.calib_pix = ttk.Label(top, text="pixel: -", foreground="#333")
        self.calib_pix.pack(side=tk.LEFT, padx=10)

        mid = ttk.Frame(p)
        mid.pack(fill=tk.BOTH, expand=True)
        left = ttk.LabelFrame(mid, text="Calibration image", padding=6)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.calib_canvas = tk.Label(left, background="#222")
        self.calib_canvas.pack(fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(mid, text="Collected points", padding=6)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        # robot XY entry
        entry = ttk.Frame(right)
        entry.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(entry, text="Robot X").grid(row=0, column=0, padx=2)
        self.rx_var = tk.StringVar()
        ttk.Entry(entry, textvariable=self.rx_var, width=10).grid(row=0, column=1)
        ttk.Label(entry, text="Robot Y").grid(row=0, column=2, padx=2)
        self.ry_var = tk.StringVar()
        ttk.Entry(entry, textvariable=self.ry_var, width=10).grid(row=0, column=3)
        ttk.Button(entry, text="Add point",
                   command=self.calib_add).grid(row=0, column=4, padx=6)

        cols = ("#", "px", "py", "robot_x", "robot_y")
        self.ptree = ttk.Treeview(right, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (30, 60, 60, 80, 80)):
            self.ptree.heading(c, text=c)
            self.ptree.column(c, width=w, anchor=tk.CENTER)
        self.ptree.pack(fill=tk.BOTH, expand=True)

        row2 = ttk.Frame(right)
        row2.pack(fill=tk.X, pady=6)
        ttk.Button(row2, text="Remove selected",
                   command=self.calib_remove).pack(side=tk.LEFT)
        ttk.Button(row2, text="Clear all",
                   command=self.calib_clear).pack(side=tk.LEFT, padx=6)
        self.calib_count = ttk.Label(row2, text="Points: 0  (need >= 4)",
                                     foreground="#b26b00")
        self.calib_count.pack(side=tk.RIGHT)

        row3 = ttk.Frame(right)
        row3.pack(fill=tk.X)
        ttk.Button(row3, text="Fit & Save homography",
                   command=self.calib_fit).pack(side=tk.LEFT)
        self.calib_result = ttk.Label(right, text="", foreground="#333")
        self.calib_result.pack(fill=tk.X, pady=(6, 0))

    # ---- config helpers ----
    def _pick_dir(self, key):
        d = filedialog.askdirectory(initialdir=self.vars[key].get() or ROOT)
        if d:
            self.vars[key].set(d)

    def _pick_file(self, key, save):
        if save:
            f = filedialog.asksaveasfilename(initialdir=ROOT,
                    defaultextension=".csv",
                    filetypes=[("CSV", "*.csv"), ("all", "*.*")])
        else:
            f = filedialog.askopenfilename(initialdir=ROOT,
                    filetypes=[("JSON", "*.json"), ("all", "*.*")])
        if f:
            self.vars[key].set(f)
            if key == "homography_file":
                self._refresh_map_info()

    def _refresh_map_info(self):
        path = self.vars["homography_file"].get()
        try:
            d = json.load(open(path))
            info = "map OK" if d.get("H") else "no H in file"
            if d.get("rms_mm") is not None:
                info += "  |  fit RMS %.2f mm" % d["rms_mm"]
            if d.get("loo_rms_mm") is not None:
                info += "  |  LOO %.2f mm" % d["loo_rms_mm"]
            self.map_info.config(text=info, foreground="#1a7f37")
        except Exception:
            self.map_info.config(text="calibration file not found / invalid",
                                 foreground="#cf222e")

    def save(self):
        try:
            self.cfg["watch_folder"] = self.vars["watch_folder"].get()
            self.cfg["image_ext"] = self.vars["image_ext"].get() or ".bmp"
            self.cfg["poll_seconds"] = float(self.vars["poll_seconds"].get())
            self.cfg["output_csv"] = self.vars["output_csv"].get()
            self.cfg["offset_x"] = float(self.vars["offset_x"].get())
            self.cfg["offset_y"] = float(self.vars["offset_y"].get())
            self.cfg["auto_delete_minutes"] = float(
                self.vars["auto_delete_minutes"].get())
            self.cfg["homography_file"] = self.vars["homography_file"].get()
            self.cfg["model"] = self.vars["model"].get() or "FastSAM-x.pt"
            for k in DET_PARAMS:
                self.cfg[k] = float(self.vars[k].get())
            self.cfg["tcp_enabled"] = bool(self.tcp_enabled_var.get())
            self.cfg["tcp_host"] = self.vars["tcp_host"].get() or "0.0.0.0"
            self.cfg["tcp_port"] = int(self.vars["tcp_port"].get())
            self.cfg["tcp_format"] = self.vars["tcp_format"].get() or "{id},{x},{y}"
        except ValueError as e:
            messagebox.showerror("Invalid value",
                                 "Numbers required for poll/offset/detection "
                                 "params and TCP port.\n%s" % e)
            return
        save_config(self.cfg)
        self._refresh_map_info()
        self._apply_tcp()
        messagebox.showinfo("Saved", "Settings saved to\n%s" % CONFIG_FILE)

    def _apply_tcp(self):
        if self.cfg.get("tcp_enabled"):
            ok = self.tcp.start(self.cfg.get("tcp_host", "0.0.0.0"),
                                self.cfg.get("tcp_port", 5000))
            self.tcp_info.config(
                text="listening on %s:%s" % (self.cfg["tcp_host"],
                                             self.cfg["tcp_port"]) if ok
                else "failed to start (port in use?)",
                foreground="#1a7f37" if ok else "#cf222e")
        else:
            self.tcp.stop()
            self.tcp_info.config(text="disabled", foreground="#555")

    # ---- live actions ----
    def start(self):
        self.worker.start_watching()
        self.status.config(text="Watching", foreground="#1a7f37")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop(self):
        self.worker.stop_watching()
        self.status.config(text="Idle", foreground="#333")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def open_image(self):
        f = filedialog.askopenfilename(
            initialdir=self.cfg.get("watch_folder", ROOT),
            filetypes=[("images", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff")])
        if f:
            self.worker.submit(f, "process")

    # ---- calibration actions ----
    def calib_grab(self):
        folder = self.cfg.get("watch_folder", "")
        ext = self.cfg.get("image_ext", ".bmp")
        files = [f for f in glob.glob(os.path.join(folder, "*" + ext))
                 if not f.endswith("_rings.png")]
        if not files:
            messagebox.showwarning("No image",
                                   "No %s image found in\n%s" % (ext, folder))
            return
        newest = max(files, key=os.path.getmtime)
        self.worker.submit(newest, "calib")

    def calib_load(self):
        f = filedialog.askopenfilename(
            initialdir=self.cfg.get("watch_folder", ROOT),
            filetypes=[("images", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff")])
        if f:
            self.worker.submit(f, "calib")

    def _selected_ring(self):
        if not self._calib_rings:
            return None
        try:
            idx = int(self.calib_ring_sel.get()) - 1
        except ValueError:
            idx = 0
        idx = max(0, min(idx, len(self._calib_rings) - 1))
        return self._calib_rings[idx]

    def calib_add(self):
        ring = self._selected_ring()
        if ring is None:
            messagebox.showwarning("No ring", "Grab/Load an image with a ring first.")
            return
        try:
            rx = float(self.rx_var.get())
            ry = float(self.ry_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "Robot X and Y must be numbers.")
            return
        px, py, _ = ring
        self.points.append((px, py, rx, ry))
        self._refresh_points()
        self.rx_var.set("")
        self.ry_var.set("")

    def calib_remove(self):
        sel = self.ptree.selection()
        if not sel:
            return
        idx = self.ptree.index(sel[0])
        del self.points[idx]
        self._refresh_points()

    def calib_clear(self):
        self.points = []
        self._refresh_points()

    def _refresh_points(self):
        for row in self.ptree.get_children():
            self.ptree.delete(row)
        for i, (px, py, rx, ry) in enumerate(self.points):
            self.ptree.insert("", tk.END, values=(i + 1, px, py, rx, ry))
        n = len(self.points)
        self.calib_count.config(
            text="Points: %d  (need >= 4)" % n,
            foreground="#1a7f37" if n >= 4 else "#b26b00")

    def calib_fit(self):
        try:
            result = fit_homography(self.points)
        except ValueError as e:
            messagebox.showerror("Cannot fit", str(e))
            return
        path = self.vars["homography_file"].get() or self.cfg["homography_file"]
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            json.dump(result, open(path, "w"), indent=2)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return
        loo = ("%.2f" % result["loo_rms_mm"]) if result["loo_rms_mm"] else "n/a"
        msg = ("Saved %s\npoints used: %d   fit RMS: %.2f mm   LOO: %s mm"
               % (os.path.basename(path), result["points_used"],
                  result["rms_mm"], loo))
        if result["excluded"]:
            msg += "\nexcluded (check robot XY): %s" % result["excluded"]
        self.calib_result.config(text=msg, foreground="#1a7f37")
        self._refresh_map_info()
        messagebox.showinfo("Homography saved", msg)

    def on_close(self):
        self.worker.stop_flag.set()
        self.root.after(100, self.root.destroy)

    # ---- pump ----
    def _pump(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "ready":
                    self.status.config(text="Idle (model ready)",
                                       foreground="#333")
                    self.start_btn.config(state=tk.NORMAL)
                elif kind == "result":
                    self._show(payload)
                elif kind == "calib_result":
                    self._show_calib(payload)
        except queue.Empty:
            pass
        self.root.after(150, self._pump)

    def _log(self, m):
        self.logbox.config(state=tk.NORMAL)
        self.logbox.insert(tk.END, time.strftime("[%H:%M:%S] ") + m + "\n")
        self.logbox.see(tk.END)
        self.logbox.config(state=tk.DISABLED)

    def _to_tk(self, bgr, box):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail(box)
        return ImageTk.PhotoImage(im)

    def _show(self, res):
        try:
            self._imgtk = self._to_tk(res["image"], (560, 470))
            self.canvas.config(image=self._imgtk)
        except Exception as e:
            self._log("display error: %s" % e)
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in res["rings"]:
            self.tree.insert("", tk.END, values=(
                r["id"], r["x"], r["y"], r["diameter"],
                r["robot_x"], r["robot_y"]))

    def _show_calib(self, res):
        self._calib_rings = res["rings"]
        try:
            self._calib_imgtk = self._to_tk(res["image"], (520, 430))
            self.calib_canvas.config(image=self._calib_imgtk)
        except Exception as e:
            self._log("calib display error: %s" % e)
        n = len(self._calib_rings)
        self.calib_ring_sel.config(from_=1, to=max(1, n))
        if n:
            px, py, _ = self._calib_rings[0]
            self.calib_pix.config(text="pixel: (%.1f, %.1f)  [%d ring(s)]"
                                       % (px, py, n))
        else:
            self.calib_pix.config(text="pixel: -  (no ring found)")


def main():
    global tk, ttk, filedialog, messagebox, Image, ImageTk
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import Image, ImageTk
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
