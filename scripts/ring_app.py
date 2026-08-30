# ============================================================
# ring_app.py  -  standalone live GUI with configuration
# ------------------------------------------------------------
# A self-contained operator application. It does NOT modify or depend on
# the other scripts - detection, homography, offset, CSV logging and the
# GUI all live here - so you can run/change it without touching the rest
# of the project.
#
# Two tabs:
#   Live           annotated image + results table + log, Start / Stop
#   Configuration  watch folder, poll time, output CSV, XY offset,
#                  calibration (homography) file - saved to
#                  config/app_config.json and remembered next launch
#
# Robot XY = homography(ring pixel) + (offset_x, offset_y).
#
# Install once:  pip install ultralytics opencv-python numpy pillow
# Run:           python scripts/ring_app.py     (or press Run in PyCharm)
# ============================================================

import os
import csv
import json
import time
import queue
import threading

import cv2
import numpy as np
from ultralytics import FastSAM

# tkinter / Pillow are imported lazily in main() so the detection worker
# can be imported and tested without a display.
tk = ttk = filedialog = messagebox = Image = ImageTk = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(ROOT, "config", "app_config.json")

# -------- detection tuning (independent copy; edit freely) --------
MODEL           = "FastSAM-x.pt"
CONF, IOU       = 0.20, 0.7
MIN_AREA_FRAC   = 0.004
MAX_AREA_FRAC   = 0.25
MIN_CIRC        = 0.75
MIN_RADIUS_FRAC = 0.03
# -----------------------------------------------------------------

DEFAULT_CONFIG = {
    "watch_folder": os.path.join(ROOT, "data", "incoming"),
    "poll_seconds": 0.5,
    "output_csv": os.path.join(ROOT, "data", "results.csv"),
    "offset_x": 0.0,
    "offset_y": 0.0,
    "homography_file": os.path.join(ROOT, "config", "robot_map.json"),
    "image_ext": ".bmp",
}


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
    """Return 3x3 H (pixel->robot) or None if not available."""
    try:
        return np.array(json.load(open(path))["H"], dtype=np.float64)
    except Exception:
        return None


# ---------------- detection (self-contained) ----------------

class Detector:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            self.model = FastSAM(MODEL)

    def find_rings(self, img):
        H, W = img.shape[:2]
        min_r = max(5, (H * W) ** 0.5 * MIN_RADIUS_FRAC)
        res = self.model(img, device="cpu", retina_masks=True, imgsz=1024,
                         conf=CONF, iou=IOU, verbose=False)[0]
        rings = []
        if res.masks is None:
            return rings
        for m in res.masks.data.cpu().numpy():
            mask = (m > 0.5).astype(np.uint8)
            a = int(mask.sum())
            if a < MIN_AREA_FRAC * H * W or a > MAX_AREA_FRAC * H * W:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            c = max(cnts, key=cv2.contourArea)
            (x, y), r = cv2.minEnclosingCircle(c)
            peri = cv2.arcLength(c, True)
            circ = 4 * np.pi * cv2.contourArea(c) / (peri * peri) if peri else 0
            if circ < MIN_CIRC or r < min_r:
                continue
            if any((x - rx) ** 2 + (y - ry) ** 2 < (0.5 * max(r, rr)) ** 2
                   for rx, ry, rr in rings):
                continue
            rings.append((x, y, r))
        return sorted(rings, key=lambda t: (t[1], t[0]))


# ---------------- background worker ----------------

class Worker(threading.Thread):
    def __init__(self, q, cfg):
        super().__init__(daemon=True)
        self.q = q
        self.cfg = cfg                 # shared dict; GUI updates it on Save
        self.det = Detector()
        self.watching = threading.Event()
        self.stop_flag = threading.Event()
        self.jobs = queue.Queue()
        self._done = set()
        self._sizes = {}

    def log(self, m):
        self.q.put(("log", m))

    def run(self):
        try:
            self.log("Loading FastSAM model (first run downloads it)...")
            self.det.load()
            self.log("Model ready.")
        except Exception as e:
            self.log("ERROR loading model: %s" % e)
        self.q.put(("ready", None))
        while not self.stop_flag.is_set():
            try:
                self._process(self.jobs.get_nowait())
                continue
            except queue.Empty:
                pass
            if self.watching.is_set():
                self._scan()
            time.sleep(max(0.05, float(self.cfg.get("poll_seconds", 0.5))))

    def start_watching(self):
        self._done.clear()
        self._sizes.clear()
        folder = self.cfg["watch_folder"]
        try:
            for n in os.listdir(folder):
                if self._is_img(n):
                    self._done.add(n)
        except OSError:
            pass
        self.watching.set()
        self.log("Watching %s" % folder)

    def stop_watching(self):
        self.watching.clear()
        self.log("Stopped watching.")

    def submit(self, path):
        self.jobs.put(path)

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
        for n in sorted(names):
            if n in self._done:
                continue
            path = os.path.join(folder, n)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if self._sizes.get(n) != size:
                self._sizes[n] = size
                continue
            self._process(path)
            self._done.add(n)
            self._sizes.pop(n, None)

    def _process(self, path):
        name = os.path.basename(path)
        try:
            img = cv2.imread(path)
            if img is None:
                self.log("cannot read %s" % name)
                return
            rings = self.det.find_rings(img)

            H = load_homography(self.cfg.get("homography_file", ""))
            ox = float(self.cfg.get("offset_x", 0.0))
            oy = float(self.cfg.get("offset_y", 0.0))

            out = []
            vis = img.copy()
            for i, (x, y, r) in enumerate(rings):
                rec = {"id": i + 1, "x": round(x, 1), "y": round(y, 1),
                       "diameter": round(2 * r, 1),
                       "robot_x": "", "robot_y": ""}
                label = str(i + 1)
                if H is not None:
                    q = cv2.perspectiveTransform(
                        np.array([[x, y]], np.float64).reshape(-1, 1, 2),
                        H).reshape(2)
                    rec["robot_x"] = round(float(q[0]) + ox, 3)
                    rec["robot_y"] = round(float(q[1]) + oy, 3)
                    label += " (%.1f,%.1f)" % (rec["robot_x"], rec["robot_y"])
                cv2.circle(vis, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.drawMarker(vis, (int(x), int(y)), (0, 0, 255),
                               cv2.MARKER_CROSS, 18, 2)
                cv2.putText(vis, label, (int(x) - 12, int(y) - int(r) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1,
                            cv2.LINE_AA)
                out.append(rec)

            self._write_csv(name, out)
            self.q.put(("result", {"name": name, "image": vis, "rings": out,
                                   "has_map": H is not None}))
            self.log("%s -> %d ring(s)%s" % (name, len(out),
                     "" if H is not None else "  (no homography loaded!)"))
        except Exception as e:
            self.log("ERROR on %s: %s" % (name, e))

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
        root.title("Ring Finder - live + configuration")
        root.geometry("1000x680")
        self.cfg = load_config()
        self.q = queue.Queue()
        self.worker = Worker(self.q, self.cfg)
        self.worker.start()
        self._imgtk = None

        nb = ttk.Notebook(root)
        nb.pack(fill=tk.BOTH, expand=True)
        self.live = ttk.Frame(nb)
        self.conf = ttk.Frame(nb)
        nb.add(self.live, text="  Live  ")
        nb.add(self.conf, text="  Configuration  ")
        self._build_live(self.live)
        self._build_config(self.conf)

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
        rows = [
            ("watch_folder", "Watch folder", "dir"),
            ("image_ext", "Image extension", "text"),
            ("poll_seconds", "Poll time (seconds)", "text"),
            ("output_csv", "Output CSV file", "savefile"),
            ("offset_x", "Offset X (mm)", "text"),
            ("offset_y", "Offset Y (mm)", "text"),
            ("homography_file", "Calibration (homography) file", "openfile"),
        ]
        frm = ttk.Frame(p, padding=14)
        frm.pack(fill=tk.X)
        for i, (key, label, kind) in enumerate(rows):
            ttk.Label(frm, text=label, width=26).grid(row=i, column=0,
                                                      sticky=tk.W, pady=5)
            var = tk.StringVar(value=str(self.cfg.get(key, "")))
            self.vars[key] = var
            ttk.Entry(frm, textvariable=var, width=56).grid(row=i, column=1,
                                                            pady=5)
            if kind == "dir":
                ttk.Button(frm, text="Browse", command=lambda k=key:
                           self._pick_dir(k)).grid(row=i, column=2, padx=6)
            elif kind == "openfile":
                ttk.Button(frm, text="Browse", command=lambda k=key:
                           self._pick_file(k, save=False)).grid(row=i, column=2,
                                                                padx=6)
            elif kind == "savefile":
                ttk.Button(frm, text="Browse", command=lambda k=key:
                           self._pick_file(k, save=True)).grid(row=i, column=2,
                                                               padx=6)

        self.map_info = ttk.Label(frm, text="", foreground="#555")
        self.map_info.grid(row=len(rows), column=1, sticky=tk.W, pady=(2, 8))
        self._refresh_map_info()

        btns = ttk.Frame(frm)
        btns.grid(row=len(rows) + 1, column=1, sticky=tk.W, pady=8)
        ttk.Button(btns, text="Save settings", command=self.save).pack(
            side=tk.LEFT)
        ttk.Button(btns, text="Reload calibration",
                   command=self._refresh_map_info).pack(side=tk.LEFT, padx=8)
        ttk.Label(frm, text="Robot XY = homography(pixel) + (offset X, offset Y). "
                            "Folder/poll changes apply on next Start.",
                  foreground="#777").grid(row=len(rows) + 2, column=1,
                                          sticky=tk.W)

    def _pick_dir(self, key):
        d = filedialog.askdirectory(initialdir=self.vars[key].get() or ROOT)
        if d:
            self.vars[key].set(d)

    def _pick_file(self, key, save):
        if save:
            f = filedialog.asksaveasfilename(
                initialdir=ROOT, defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("all", "*.*")])
        else:
            f = filedialog.askopenfilename(
                initialdir=ROOT, filetypes=[("JSON", "*.json"), ("all", "*.*")])
        if f:
            self.vars[key].set(f)
            if key == "homography_file":
                self._refresh_map_info()

    def _refresh_map_info(self):
        path = self.vars["homography_file"].get()
        try:
            d = json.load(open(path))
            H = d.get("H")
            info = "map OK" if H else "no H in file"
            if "rms_mm" in d:
                info += "  |  fit RMS %.2f mm" % d["rms_mm"]
            if "loo_rms_mm" in d:
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
            self.cfg["homography_file"] = self.vars["homography_file"].get()
        except ValueError as e:
            messagebox.showerror("Invalid value",
                                 "Poll time and offsets must be numbers.\n%s" % e)
            return
        save_config(self.cfg)      # worker reads self.cfg live
        self._refresh_map_info()
        messagebox.showinfo("Saved", "Settings saved to\n%s" % CONFIG_FILE)

    # ---- actions ----
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
            self.worker.submit(f)

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
        except queue.Empty:
            pass
        self.root.after(150, self._pump)

    def _log(self, m):
        self.logbox.config(state=tk.NORMAL)
        self.logbox.insert(tk.END, time.strftime("[%H:%M:%S] ") + m + "\n")
        self.logbox.see(tk.END)
        self.logbox.config(state=tk.DISABLED)

    def _show(self, res):
        try:
            rgb = cv2.cvtColor(res["image"], cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            im.thumbnail((560, 470))
            self._imgtk = ImageTk.PhotoImage(im)
            self.canvas.config(image=self._imgtk)
        except Exception as e:
            self._log("display error: %s" % e)
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in res["rings"]:
            self.tree.insert("", tk.END, values=(
                r["id"], r["x"], r["y"], r["diameter"],
                r["robot_x"], r["robot_y"]))


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
