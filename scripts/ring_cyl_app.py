# ============================================================
# ring_cyl_app.py  -  ring + cylinder version (separate app)
# ------------------------------------------------------------
# A SEPARATE version of the live app that adds the cylinder/"pin" component
# ON TOP of the existing ring app, without touching ring_app.py. It reuses the
# whole ring GUI, calibration, TCP, CSV and folder-watching by subclassing, and
# adds a cylinder branch driven by a marker file in the watch folder:
#
#   mode.txt  containing  "cylinder"  -> run the cylinder (pin) model
#   mode.txt  containing  "circle"    -> run the existing ring pipeline
#   (missing / anything else)         -> circle  (so nothing changes by default)
#
# Cylinder detection + table come from scripts/cylinder.py (ported detection +
# table logic only). Left/right/center are mapped to robot mm through the SAME
# calibration map the ring app uses, and the angle is reported in the robot
# frame. Cylinder CSV/latest are written next to the ring ones with a "_cyl"
# suffix so the two never mix.
#
# Run:  python scripts/ring_cyl_app.py
# The pure-ring app (scripts/ring_app.py) and its .exe are unchanged.
# ============================================================

import os
import time
import csv

import cv2
import numpy as np

import ring_app
import cylinder as C

# the original Worker/App we extend (captured before we swap Worker in)
_BaseWorker = ring_app.Worker
_BaseApp = ring_app.App

CYL_DEFAULTS = {
    "cylinder_model": "",                 # path to the pin/cylinder best.pt
    "cyl_conf": 0.25,
    "cyl_imgsz": 640,
    "cyl_tcp_format": "{id},{left_x},{left_y},{right_x},{right_y},{angle_deg}",
}


def _cyl_path(path):
    """foo.csv -> foo_cyl.csv, so cylinder rows never mix with ring rows."""
    base, ext = os.path.splitext(path)
    return base + "_cyl" + (ext or ".csv")


# ---------------- worker: add a cylinder branch -----------------------------

class CylWorker(_BaseWorker):
    def __init__(self, q, cfg, tcp=None):
        for k, v in CYL_DEFAULTS.items():
            cfg.setdefault(k, v)
        super().__init__(q, cfg, tcp)
        self._cyl = None
        self._cyl_loaded = None

    def _load_cyl(self):
        path = self.cfg.get("cylinder_model", "")
        if not path or not os.path.isfile(path):
            raise RuntimeError("cylinder model not set/found: %r "
                               "(Configuration -> Cylinder -> model .pt)" % path)
        if self._cyl is None or self._cyl_loaded != path:
            self.log("Loading cylinder model %s ..." % os.path.basename(path))
            self._cyl = C.CylinderDetector().load(path, "segment")
            self._cyl_loaded = path
            self.log("Cylinder model ready.")
        return self._cyl

    def _mode(self):
        return C.read_mode(self.cfg.get("watch_folder", ""), "circle")

    def _process(self, path, allow_avg=True):
        if self._mode() != "cylinder":
            return super()._process(path, allow_avg)     # existing ring pipeline
        name = os.path.basename(path)
        try:
            img = cv2.imread(path)
            if img is None:
                self.log("cannot read %s" % name)
                return
            t0 = time.time()
            det = self._load_cyl()
            dets = det.detect(img, float(self.cfg.get("cyl_conf", 0.25)))
            detect_ms = (time.time() - t0) * 1000.0
            mapper = ring_app.load_mapper(self.cfg.get("homography_file", ""))
            vis, recs = C.annotate_cylinders(img, dets, self.cfg, mapper)
            total_ms = (time.time() - t0) * 1000.0
            self._cyl_write_csv(name, recs)
            self._cyl_write_latest(name, recs)
            self._cyl_tcp(name, recs)
            self.q.put(("result", {"name": name, "image": vis, "rings": recs,
                                   "mode": "cylinder",
                                   "has_map": mapper is not None,
                                   "empty": len(recs) == 0,
                                   "detect_ms": detect_ms, "total_ms": total_ms}))
            self.log("%s -> %d pin(s) [cylinder]%s  [detect %.0f ms]"
                     % (name, len(recs),
                        "" if mapper is not None else "  (no calibration loaded!)",
                        detect_ms))
        except Exception as e:
            self.log("ERROR (cylinder) on %s: %s" % (name, e))

    def _cyl_tcp(self, image, recs):
        if not self.tcp:
            return
        if not recs:
            msg = self.cfg.get("tcp_empty_message", "EMPTY")
            if msg:
                self.tcp.broadcast(msg + "\n")
            return
        fmt = self.cfg.get("cyl_tcp_format", CYL_DEFAULTS["cyl_tcp_format"])
        lines = []
        for r in recs:
            if r["left_x"] == "":            # only send pins with a robot mapping
                continue
            try:
                lines.append(fmt.format(image=image, **r))
            except Exception:
                lines.append("%s,%s,%s,%s,%s,%s" % (
                    r["id"], r["left_x"], r["left_y"],
                    r["right_x"], r["right_y"], r["angle_deg"]))
        if lines:
            self.tcp.broadcast("\n".join(lines) + "\n")

    def _cyl_rows(self, image, recs):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        return [[ts, image] + [r.get(c, "") for c in C.CYL_COLUMNS] for r in recs]

    def _cyl_write_csv(self, image, recs):
        path = self.cfg.get("output_csv", "")
        if not path:
            return
        path = _cyl_path(path)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["timestamp", "image"] + C.CYL_COLUMNS)
                w.writerows(self._cyl_rows(image, recs))
        except Exception as e:
            self.log("cylinder CSV write failed: %s" % e)

    def _cyl_write_latest(self, image, recs):
        path = self.cfg.get("latest_csv", "")
        if not path:
            return
        path = _cyl_path(path)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "image"] + C.CYL_COLUMNS)
                w.writerows(self._cyl_rows(image, recs))
        except Exception as e:
            self.log("cylinder latest CSV write failed: %s" % e)


# ---------------- app: add a Cylinder config tab + cylinder table -----------

def _find_notebook(widget):
    from tkinter import ttk
    for ch in widget.winfo_children():
        if isinstance(ch, ttk.Notebook):
            return ch
    return None


# the ring table columns, to restore when switching back from cylinder mode
_RING_COLS = ("id", "x_px", "y_px", "dia_px", "robot_x", "robot_y", "dia_mm",
              "in_dia_px", "in_dia_mm")
_RING_W = (28, 50, 50, 52, 78, 78, 64, 60, 62)
_CYL_COLS = ("id", "left_x", "left_y", "right_x", "right_y", "angle_deg",
             "cx_mm", "cy_mm")
_CYL_W = (28, 72, 72, 72, 72, 66, 72, 72)


class CylApp(_BaseApp):
    def _build_config(self, page):
        super()._build_config(page)
        for k, v in CYL_DEFAULTS.items():
            self.cfg.setdefault(k, v)
        from tkinter import ttk
        nb = _find_notebook(page)
        if nb is None:
            return
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Cylinder  ")
        self._config_row(tab, 0, "cylinder_model", "Cylinder model (.pt)", "openfile")
        self._config_row(tab, 1, "cyl_conf", "Cylinder confidence", "text")
        self._config_row(tab, 2, "cyl_imgsz", "Cylinder imgsz", "text")
        self._config_row(tab, 3, "cyl_tcp_format", "Cylinder TCP line", "text")
        ttk.Label(tab, foreground="#555", wraplength=580, justify="left",
                  text=("Mode is chosen by a file 'mode.txt' in the watch folder: "
                        "put 'cylinder' to run this model, 'circle' (or no file) to "
                        "run the ring pipeline. Cylinder output = left/right/center "
                        "in robot mm + angle (robot frame); it is written to the "
                        "output/latest CSV with a '_cyl' suffix, and sent over TCP "
                        "using the line above. Uses the SAME calibration map.")
                  ).grid(row=5, column=1, sticky="w", pady=(12, 0))

    def _read_fields(self):
        ok = super()._read_fields()
        if not ok:
            return False
        from tkinter import messagebox
        try:
            self.cfg["cylinder_model"] = self.vars["cylinder_model"].get()
            self.cfg["cyl_conf"] = float(self.vars["cyl_conf"].get() or 0.25)
            self.cfg["cyl_imgsz"] = int(self.vars["cyl_imgsz"].get() or 640)
            self.cfg["cyl_tcp_format"] = (self.vars["cyl_tcp_format"].get()
                                          or CYL_DEFAULTS["cyl_tcp_format"])
        except ValueError as e:
            messagebox.showerror("Invalid value", "Cylinder settings: %s" % e)
            return False
        return ok

    def _set_columns(self, cols, widths, mode):
        if getattr(self, "_tree_mode", "ring") == mode:
            return
        import tkinter as tk
        self.tree.config(columns=cols)
        for c, w in zip(cols, widths):
            self.tree.heading(c, text=c,
                              command=lambda cc=c: self._sort_tree(cc))
            self.tree.column(c, width=w, anchor=tk.CENTER)
        self._tree_mode = mode

    def _show(self, res):
        if res.get("mode") == "cylinder":
            self._show_cyl(res)
        else:
            self._set_columns(_RING_COLS, _RING_W, "ring")
            super()._show(res)

    def _show_cyl(self, res):
        import tkinter as tk
        if res.get("empty"):
            self.banner.config(text="NO PIN  -  %s" % res["name"], bg="#cf222e")
        else:
            self.banner.config(text="%d pin(s)  -  %s"
                                    % (len(res["rings"]), res["name"]),
                               bg="#1a7f37")
        if "total_ms" in res:
            self.timing_lbl.config(text="detect %.0f ms | total %.0f ms"
                                        % (res.get("detect_ms", 0), res["total_ms"]))
        self._last_bgr = res["image"]
        self._render_live()
        self._set_columns(_CYL_COLS, _CYL_W, "cyl")
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in res["rings"]:
            self.tree.insert("", tk.END, values=(
                r["id"], r["left_x"], r["left_y"], r["right_x"], r["right_y"],
                r["angle_deg"], r["cx_mm"], r["cy_mm"]))


def main():
    # make the app build a CylWorker instead of the plain ring Worker
    ring_app.Worker = CylWorker
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    from PIL import Image, ImageTk
    # ring_app's module-level GUI globals are set inside its main(); set them
    # here too so the reused App methods find them.
    ring_app.tk = tk
    ring_app.ttk = ttk
    ring_app.filedialog = filedialog
    ring_app.messagebox = messagebox
    ring_app.simpledialog = simpledialog
    ring_app.Image = Image
    ring_app.ImageTk = ImageTk
    root = tk.Tk()
    root.title("Ring + Cylinder - live / configuration / calibration")
    CylApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
