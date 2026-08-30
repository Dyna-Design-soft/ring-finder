# ============================================================
# gui.py  -  desktop GUI for ring detection + robot XY
# ------------------------------------------------------------
# An operator window (Tkinter):
#   * pick the folder the camera writes to, Start / Stop watching
#   * every new .bmp is detected and its robot XY computed automatically
#   * shows the annotated image, a results table, and a log
#   * "Open image..." runs one file on demand
#
# The model load and folder watching run on a background thread
# (ringworker.Worker), so the window never freezes; results come back
# through a thread-safe queue.
#
# Install once:  pip install ultralytics opencv-python numpy pillow
# Run:           python scripts/gui.py       (or press Run in PyCharm)
# ============================================================

import os
import time
import queue
import tkinter as tk
from tkinter import ttk, filedialog

from PIL import Image, ImageTk

from ringworker import Worker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WATCH = os.path.join(ROOT, "data", "incoming")


class App:
    def __init__(self, root):
        self.root = root
        root.title("Ring Finder - detection + robot XY")
        root.geometry("980x640")

        self.q = queue.Queue()
        self.worker = Worker(self.q)
        self.worker.start()
        self._imgtk = None

        bar = ttk.Frame(root, padding=8)
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text="Watch folder:").pack(side=tk.LEFT)
        self.folder_var = tk.StringVar(value=DEFAULT_WATCH)
        ttk.Entry(bar, textvariable=self.folder_var, width=52).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Browse", command=self.browse).pack(side=tk.LEFT)
        self.start_btn = ttk.Button(bar, text="Start", command=self.start,
                                    state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=(10, 2))
        self.stop_btn = ttk.Button(bar, text="Stop", command=self.stop,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)
        ttk.Button(bar, text="Open image...", command=self.open_image).pack(
            side=tk.LEFT, padx=(10, 0))

        self.status = ttk.Label(root, text="Loading model...", padding=(8, 2),
                                foreground="#b26b00")
        self.status.pack(side=tk.TOP, fill=tk.X)

        mid = ttk.Frame(root)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(mid, text="Latest image", padding=6)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas = tk.Label(left, background="#222")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(mid, text="Rings (pixel + robot mm)", padding=6)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        cols = ("id", "x_px", "y_px", "dia_px", "robot_x", "robot_y")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=10)
        for c, w in zip(cols, (36, 60, 60, 64, 90, 90)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

        logf = ttk.LabelFrame(root, text="Log", padding=4)
        logf.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(0, 6))
        self.logbox = tk.Text(logf, height=7, wrap=tk.WORD, state=tk.DISABLED,
                              background="#111", foreground="#ddd")
        self.logbox.pack(fill=tk.X)

        self.root.after(150, self._pump)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -- actions --
    def browse(self):
        d = filedialog.askdirectory(initialdir=self.folder_var.get() or ROOT)
        if d:
            self.folder_var.set(d)

    def start(self):
        self.worker.start_watching(self.folder_var.get())
        self.status.config(text="Watching: %s" % self.folder_var.get(),
                           foreground="#1a7f37")
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

    def stop(self):
        self.worker.stop_watching()
        self.status.config(text="Idle (model ready).", foreground="#333")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def open_image(self):
        f = filedialog.askopenfilename(
            initialdir=self.folder_var.get() or ROOT,
            filetypes=[("images", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff")])
        if f:
            self.worker.submit_file(f)

    def on_close(self):
        self.worker.stop_flag.set()
        self.root.after(100, self.root.destroy)

    # -- GUI update pump (runs on the Tk thread) --
    def _pump(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "ready":
                    self.status.config(text="Idle (model ready).",
                                       foreground="#333")
                    self.start_btn.config(state=tk.NORMAL)
                elif kind == "result":
                    self._show_result(payload)
        except queue.Empty:
            pass
        self.root.after(150, self._pump)

    def _log(self, msg):
        self.logbox.config(state=tk.NORMAL)
        self.logbox.insert(tk.END, time.strftime("[%H:%M:%S] ") + msg + "\n")
        self.logbox.see(tk.END)
        self.logbox.config(state=tk.DISABLED)

    def _show_result(self, res):
        try:
            im = Image.open(res["png"])
            im.thumbnail((560, 460))
            self._imgtk = ImageTk.PhotoImage(im)
            self.canvas.config(image=self._imgtk)
        except Exception as e:
            self._log("cannot show image: %s" % e)
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in res["rings"]:
            self.tree.insert("", tk.END, values=(
                r.get("id"), r.get("x"), r.get("y"), r.get("diameter"),
                r.get("robot_x", ""), r.get("robot_y", "")))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
