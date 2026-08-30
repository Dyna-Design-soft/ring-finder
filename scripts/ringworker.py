# ============================================================
# ringworker.py  -  background detection worker (no GUI deps)
# ------------------------------------------------------------
# Loads the FastSAM model once, watches a folder for new images, and runs
# on-demand single-file jobs. All results are handed back through a
# thread-safe queue as (kind, payload) tuples:
#   ("log",    "text")
#   ("ready",  None)                       model finished loading
#   ("result", {"name","png","rings"})     one image processed
#
# Kept free of tkinter/Pillow so it can be unit-tested and reused headless
# (e.g. by gui.py, or a web front-end).
# ============================================================

import os
import json
import time
import queue
import threading

import ring_finder as rf

IMAGE_EXTS = (".bmp",)
POLL_SECONDS = 0.5


class Worker(threading.Thread):
    def __init__(self, q, image_exts=IMAGE_EXTS, poll=POLL_SECONDS):
        super().__init__(daemon=True)
        self.q = q
        self.image_exts = tuple(e.lower() for e in image_exts)
        self.poll = poll
        self.folder = None
        self.watching = threading.Event()
        self.stop_flag = threading.Event()
        self.jobs = queue.Queue()
        self._done = set()
        self._sizes = {}
        self._H = None

    def log(self, msg):
        self.q.put(("log", msg))

    def run(self):
        try:
            self.log("Loading FastSAM model (first run downloads it)...")
            self._H = rf.load_homography(None)
            rf.get_model()
            self.log("Model ready.")
        except Exception as e:
            self.log("ERROR loading model: %s" % e)
        self.q.put(("ready", None))

        while not self.stop_flag.is_set():
            try:
                path = self.jobs.get_nowait()
                self._process(path)
                continue
            except queue.Empty:
                pass
            if self.watching.is_set():
                self._scan_once()
            time.sleep(self.poll)

    # ---- control ----
    def start_watching(self, folder):
        self.folder = folder
        self._done.clear()
        self._sizes.clear()
        try:
            for n in os.listdir(folder):
                if self._is_img(n):
                    self._done.add(n)          # ignore pre-existing -> only new fire
        except OSError:
            pass
        self.watching.set()
        self.log("Watching %s" % folder)

    def stop_watching(self):
        self.watching.clear()
        self.log("Stopped watching.")

    def submit_file(self, path):
        self.jobs.put(path)

    # ---- internals ----
    def _is_img(self, name):
        low = name.lower()
        return low.endswith(self.image_exts) and not low.endswith("_rings.png")

    def _scan_once(self):
        try:
            names = [n for n in os.listdir(self.folder) if self._is_img(n)]
        except OSError:
            return
        for n in sorted(names):
            if n in self._done:
                continue
            path = os.path.join(self.folder, n)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if self._sizes.get(n) != size:     # still being written
                self._sizes[n] = size
                continue
            self._process(path)
            self._done.add(n)
            self._sizes.pop(n, None)

    def _process(self, path):
        name = os.path.basename(path)
        try:
            rf.detect_rings(path, self._H)
            jp = os.path.splitext(path)[0] + "_rings.json"
            data = json.load(open(jp)) if os.path.exists(jp) else {"rings": []}
            png = os.path.splitext(path)[0] + "_rings.png"
            self.q.put(("result", {"name": name, "png": png,
                                   "rings": data.get("rings", [])}))
            self.log("%s -> %d ring(s)" % (name, len(data.get("rings", []))))
        except Exception as e:
            self.log("ERROR on %s: %s" % (name, e))
