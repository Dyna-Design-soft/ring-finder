# PyInstaller spec for ring_app.py  ->  build a Windows .exe
# ------------------------------------------------------------
# Build:  pyinstaller ring_app.spec        (run on Windows)
# Output: dist/ring_app/ring_app.exe  (a folder you can copy to the machine)
#
# Notes
#  * ultralytics + torch are large; this collects their data files, submodules
#    and package metadata so FastSAM loads inside the frozen app.
#  * config/ and data/ live NEXT TO the exe at runtime (ring_app.py switches
#    ROOT to the exe folder when frozen), so settings and results persist and
#    are editable without touching the bundle.
#  * The FastSAM weight is NOT bundled (it is large / git-ignored). On first
#    run the app downloads it, or place FastSAM-x.pt next to the exe.

from PyInstaller.utils.hooks import collect_all, copy_metadata

datas = []
binaries = []
hiddenimports = []

# ship the fitted homography so the exe has a default map
datas += [("config/robot_map.json", "config")]

# collect the heavy ML packages (data, binaries, submodules)
for pkg in ("ultralytics", "torch", "torchvision"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# package metadata some libraries look up at runtime
for pkg in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python",
            "pillow", "tqdm", "pyyaml", "psutil"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

a = Analysis(
    ["scripts/ring_app.py"],
    pathex=["scripts"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "pandas"],   # not needed by the app; trims size
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ring_app",
    console=False,          # set True to see a console with errors while debugging
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ring_app",
)
