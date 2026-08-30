@echo off
REM Build the Windows .exe for ring_app.py
REM Run this on Windows from the project root (needs Python + the deps installed).

echo === installing build + runtime dependencies ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

echo === building (this takes a few minutes; torch is large) ===
pyinstaller --noconfirm ring_app.spec

echo.
echo === done ===
echo Executable: dist\ring_app\ring_app.exe
echo Copy the whole dist\ring_app folder to the target machine.
echo On first run it creates config\ and data\ next to the exe.
pause
