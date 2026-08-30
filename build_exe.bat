@echo off
REM Build the Windows .exe for ring_app.py
REM Run this on Windows from the project root (needs Python + the deps installed).

echo === installing build + runtime dependencies ===
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller

echo === cleaning old build/dist ===
rmdir /s /q build 2>nul
rmdir /s /q dist 2>nul

echo === building (this takes a few minutes; torch is large) ===
pyinstaller --noconfirm --clean ring_app.spec

echo.
echo === done ===
echo RUN THIS ONE:  dist\ring_app\ring_app.exe
echo (do NOT run the copy under build\ - that is only scratch)
echo Copy the whole dist\ring_app folder to the target machine, keeping
echo ring_app.exe next to its _internal folder.
echo If you see a python3xx.dll error, install the Microsoft Visual C++
echo Redistributable x64:  https://aka.ms/vs/17/release/vc_redist.x64.exe
pause
