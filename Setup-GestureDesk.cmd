@echo off
cd /d "%~dp0"
python -m venv .venv
if errorlevel 1 goto fail
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto fail
.venv\Scripts\python.exe tools\download_model.py
if errorlevel 1 goto fail
echo Ready. Open Start-GestureDesk.cmd.
pause
exit /b 0
:fail
echo Setup failed. Check Python installation and network access.
pause
exit /b 1
