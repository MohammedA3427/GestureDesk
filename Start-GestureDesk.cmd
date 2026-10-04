@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run Setup-GestureDesk.cmd first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m gesturedesk run --backend dshow %*
if errorlevel 1 pause
