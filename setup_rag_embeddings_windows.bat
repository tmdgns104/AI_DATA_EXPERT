@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
if not exist ".venv\Scripts\python.exe" (
  echo Run setup_windows.bat first.
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install -r requirements-rag-optional.txt
if errorlevel 1 (
  echo Optional embedding setup FAILED. V4 will continue with the offline hybrid vector fallback.
  exit /b 1
)
echo Optional semantic embedding + FAISS support installed.
exit /b 0
