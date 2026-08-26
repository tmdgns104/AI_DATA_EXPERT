@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set LOKY_MAX_CPU_COUNT=1
set JUPYTER_CONFIG_DIR=%CD%\.jupyter-config
set JUPYTER_DATA_DIR=%CD%\.jupyter-data
set JUPYTER_RUNTIME_DIR=%CD%\.jupyter-runtime
set IPYTHONDIR=%CD%\.ipython

if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto :fail
)
call .venv\Scripts\activate.bat
if errorlevel 1 goto :fail
python -m pip install --upgrade pip
if errorlevel 1 goto :fail
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

python verify_install.py
if errorlevel 1 goto :fail

echo.
where codex >nul 2>&1
if errorlevel 1 (
  echo Setup verified. Codex CLI was not found in PATH.
  echo You can still use the direct expert scripts, or install/configure Codex CLI separately.
) else (
  echo Setup verified. Start Codex in this folder with: codex
)
echo.
echo AI Data Expert V4 is ready.
exit /b 0

:fail
echo.
echo Setup FAILED. Check the error above. The project was not marked ready.
exit /b 1
