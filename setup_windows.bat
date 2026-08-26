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

where python >nul 2>&1
if errorlevel 1 (
  where py >nul 2>&1
  if errorlevel 1 (
    echo Python 3 was not found in PATH.
    goto :fail
  )
  set "BOOTSTRAP_PY=py -3"
) else (
  set "BOOTSTRAP_PY=python"
)

if not exist ".venv\Scripts\python.exe" (
  %BOOTSTRAP_PY% -m venv .venv
  if errorlevel 1 goto :fail
)
call .venv\Scripts\activate.bat
if errorlevel 1 goto :fail
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  echo Python 3.11 or newer is required.
  goto :fail
)
python -m pip install --upgrade pip
if errorlevel 1 goto :fail
python -m pip install -r requirements.txt
if errorlevel 1 goto :fail

python examples\generate_demo_data.py
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
