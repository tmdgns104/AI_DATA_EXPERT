@echo off
setlocal EnableExtensions
cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Running setup first...
  call setup_windows.bat
  if errorlevel 1 exit /b 1
)

call .venv\Scripts\activate.bat
if errorlevel 1 exit /b 1
python examples\generate_demo_data.py
if errorlevel 1 exit /b 1
python .agents\skills\ai-data-expert\scripts\solve_notebook.py ^
  --input examples\DNN_regression_question.ipynb ^
  --data examples\4_manufacturing_yield.csv ^
  --output outputs\DNN_regression_answer.ipynb
if errorlevel 1 exit /b 1

echo.
echo Demo complete: outputs\DNN_regression_answer.ipynb
exit /b 0
