@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call .venv\Scripts\activate.bat
python .agents\skills\ai-data-expert\scripts\solve_notebook.py ^
  --input examples\DNN_regression_question.ipynb ^
  --data examples\4_manufacturing_yield.csv ^
  --output outputs\DNN_regression_answer.ipynb
endlocal
