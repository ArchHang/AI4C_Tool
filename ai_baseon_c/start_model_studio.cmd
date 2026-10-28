@echo off
setlocal
set "STUDIO_PYTHON=%MODEL_STUDIO_PYTHON%"
if defined STUDIO_PYTHON goto launch
if exist "%~dp0..\.venv\Scripts\pythonw.exe" set "STUDIO_PYTHON=%~dp0..\.venv\Scripts\pythonw.exe"
if defined STUDIO_PYTHON goto launch
if exist "%~dp0..\.venv\Scripts\python.exe" set "STUDIO_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if defined STUDIO_PYTHON goto launch
set "STUDIO_PYTHON=python"
:launch
"%STUDIO_PYTHON%" -s -B -X utf8 "%~dp0tools\model_studio_gui.py" %*
exit /b %errorlevel%
