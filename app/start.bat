@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ==============================================
echo   VAE Emulsion Property Prediction System V3.9.1
echo ==============================================
echo.

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

if exist "%VENV_PY%" goto CHECK_DEPS

echo [1/3] Looking for Python 3.12...
where py >nul 2>nul
if errorlevel 1 goto TRY_PYTHON
py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if errorlevel 1 goto TRY_PYTHON

echo [1/3] Creating virtual environment with Python 3.12...
py -3.12 -m venv ".venv"
if errorlevel 1 goto VENV_ERROR
goto INSTALL_DEPS

:TRY_PYTHON
where python >nul 2>nul
if errorlevel 1 goto PYTHON_ERROR
python -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
if errorlevel 1 goto PYTHON_ERROR

echo [1/3] Creating virtual environment with Python 3.12...
python -m venv ".venv"
if errorlevel 1 goto VENV_ERROR
goto INSTALL_DEPS

:CHECK_DEPS
echo [1/3] Existing virtual environment found. Checking dependencies...
"%VENV_PY%" -c "import fastapi,uvicorn,pandas,numpy,joblib,sklearn,catboost,xgboost,openpyxl,PIL,playwright" >nul 2>nul
if errorlevel 1 goto INSTALL_DEPS
goto RUN_APP

:INSTALL_DEPS
if not exist "%VENV_PY%" goto VENV_ERROR
echo [2/3] Installing/repairing Python packages. The first run may take several minutes...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto INSTALL_ERROR
"%VENV_PY%" -m pip install -r "requirements.txt"
if errorlevel 1 goto INSTALL_ERROR
goto RUN_APP

:RUN_APP
echo [2/3] Dependencies are ready.
echo [3/3] Starting server: http://127.0.0.1:8000
echo.
"%VENV_PY%" "run.py"
if errorlevel 1 goto RUN_ERROR
goto END

:PYTHON_ERROR
echo.
echo ERROR: Python 3.12 was not found.
echo Install Python 3.12 (64-bit), enable the Python launcher or PATH, then run start.bat again.
echo You can check with: py -3.12 --version
pause
exit /b 1

:VENV_ERROR
echo.
echo ERROR: Failed to create the .venv virtual environment.
echo Delete the .venv folder if it exists, then run start.bat again.
pause
exit /b 1

:INSTALL_ERROR
echo.
echo ERROR: Package installation failed.
echo This is a dependency installation error. Copy the messages above if troubleshooting is needed.
pause
exit /b 1

:RUN_ERROR
echo.
echo ERROR: The prediction server stopped with an error.
echo Copy the error messages above and send them for troubleshooting.
pause
exit /b 1

:END
endlocal
