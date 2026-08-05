@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Setting up the local Motor Imbalance Detector...
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        py -3 -m venv .venv
    ) else (
        python -m venv .venv
    )
    if errorlevel 1 goto setup_error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto setup_error
)

echo Opening the Motor Imbalance Detector...
".venv\Scripts\python.exe" app.py
goto end

:setup_error
echo.
echo Setup could not be completed. Confirm that Python and internet access are available.
pause

:end
endlocal
