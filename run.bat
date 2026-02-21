@echo off
setlocal

:: Navigate to project directory
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist .venv (
    echo [ERROR] Virtual environment not found. Please run scripts\train.py first.
    pause
    exit /b 1
)

:: Activate virtual environment and run Streamlit
echo "[INFO] Launching Customer Segmentation & Basket Intelligence System..."
.venv\Scripts\streamlit run ui\app.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Application failed to start.
    pause
)

endlocal
