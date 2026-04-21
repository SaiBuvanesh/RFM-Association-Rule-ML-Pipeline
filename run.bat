@echo off
setlocal

:: Navigate to project directory
cd /d "%~dp0"

:: Check if virtual environment exists
if not exist .venv (
    echo [INFO] Virtual environment not found. Creating one now...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python is installed.
        pause
        exit /b 1
    )
    echo [INFO] Activating virtual environment and installing dependencies...
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: Check if artifacts exist
if not exist artifacts\rfm_segments.csv (
    echo [INFO] Artifacts not found. Running training script...
    call .venv\Scripts\activate.bat
    python scripts\train.py
    if errorlevel 1 (
        echo [ERROR] Training script failed.
        pause
        exit /b 1
    )
)

:: Activate virtual environment and run Streamlit
echo [INFO] Launching Customer Segmentation & Basket Intelligence System...
call .venv\Scripts\activate.bat
streamlit run ui\app.py

if %ERRORLEVEL% neq 0 (
    echo [ERROR] Application failed to start.
    pause
)

endlocal
