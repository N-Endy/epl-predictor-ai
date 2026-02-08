@echo off
echo Setting up Python dependencies for PredictorBlazor...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed. Please install Python 3.8+ and try again.
    exit /b 1
)

REM Check if pip is installed
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip is not installed. Please install pip and try again.
    exit /b 1
)

REM Install dependencies
echo Installing Python packages...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo Setup complete! Python dependencies are installed.
) else (
    echo Failed to install dependencies. Please check your Python/pip installation.
    exit /b 1
)
