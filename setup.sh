#!/bin/bash

echo "Setting up Python dependencies for PredictorBlazor..."

# Check if Python is installed
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Python is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies in virtual environment
echo "Installing Python packages in virtual environment..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "Setup complete! Virtual environment is ready and dependencies are installed."
else
    echo "Failed to install dependencies. Please check your Python/pip installation."
    exit 1
fi
