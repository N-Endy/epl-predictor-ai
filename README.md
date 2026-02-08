# PredictorBlazor

A Blazor Server app for predicting English Premier League football match outcomes using Elo ratings, rolling statistics, and Random Forest machine learning.

## Setup

1. Ensure Python 3.8+ is installed on your system.

2. Run the setup script to install Python dependencies:
   - On Linux/Mac: `./setup.sh`
   - On Windows: `setup.bat`
   - Or manually: `pip install -r requirements.txt`

3. Build and run the .NET project:
   ```bash
   dotnet build
   dotnet run
   ```

The app will load `wwwroot/epl.csv` and generate predictions for the next round of matches, with optional backtesting.

## Python Integration

The prediction logic is implemented in `predict.py` using pandas, numpy, and scikit-learn. The Blazor backend calls this script via subprocess to leverage your Python models seamlessly.
