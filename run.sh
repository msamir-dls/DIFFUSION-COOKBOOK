#!/bin/bash

# 1. Exit on error
set -e

echo "--- Starting MNIST Diffusion Pipeline ---"

# 2. Setup Virtual Environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
fi

# 3. Activate Environment
source venv/bin/activate

# 4. Install Dependencies
echo "[*] Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Run the Master Python Pipeline
python run_all.py

echo "--- Pipeline Complete ---"
echo "To view your results, run: mlflow ui"