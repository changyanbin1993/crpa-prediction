#!/bin/bash
# Quick Start Script for CRPA Prediction Model

echo "=========================================="
echo "CRPA Prediction Model - Quick Start"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if data file exists
echo ""
if [ -f "CRPA.csv" ]; then
    echo "✓ Data file found: CRPA.csv"
    echo ""
    echo "Ready to run! Execute the following commands:"
    echo ""
    echo "  python 01_data_preprocessing.py"
    echo "  python 02_feature_selection.py"
    echo "  python 03_model_training.py"
    echo "  python 04_model_interpretation.py"
else
    echo "⚠ Warning: CRPA.csv not found"
    echo "Please place your data file in the current directory"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
