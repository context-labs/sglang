#!/bin/bash

# Exit on any error
set -e

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install required packages if not already installed
echo "Checking/installing required packages..."
pip install requests

# Run the test
echo "Running proxy server test..."
python3 test-proxy.py

# Deactivate virtual environment
deactivate 