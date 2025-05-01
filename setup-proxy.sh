#!/bin/bash

# Exit on error
set -e

echo "Setting up SGLang proxy server environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3 first."
    exit 1
fi

# Create and activate virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install basic requirements
echo "Upgrading pip and installing basic requirements..."
pip install --upgrade pip setuptools wheel html5lib six

# Install sglang with runtime dependencies
echo "Installing sglang with runtime dependencies..."
# First install torch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install sglang in editable mode with runtime dependencies
pip install -e "python[runtime_common]"

# Install proxy server requirements
echo "Installing proxy server requirements..."
pip install -r requirements-proxy.txt

echo "Setup complete! You can now run the proxy server with:"
echo "python proxy_server.py" 