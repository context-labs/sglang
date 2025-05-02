#!/bin/bash

# Exit on error
set -e

echo "Starting deployment process..."

# Define source and destination paths
SOURCE_DIR="$(pwd)"
TARGET_DIR="$(pwd)/../kuzco-custom-worker/apps/cli"
PATCHES_DIR="${TARGET_DIR}/src/lib/patches"

# Ensure target directory exists
mkdir -p "${PATCHES_DIR}"

# Copy sglang-proxy-server.py to target location
echo "Copying sglang-patched-server.py to ${PATCHES_DIR}/sglang-patched-server.py"
cp "${SOURCE_DIR}/sglang-patched-server.py" "${PATCHES_DIR}/sglang-patched-server.py"

# Verify files were copied successfully
if [ -f "${PATCHES_DIR}/sglang-patched-server.py" ]; then
    echo "Files copied successfully!"
else
    echo "Error: Failed to copy files."
    exit 1
fi

# Change to the CLI directory
echo "Changing to ${TARGET_DIR} directory..."
cd "${TARGET_DIR}"

# Build the Docker image
echo "Building Docker image with task..."
task build-sglang-docker-image

echo "Deployment completed successfully!" 