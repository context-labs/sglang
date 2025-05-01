#!/bin/bash

# Exit on error
set -e

echo "Starting deployment process..."

# Define source and destination paths
SOURCE_DIR="$(pwd)"
TARGET_DIR="$(pwd)/../kuzco-custom-worker/apps/cli"
PROXY_SERVERS_DIR="${TARGET_DIR}/src/lib/proxy-servers"

# Ensure target directory exists
mkdir -p "${PROXY_SERVERS_DIR}"

# Copy sglang-launch-proxy-server.sh to target location
echo "Copying sglang-launch-proxy-server.sh to ${PROXY_SERVERS_DIR}/sglang-launch-proxy-server.sh"
cp "${SOURCE_DIR}/sglang-launch-proxy-server.sh" "${PROXY_SERVERS_DIR}/sglang-launch-proxy-server.sh"

# Copy sglang-proxy-server.py to target location
echo "Copying sglang-proxy-server.py to ${PROXY_SERVERS_DIR}/sglang-proxy-server.py"
cp "${SOURCE_DIR}/sglang-proxy-server.py" "${PROXY_SERVERS_DIR}/sglang-proxy-server.py"

# Verify files were copied successfully
if [ -f "${PROXY_SERVERS_DIR}/sglang-launch-proxy-server.sh" ] && [ -f "${PROXY_SERVERS_DIR}/sglang-proxy-server.py" ]; then
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