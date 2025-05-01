#!/bin/bash

# Exit on error
set -e

# Set current directory to this file's directory
cd "$(dirname "$0")"

# Build the command line arguments for launch_server
LAUNCH_SERVER_ARGS=()

# Parse arguments: map --port to PROXY_PORT and --sglang-internal-port to launch_server's --port
while [[ $# -gt 0 ]]; do
    case "$1" in
        --port=*)
            # Extract port value after '=' for the proxy
            PROXY_PORT="${1#*=}"
            shift
            ;;
        --port)
            PROXY_PORT="$2"
            shift 2
            ;;
        --sglang-internal-port=*)
            # Extract port value after '='
            SGLANG_INTERNAL_PORT="${1#*=}"
            # Add as --port to launch_server
            LAUNCH_SERVER_ARGS+=("--port" "$SGLANG_INTERNAL_PORT")
            shift
            ;;
        --sglang-internal-port)
            SGLANG_INTERNAL_PORT="$2"
            # Add as --port to launch_server
            LAUNCH_SERVER_ARGS+=("--port" "$SGLANG_INTERNAL_PORT")
            shift 2
            ;;
        *)
            # Forward all other arguments to launch_server
            LAUNCH_SERVER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Check if SGLANG_INTERNAL_PORT is defined
if [ -z "$SGLANG_INTERNAL_PORT" ]; then
    echo "Error: --sglang-internal-port is required"
    exit 1
fi

# Check if PROXY_PORT is defined
if [ -z "$PROXY_PORT" ]; then
    echo "Error: --port is required"
    exit 1
fi


# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start proxy server
echo "Starting proxy server..."
python sglang-proxy-server.py \
    --port $PROXY_PORT \
    --sglang-server "http://localhost:$SGLANG_INTERNAL_PORT" &
PROXY_PID=$!

# Start sglang server
echo "Starting sglang server..."
echo "Command: python -m sglang.launch_server ${LAUNCH_SERVER_ARGS[@]}"
python -m sglang.launch_server ${LAUNCH_SERVER_ARGS[@]} &
SGLANG_PID=$!

echo "SGLang server: http://localhost:$SGLANG_INTERNAL_PORT"
echo "Proxy server: http://localhost:$PROXY_PORT"

# Handle cleanup on script exit
trap 'echo "Shutting down servers..."; kill $SGLANG_PID $PROXY_PID' EXIT

# Keep script running and handle Ctrl+C
wait 