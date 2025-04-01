#!/bin/bash
# Script to run the entire TopLoc verification pipeline
# Usage: ./run_verification_pipeline.sh [model_path]
# Example: ./run_verification_pipeline.sh meta-llama/Llama-3.1-70B-Instruct

set -e # Exit immediately if a command exits with a non-zero status.

if [ -d .sglang ]; then
    source .sglang/bin/activate
else
    echo "Error: .sglang directory not found. Please run setup.sh first, or run from root of repo if you aren't doing that already."
    exit 1
fi

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if the model path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [model_path]"
    echo "Example: $0 meta-llama/Llama-3.1-70B-Instruct"
    exit 1
fi

echo "===download ultrachat==="
python "$SCRIPT_DIR/download_ultrachat.py"

SPOOF_MODEL_PATH="$1"
ORIGINAL_JSON="ultrachat_verification_results.json"
SPOOF_JSON="ultrachat_spoof_verification_results.json"
COMPARISON_CSV="verification_comparison.csv"

echo "===== Starting TopLoc verification pipeline ====="
echo "Original model: meta-llama/Llama-3.1-8B-Instruct (hardcoded in test_ultrachat.py)"
echo "Spoof model: $SPOOF_MODEL_PATH"
echo ""

# Step 1: Run test_ultrachat.py
echo "===== Step 1: Running test_ultrachat.py ====="
echo "Generating verification proofs with the original model..."
python "$SCRIPT_DIR/test_ultrachat.py" --output "$ORIGINAL_JSON"
echo "Completed Step 1!"
echo ""

# Step 2: Run test_spoof_ultrachat.py
echo "===== Step 2: Running test_spoof_ultrachat.py ====="
echo "Testing verification proofs with the spoof model: $SPOOF_MODEL_PATH"
python "$SCRIPT_DIR/test_spoof_ultrachat.py" --model-path "$SPOOF_MODEL_PATH" --input-json "$ORIGINAL_JSON" --output-json "$SPOOF_JSON"
echo "Completed Step 2!"
echo ""

# Step 3: Run compare_verification_results.py
echo "===== Step 3: Running compare_verification_results.py ====="
echo "Comparing verification results between models..."
python "$SCRIPT_DIR/compare_verification_results.py" --original-json "$ORIGINAL_JSON" --spoof-json "$SPOOF_JSON" --output-csv "$COMPARISON_CSV"
echo "Completed Step 3!"
echo ""

echo "===== Pipeline completed successfully! ====="
echo "Results are available in: $COMPARISON_CSV"
