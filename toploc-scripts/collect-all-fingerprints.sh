#!/bin/bash

# Check if machine name is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <machine_name>"
    exit 1
fi

source .venv/bin/activate

MACHINE=$1

# Array of models to process
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-70B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2")

for MODEL in "${MODELS[@]}"; do
    # Sanitize model name for filename
    SANITIZED_MODEL=$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')

    echo "Processing model: $MODEL"
    python collect_fingerprints.py --N 100 --model "$MODEL" --output_filename "train0_${MACHINE}_${SANITIZED_MODEL}.fingerprint"

    # Optional: add a small delay between runs
    sleep 2
done

echo "All models processed successfully!"
