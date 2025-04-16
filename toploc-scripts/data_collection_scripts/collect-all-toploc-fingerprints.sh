#!/bin/bash

# Check if machine name is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <machine_name>"
    exit 1
fi

source .venv/bin/activate

MACHINE=$1

# Array of models to process
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-8B-Instruct;fp8" "meta-llama/Llama-3.2-3B-Instruct")

for MODEL in "${MODELS[@]}"; do
    # Sanitize model name for filename
    SANITIZED_MODEL=$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')

    echo "Processing model: $MODEL"
    python toploc-scripts/data_collection_scripts/collect_toploc_fingerprints.py --N 100 --machine "$MACHINE" --model "$MODEL" --output_filename "train0_${MACHINE}_${SANITIZED_MODEL}.fingerprint" --disable-cuda-graph

    # Optional: add a small delay between runs
    sleep 2
done

echo "All models processed successfully!"
