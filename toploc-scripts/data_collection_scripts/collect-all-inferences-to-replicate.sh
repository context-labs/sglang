#!/bin/bash

# Check if machine name is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <machine_name>"
    exit 1
fi

source ../sglang-current/.venv/bin/activate
pip install dotenv
pip install huggingface-hub
pip install tabulate

MACHINE=$1

if [ -z "$MACHINE" ]; then
    echo "Error: Machine name is empty"
    exit 1
fi


# Array of models to process
# "meta-llama/Llama-3.1-8B-Instruct;fp8"
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "context-labs/neuralmagic-llama-3.1-8b-instruct-FP8" "meta-llama/Llama-3.2-3B-Instruct")

if [ ! -d "toploc-scripts/inferences_to_replicate" ]; then
    mkdir -p toploc-scripts/inferences_to_replicate
fi

for MODEL in "${MODELS[@]}"; do
    # Sanitize model name for filename
    SANITIZED_MODEL=$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')

    echo "Processing model: $MODEL"
    OUTPUT_FILENAME="train0_${MACHINE}_${SANITIZED_MODEL}.inference"
    if [ -f "toploc-scripts/inferences_to_replicate/${OUTPUT_FILENAME}" ]; then
        echo "Output file already exists: ${OUTPUT_FILENAME}"
        continue
    fi
    python toploc-scripts/data_collection_scripts/collect_inferences_to_replicate.py --N 100 --machine "$MACHINE" --model "$MODEL" --output_filename "${OUTPUT_FILENAME}" --disable-cuda-graph

    # Optional: add a small delay between runs
    sleep 2
done

echo "All models processed successfully!"
