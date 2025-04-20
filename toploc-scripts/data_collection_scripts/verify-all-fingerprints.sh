#!/bin/bash

# Check if machine name is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <machine_name>"
    exit 1
fi


source .venv/bin/activate

MACHINE=$1

if [ ! -d "toploc-scripts/fingerprints" ]; then
    echo "Error: fingerprints directory not found"
    exit 1
fi

# Array of models to process
MODELS=("meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-8B-Instruct;fp8" "meta-llama/Llama-3.2-3B-Instruct")

# Loop over all .fingerprint files
for fingerprint_filepath in toploc-scripts/fingerprints/*.fingerprint; do
    if [ -f "$fingerprint_filepath" ]; then
        for MODEL in "${MODELS[@]}"; do
            # Sanitize model name for filename
            SANITIZED_MODEL=$(echo "$MODEL" | tr '/' '_')
            echo "Verifying fingerprint: $fingerprint_filepath"
            fingerprint_file=$(basename "$fingerprint_filepath")
            output_file=${MACHINE}_${SANITIZED_MODEL}_for_${fingerprint_file}.verification
            echo "Output file: $output_file"
            python toploc-scripts/verify_fingerprints.py --input-file "$fingerprint_file" --machine "$MACHINE" --model "$MODEL" --output-file "$output_file" --disable-cuda-graph

            # Optional: Add a small delay between verifications
            sleep 1
        done
    else
        echo "File $fingerprint_file does not exist"
    fi
done

echo "All fingerprints verified!"
