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

if [ ! -d "toploc-scripts/inferences_to_replicate" ]; then
    echo "Error: inferences_to_replicate directory not found"
    exit 1
fi

# Loop over all .inference files
for inference_filepath in toploc-scripts/inferences_to_replicate/*.inference; do
    if [ -f "$inference_filepath" ]; then
        echo "Performing replications for inference: $inference_filepath"
        inference_file=$(basename "$inference_filepath")
        output_file=${MACHINE}_for_${inference_file}.replication
        output_filepath="toploc-scripts/replications/$output_file"
        if [ -f "$output_filepath" ]; then
            echo "Output file already exists: $output_file"
            continue
        fi
        echo "Output file: $output_file"
        python toploc-scripts/perform_replications.py --input-file "$inference_file" --machine "$MACHINE" --output-file "$output_file" --disable-cuda-graph --quiet

        # Optional: Add a small delay between replications
        sleep 1
    else
        echo "File $inference_file does not exist"
    fi
done

echo "All inferences replicated!"
