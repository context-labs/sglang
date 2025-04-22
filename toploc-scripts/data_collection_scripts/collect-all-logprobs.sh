#!/bin/bash

source ../sglang-current/.venv/bin/activate


MACHINE=$1

if [ -z "$MACHINE" ]; then
    echo "Error: Machine name is empty"
    exit 1
fi

if [ ! -d "toploc-scripts/logprobs" ]; then
    mkdir -p toploc-scripts/logprobs
fi

MODELS=("meta-llama/Llama-3.1-8B-Instruct" "context-labs/neuralmagic-llama-3.1-8b-instruct-FP8" "meta-llama/Llama-3.2-3B-Instruct")

for inference_filepath in toploc-scripts/inferences_to_replicate/*.inference; do

    filename=$(basename "$inference_filepath")
    filename_no_ext=${filename%.inference}

    for MODEL in "${MODELS[@]}"; do
        # Sanitize model name for filename
        SANITIZED_MODEL=$(echo "$MODEL" | tr '/' '_' | tr ' ' '_')

        OUTPUT_FILENAME="logprobs_${MACHINE}_${SANITIZED_MODEL}_for_${filename_no_ext}.logprob"

        if [ -f "toploc-scripts/logprobs/${OUTPUT_FILENAME}" ]; then
            echo "Output file already exists: ${OUTPUT_FILENAME}"
            continue
        fi

        echo "Processing model: $MODEL"

        cmd="python toploc-scripts/data_collection_scripts/collect_logprobs.py --N 1 --machine \"$MACHINE\" --model \"$MODEL\" --input-file \"$filename\" --output-file \"$OUTPUT_FILENAME\" --disable-cuda-graph --debugging"
        # Add command to array for later printing
        commands+=("$cmd")


        echo "Running: $cmd"
        eval "$cmd"

        # Optional: add a small delay between runs
        sleep 2

        break

    done

    break

done

# Print all commands
for cmd in "${commands[@]}"; do
    echo "$cmd"
done
