import argparse
import csv
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from sglang.utils import print_highlight, terminate_process, wait_for_server

DISABLE_CUDA_GRAPH = True
MAYBE_DISABLE_CUDA_GRAPH = "--disable-cuda-graph" if DISABLE_CUDA_GRAPH else ""

# Load .env file in the same directory as this script
script_dir = Path(__file__).parent.absolute()
env_path = script_dir / ".env"


# Kill any GPU processes that might be running
def kill_gpu_processes():
    import subprocess

    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
    pids = [line.decode().strip() for line in result.stdout.splitlines()]
    for pid in pids:
        print(f"Killing process {pid}")
        os.kill(int(pid), signal.SIGKILL)


def is_verified(exp_mismatches, mant_err_mean, mant_err_median):
    """
    Determine if a verification proof is valid based on metrics.
    Adjust thresholds as needed.
    """
    # Example thresholds - adjust based on your actual requirements
    return exp_mismatches <= 90 and mant_err_mean <= 10.0 and mant_err_median <= 8.0


def load_ultrachat_samples(dataset_path, seed, num_samples=10):
    """
    Load samples from the UltraChat dataset.
    Specifically reads the first n lines from train_0.jsonl in a streaming fashion.
    Each line in the jsonl file contains a JSON array, and we use the first item as the prompt.

    Args:
        dataset_path: Path to the UltraChat dataset
        seed: Random seed (not used in this implementation)
        num_samples: Number of samples to extract

    Returns:
        List of samples with ID and prompt
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"UltraChat dataset not found at {dataset_path}")

    # Specifically use train_0.jsonl
    file_path = dataset_path / "train_0.jsonl"
    if not file_path.exists():
        raise FileNotFoundError(f"train_0.jsonl not found at {dataset_path}")

    print(f"Reading from file: {file_path}")

    samples = []
    try:
        # Process the file in a streaming fashion (line by line)
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break

                try:
                    # Each line is a JSON array
                    data = json.loads(line.strip())
                    ultrachat_id = data["id"]
                    ultrchat_data = data["data"]
                    if isinstance(ultrchat_data, list) and len(ultrchat_data) > 0:
                        # Use the first item in the array as the prompt
                        prompt = ultrchat_data[0]
                        # Assign a unique ID using the line number
                        samples.append(
                            {
                                "ultrachat_filepath": file_path,
                                "id": ultrachat_id,
                                "data": prompt,
                            }
                        )
                    else:
                        print(
                            f"Error: Line {i} does not contain a valid JSON array - {line}"
                        )
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line {i} in file: {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")

    if not samples:
        print(f"Warning: No valid samples found in {file_path}")

    return samples


def start_server(args):
    """
    Start the SGL server with TopLoc fingerprint verification enabled.
    """
    from sglang.utils import launch_server_cmd

    print("Starting server with TopLoc fingerprint verification enabled...")
    if args.quiet:
        MAYBE_NOISY = ""
    else:
        MAYBE_NOISY = "--log-level debug"

    server_process, port = launch_server_cmd(
        f"""
        python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --toploc-fingerprint {MAYBE_NOISY} {MAYBE_DISABLE_CUDA_GRAPH}
        """
    )

    # Wait for the server to start
    wait_for_server(f"http://localhost:{port}")

    # Add additional delay to ensure server is fully initialized
    print("Waiting 3 more seconds for server to be fully initialized...")
    time.sleep(3)

    return server_process, port


def process_samples(args, samples, port, seed, output_json):
    """
    Process UltraChat samples, generate responses, and verify proofs.
    """
    output_json = script_dir / output_json
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    # Create a list to store all results
    results = []

    for sample in tqdm(samples, desc="Processing samples"):
        sample_id = sample["id"]

        # With our updated load_ultrachat_samples, data now directly contains the prompt string
        prompt = sample["data"]
        if not isinstance(prompt, str):
            print(f"Unexpected data format in sample {sample_id}: {prompt}")
            # Add error record to results
            results.append(
                {
                    "ultrachat_filepath": sample["ultrachat_filepath"],
                    "ultrachat_id": sample_id,
                    "prompt": "ERROR",
                    "response": "ERROR",
                    "last_token_proof": "???",
                    "exp_mismatches": "ERROR",
                    "mant_err_mean": "ERROR",
                    "mant_err_median": "ERROR",
                    "verified": False,
                }
            )
            continue

        params = {
            "temperature": 0,
            "seed": seed,
        }

        try:
            # Generate a response
            print(f"Generating response for sample {sample_id}...")
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                **params,
                extra_body={"return_verification_proofs": True},
            )

            # Get the response and last token proof
            response_dump = response.model_dump()

            print("Response received:")
            print(json.dumps(response_dump, indent=4))

            if args.interactive:
                input("Press Enter to continue...")

            original_content = response_dump["choices"][0]["message"]["content"]
            last_token_proof = response_dump["choices"][0]["message"][
                "verification_proofs"
            ][-1]

            # Create verification request
            prefill_response = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": original_content},
                ],
                max_tokens=0,
                **params,
                extra_body={
                    "verification_proof_to_validate": last_token_proof,
                },
            )

            prefill_dump = prefill_response.model_dump()

            print("Prefill response received:")
            print(json.dumps(prefill_dump, indent=4))

            if args.interactive:
                input("Press Enter to continue...")

            # Parse verification result
            validation_result = json.loads(
                prefill_dump["choices"][0]["message"][
                    "verification_proof_validation_result"
                ]
            )

            exp_mismatches = validation_result["exp_mismatches"]
            mant_err_mean = validation_result["mant_err_mean"]
            mant_err_median = validation_result["mant_err_median"]
            verified = is_verified(exp_mismatches, mant_err_mean, mant_err_median)

            # Add the result to our list
            results.append(
                {
                    "ultrachat_filepath": str(sample["ultrachat_filepath"]),
                    "ultrachat_id": sample_id,
                    "prompt": prompt,
                    "response": original_content,
                    "last_token_proof": last_token_proof,
                    "exp_mismatches": exp_mismatches,
                    "mant_err_mean": mant_err_mean,
                    "mant_err_median": mant_err_median,
                    "verified": verified,
                }
            )

            print(
                f"Processed sample {sample_id}: {'Verified' if verified else 'Not Verified'}"
            )

        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            # Add error record to results
            results.append(
                {
                    "ultrachat_filepath": str(sample["ultrachat_filepath"]),
                    "ultrachat_id": sample_id,
                    "prompt": prompt if isinstance(prompt, str) else "ERROR",
                    "response": "ERROR",
                    "last_token_proof": "???",
                    "exp_mismatches": "ERROR",
                    "mant_err_mean": "ERROR",
                    "mant_err_median": "ERROR",
                    "verified": False,
                }
            )

    # Save results to JSON file
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Test TopLoc fingerprint verification with UltraChat dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for sampling and generation",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to process"
    )
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ultrachat_verification_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    # Load environment variables
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        raise Exception(".env file not found")

    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise Exception("HF_TOKEN environment variable not set")

    # Kill any existing GPU processes
    kill_gpu_processes()

    # Ensure UltraChat dataset is downloaded
    dataset_path = Path(script_dir / "ultrachat")

    # Load UltraChat samples
    samples = load_ultrachat_samples(dataset_path, args.seed, args.num_samples)

    # Start server
    server_process, port = start_server(args)

    try:
        # Process samples
        process_samples(args, samples, port, args.seed, args.output)
    finally:
        # Terminate server
        print("Terminating server...")
        terminate_process(server_process)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
