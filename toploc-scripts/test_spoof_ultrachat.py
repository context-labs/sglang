import argparse
import json
import os
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
    # Use the same thresholds as in test_ultrachat.py
    return exp_mismatches <= 90 and mant_err_mean <= 10.0 and mant_err_median <= 8.0


def start_server(args, model_path):
    """
    Start the SGL server with TopLoc fingerprint verification enabled
    using the specified model path.
    """
    from sglang.utils import launch_server_cmd

    if args.quiet:
        MAYBE_NOISY = ""
    else:
        MAYBE_NOISY = "--log-level debug"

    print(
        f"Starting server with TopLoc fingerprint verification enabled for model: {model_path}..."
    )
    server_process, port = launch_server_cmd(
        f"""
        python -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 --toploc-fingerprint {MAYBE_NOISY} {MAYBE_DISABLE_CUDA_GRAPH}
        """
    )

    # Wait for the server to start
    wait_for_server(f"http://localhost:{port}")

    # Add additional delay to ensure server is fully initialized
    print("Waiting 3 more seconds for server to be fully initialized...")
    time.sleep(3)

    return server_process, port


def process_verification_results(input_json, output_json, port, seed):
    """
    Process verification results from the input JSON, attempting to verify proofs
    with the current model.
    """
    output_json = script_dir / output_json

    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    # Read the input JSON file
    with open(input_json, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    # Create a list to store results
    results = []

    # Process each row in the input data
    for row in tqdm(input_data, desc="Processing verification proofs"):
        ultrachat_filepath = row["ultrachat_filepath"]
        ultrachat_id = row["ultrachat_id"]
        last_token_proof = row["last_token_proof"]
        prompt = row["prompt"]
        response = row["response"]

        # Skip rows with errors
        if last_token_proof == "???" or "ERROR" in str(row.values()):
            results.append(
                {
                    "ultrachat_filepath": str(ultrachat_filepath),
                    "ultrachat_id": ultrachat_id,
                    "prompt": prompt,
                    "response": response,
                    "last_token_proof": last_token_proof,
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
            # Create a verification request using the proof from the original JSON
            # Use the actual prompt and response from the input JSON
            verify_response = client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",  # This is just a placeholder, actual model is set in the server
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                max_tokens=0,
                **params,
                extra_body={
                    "verification_proof_to_validate": last_token_proof,
                },
            )

            verify_dump = verify_response.model_dump()

            # Parse verification result
            validation_result = json.loads(
                verify_dump["choices"][0]["message"][
                    "verification_proof_validation_result"
                ]
            )

            exp_mismatches = validation_result["exp_mismatches"]
            mant_err_mean = validation_result["mant_err_mean"]
            mant_err_median = validation_result["mant_err_median"]
            verified = is_verified(exp_mismatches, mant_err_mean, mant_err_median)

            # Add results to our list
            results.append(
                {
                    "ultrachat_filepath": str(ultrachat_filepath),
                    "ultrachat_id": ultrachat_id,
                    "prompt": prompt,
                    "response": response,
                    "last_token_proof": last_token_proof,
                    "exp_mismatches": exp_mismatches,
                    "mant_err_mean": mant_err_mean,
                    "mant_err_median": mant_err_median,
                    "verified": verified,
                }
            )

            print(
                f"Processed proof for {ultrachat_id}: {'Verified' if verified else 'Not Verified'}"
            )

        except Exception as e:
            print(f"Error processing sample {ultrachat_id}: {str(e)}")
            # Add error record to results
            results.append(
                {
                    "ultrachat_filepath": str(ultrachat_filepath),
                    "ultrachat_id": ultrachat_id,
                    "prompt": prompt,
                    "response": response,
                    "last_token_proof": last_token_proof,
                    "exp_mismatches": "ERROR",
                    "mant_err_mean": "ERROR",
                    "mant_err_median": "ERROR",
                    "verified": False,
                }
            )

    # Save results to JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Test TopLoc fingerprint verification spoofing with UltraChat dataset"
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="ultrachat_verification_results.json",
        help="Input JSON file containing verification results",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="ultrachat_spoof_verification_results.json",
        help="Output JSON file to store spoofing results",
    )
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model to test spoofing against",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for LLM generation"
    )
    args = parser.parse_args()

    # Load environment variables
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    args.input_json = script_dir / args.input_json
    args.output_json = script_dir / args.output_json

    # Make sure input JSON exists
    if not os.path.exists(args.input_json):
        print(
            f"Error: Input JSON file {args.input_json} not found. Run test_ultrachat.py first."
        )
        sys.exit(1)

    # Kill any existing GPU processes
    try:
        kill_gpu_processes()
    except Exception as e:
        print(f"Failed to kill GPU processes: {str(e)}")

    # Start the server with the specified model
    server_process, port = start_server(args, args.model_path)

    try:
        # Process the verification results
        process_verification_results(args.input_json, args.output_json, port, args.seed)
    finally:
        # Clean up the server process
        terminate_process(server_process)


if __name__ == "__main__":
    main()
