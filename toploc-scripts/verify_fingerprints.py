import argparse
import json
import os
import signal
import subprocess
import sys
import time

import openai
from dotenv import load_dotenv
from tqdm import tqdm

from sglang.utils import (
    launch_server_cmd,
    print_highlight,
    terminate_process,
    wait_for_server,
)

load_dotenv()

if not os.getenv("HF_TOKEN"):
    print_highlight("HF_TOKEN not found in environment variables!", color="red")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, required=True, help="Machine name")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="JSON filename containing fingerprints to analyze (from the fingerprints directory)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=False,
        default=None,
        help="Filename to write the results (goes to verifications directory)",
    )
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="Disable CUDA graph"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of fingerprints to analyze",
    )
    return parser.parse_args()


def kill_gpu_processes():
    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
    pids = [line.decode().strip() for line in result.stdout.splitlines()]
    for pid in pids:
        print(f"Killing process {pid}")
        os.kill(int(pid), signal.SIGKILL)


def start_server(args):
    """
    Start the SGL server with TopLoc fingerprint verification enabled.
    """
    print("Starting server with TopLoc fingerprint verification enabled...")
    if args.quiet:
        MAYBE_NOISY = ""
    else:
        MAYBE_NOISY = "--log-level debug"

    if args.disable_cuda_graph:
        MAYBE_DISABLE_CUDA_GRAPH = "--disable-cuda-graph"
    else:
        MAYBE_DISABLE_CUDA_GRAPH = ""

    print(f"Starting server with model {args.model}...")

    server_process, port = launch_server_cmd(
        f"""
        python -m sglang.launch_server --model-path {args.model} --host 0.0.0.0 --toploc-verification {MAYBE_NOISY} {MAYBE_DISABLE_CUDA_GRAPH}
        """
    )

    print(f"Starting on port {port}...")

    # Wait for the server to start
    wait_for_server(f"http://localhost:{port}")

    # Add additional delay to ensure server is fully initialized
    print("Waiting 3 more seconds for server to be fully initialized...")
    time.sleep(3)

    return server_process, port


def load_fingerprints(args):
    """
    Load fingerprints from the input file.
    """
    input_filepath = args.input_file
    if not os.path.isabs(input_filepath):
        input_filepath = os.path.join(SCRIPT_DIR, "fingerprints", input_filepath)

    print(f"Loading fingerprints from {input_filepath}")
    with open(input_filepath, "r") as f:
        fingerprints = json.load(f)

    if args.limit is not None and args.limit > 0:
        fingerprints = fingerprints[: args.limit]
        print(f"Limited to first {args.limit} fingerprints")

    return fingerprints


def verify_fingerprints(args, port, fingerprints):
    """
    Verify the fingerprints by making verification requests to the server.
    """
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
    verification_results = []

    for i, item in enumerate(tqdm(fingerprints)):
        prompt = item["prompt"]
        model = item["model"]
        fingerprint = item["fingerprint"]
        original_response = item["complete_response"]["choices"][0]["message"][
            "content"
        ]

        # Create a verification request
        request = dict(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": original_response},
            ],
            max_tokens=0,  # This is a prefill-only operation
            extra_body={"toploc_verification_fingerprint_to_validate": fingerprint},
        )

        try:
            response = client.chat.completions.create(**request)
            response_dump = response.model_dump()

            # Extract verification result
            verification_result = {
                "prompt": prompt,
                "verification_request": request,
                "verification_response": response_dump,
                "original_request": item["complete_request"],
                "original_response": item["complete_response"],
                "original_machine": item["machine"],
                "original_model": item["model"],
                "verification_machine": args.machine,
                "verification_model": args.model,
                "original_fingerprint": fingerprint,
                "verification_result": response_dump["choices"][0]["message"].get(
                    "toploc_verification_fingerprint_validation_result", False
                ),
            }

            verification_results.append(verification_result)
        except Exception as e:
            print(f"Error verifying fingerprint {i}: {e}")

    return verification_results


def write_to_file(args, verification_results):
    """
    Write verification results to the output file.
    """
    if args.output_file is None:
        args.output_file = args.model.replace("/", "_") + "_for_" + args.input_file
    verifications_dir = os.path.join(SCRIPT_DIR, "verifications")
    output_filepath = os.path.join(verifications_dir, args.output_file)
    os.makedirs(verifications_dir, exist_ok=True)
    with open(output_filepath, "w") as f:
        json.dump(verification_results, f, indent=4)
    print(f"Verification results written to {output_filepath}")


def main():
    args = parse_args()
    fingerprints = load_fingerprints(args)
    print(f"Loaded {len(fingerprints)} fingerprints, preparing to verify...")

    kill_gpu_processes()
    server_process, port = start_server(args)

    try:
        verification_results = verify_fingerprints(args, port, fingerprints)
        write_to_file(args, verification_results)

        # Print a summary of verification results
        print(f"\nVerification summary:")
        print(f"Total fingerprints: {len(verification_results)}")
    finally:
        print("Terminating server...")
        terminate_process(server_process)
        print("Server terminated.")


if __name__ == "__main__":
    main()
