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


def do_verification_flow(args, port):

    # Setup
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
    prompt = "What is the capital of Bulgaria?"
    params = {
        "temperature": 0,
        "seed": args.seed,
    }

    # Initial request (send by user)
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": prompt},
        ],
        **params,
        extra_body={"return_verification_proofs": True},
    )
    response_dump = response.model_dump()
    response_content = response_dump["choices"][0]["message"]["content"]
    fingerprint = response_dump["choices"][0]["message"][
        "toploc_verification_fingerprints"
    ][-1]
    print("Prompt: ", prompt)
    print("Response: ", response_content)
    print("Fingerprint: ", fingerprint)
    input("Press Enter to continue...")

    # Send verification request to verification instance
    prefill_response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response_content},
        ],
        max_tokens=0,
        **params,
        extra_body={
            "toploc_verification_fingerprint_to_validate": fingerprint,
        },
    )
    prefill_dump = prefill_response.model_dump()
    verification_result = prefill_dump["choices"][0]["message"][
        "toploc_verification_fingerprint_validation_result"
    ]
    error_statistics = json.loads(verification_result)
    print("Verification Result", verification_result)

    # Apply error thresholds
    verified = is_verified(**error_statistics)
    print("Verified:", verified)
    input("Press Enter to exit...")


def parse_args():
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
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="Disable CUDA graph"
    )
    args = parser.parse_args()
    return args


# Kill any GPU processes that might be running
def kill_gpu_processes():
    import subprocess

    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
    pids = [line.decode().strip() for line in result.stdout.splitlines()]
    for pid in pids:
        print(f"Killing process {pid}")
        os.kill(int(pid), signal.SIGKILL)


# Is it verified, based on activation error statistics?
def is_verified(exp_mismatches, mant_err_mean, mant_err_median):
    """
    Determine if a verification proof is valid based on metrics.
    Adjust thresholds as needed.
    """
    # Example thresholds - adjust based on your actual requirements
    return exp_mismatches <= 90 and mant_err_mean <= 10.0 and mant_err_median <= 8.0


# Fire up the server
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

    if args.disable_cuda_graph:
        MAYBE_DISABLE_CUDA_GRAPH = "--disable-cuda-graph"
    else:
        MAYBE_DISABLE_CUDA_GRAPH = ""

    server_process, port = launch_server_cmd(
        f"""
        python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --toploc-verification {MAYBE_NOISY} {MAYBE_DISABLE_CUDA_GRAPH}
        """
    )

    # Wait for the server to start
    wait_for_server(f"http://localhost:{port}")

    # Add additional delay to ensure server is fully initialized
    print("Waiting 3 more seconds for server to be fully initialized...")
    time.sleep(3)

    return server_process, port


if __name__ == "__main__":
    kill_gpu_processes()
    args = parse_args()
    server_process, port = start_server(args)
    do_verification_flow(args, port)
    server_process.terminate()
