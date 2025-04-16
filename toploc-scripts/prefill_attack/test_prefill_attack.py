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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)

ERROR_THRESHOLDS = {
    "exp_mismatches": 90,  # Maximum number of exponent mismatches allowed
    "mant_err_mean": 10,  # Maximum mean mantissa error allowed
    "mant_err_median": 8,  # Maximum median mantissa error allowed
}

load_dotenv()


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

    print(f"Starting server with model {args.model}...")

    model, *quantization = args.model.split(";")
    if quantization:
        quantization = quantization[0]
        print(f"Quantization: {quantization}")
        MAYBE_QUANTIZATION = f"--quantization {quantization}"
    else:
        MAYBE_QUANTIZATION = ""

    server_process, port = launch_server_cmd(
        f"""
        python -m sglang.launch_server --model-path {model} {MAYBE_QUANTIZATION} --host 0.0.0.0 --toploc-verification {MAYBE_NOISY} {MAYBE_DISABLE_CUDA_GRAPH}
        """
    )

    print(f"Starting on port {port}...")

    # Wait for the server to start
    wait_for_server(f"http://localhost:{port}")

    # Add additional delay to ensure server is fully initialized
    print("Waiting 3 more seconds for server to be fully initialized...")
    time.sleep(3)

    return server_process, port


def kill_gpu_processes():
    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
    pids = [line.decode().strip() for line in result.stdout.splitlines()]
    for pid in pids:
        print(f"Killing process {pid}")
        os.kill(int(pid), signal.SIGKILL)


def test_prefills(args, port):
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    ultrachat_file = os.path.join(ROOT_DIR, "ultrachat", args.ultrachat_file)

    # read ultrachat file in a loop
    with open(ultrachat_file, "r") as f:
        for i, line in enumerate(f):
            if i >= args.N:
                break
            # Load the prompt and the response from the line
            line = json.loads(line)
            prompt = line["data"][0]
            response = line["data"][1]

            spoofed_response = "A made up response"

            # Generate fingerprint using a prefill
            request = dict(
                model=args.model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": spoofed_response},
                ],
                max_tokens=0,
                temperature=args.temperature,
                seed=args.seed,
                extra_body={"return_verification_proofs": True},
            )

            response = client.chat.completions.create(**request)
            response_dump = response.model_dump()
            fingerprint = response_dump["choices"][0]["message"][
                "toploc_verification_fingerprints"
            ][-1]
            response_content = response_dump["choices"][0]["message"]["content"]

            print(f"Fingerprint for prompt {i}: {fingerprint}")

            # Verify fingerprint
            request = dict(
                model=args.model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": spoofed_response},
                ],
                max_tokens=0,
                temperature=args.temperature,
                seed=args.seed,
                extra_body={"toploc_verification_fingerprint_to_validate": fingerprint},
            )

            response = client.chat.completions.create(**request)
            response_dump = response.model_dump()
            verification_result = response_dump["choices"][0]["message"].get(
                "toploc_verification_fingerprint_validation_result", False
            )

            print(f"Verification fingerprint for prompt {i}: {verification_result}")

            verified = is_verified(json.loads(verification_result))

            print(f"Verified: {verified}")


def is_verified(verification_result):
    exp_check = (
        verification_result["exp_mismatches"] <= ERROR_THRESHOLDS["exp_mismatches"]
    )
    mean_check = (
        verification_result["mant_err_mean"] <= ERROR_THRESHOLDS["mant_err_mean"]
    )
    median_check = (
        verification_result["mant_err_median"] <= ERROR_THRESHOLDS["mant_err_median"]
    )
    passed = exp_check and mean_check and median_check
    return passed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=1,
        help="Number of prompts to test",
    )
    parser.add_argument(
        "--ultrachat_file", type=str, default="train_0.jsonl", help="ultrachat filename"
    )
    parser.add_argument(
        "--disable-cuda-graph", action="store_true", help="Disable CUDA graph"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for sampling and generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--output_filename", type=str, default=None, help="Output filename"
    )
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    return parser.parse_args()


def main():
    args = parse_args()
    kill_gpu_processes()
    server_process, port = start_server(args)
    test_prefills(args, port)
    server_process.terminate()


if __name__ == "__main__":
    main()
