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
        help="Filename to write the results (goes to replications directory)",
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


def start_server(args, model_path):
    """
    Start the SGL server.
    """
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
        python -m sglang.launch_server --model-path {model_path} --host 0.0.0.0 {MAYBE_NOISY} {MAYBE_DISABLE_CUDA_GRAPH}
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


def perform_replications(fingerprints, machine_name, args):
    """
    Rerun the prompts from the fingerprints and collect new responses.
    """
    replication_results = []
    server_process = None
    port = None
    client = None

    for i, item in enumerate(tqdm(fingerprints)):
        prompt = item["prompt"]
        model_from_fingerprint = item.get("model")
        original_request = item["complete_request"]
        original_response = item["complete_response"]
        original_fingerprint = item["fingerprint"]

        # Start server with the model from the first fingerprint if not already running
        if port is None:
            model_to_use = model_from_fingerprint
            print(f"Starting server with model {model_to_use}...")
            kill_gpu_processes()
            server_process, port = start_server(args, model_to_use)
            client = openai.Client(
                base_url=f"http://127.0.0.1:{port}/v1", api_key="None"
            )

        # Copy all parameters from the original request
        request = dict(original_request)

        # Ensure we're using chat completion format
        if "messages" not in request:
            request["messages"] = [{"role": "user", "content": prompt}]

        # Make sure we're not verifying fingerprints in this run
        if "extra_body" in request:
            extra_body = request["extra_body"]
            if "toploc_verification_fingerprint_to_validate" in extra_body:
                del extra_body["toploc_verification_fingerprint_to_validate"]

        try:
            response = client.chat.completions.create(**request)
            response_dump = response.model_dump()

            # Create result entry
            replication_result = {
                "replication_machine": machine_name,
                "inference_machine": item["machine"],
                "prompt": prompt,
                "original_request": original_request,
                "original_response": original_response,
                "original_fingerprint": original_fingerprint,
                "replication_request": request,
                "replication_response": response_dump,
            }

            replication_results.append(replication_result)
        except Exception as e:
            print(f"Error replicating prompt {i}: {e}")
            replication_results.append(
                {
                    "replication_machine": machine_name,
                    "inference_machine": item["machine"],
                    "prompt": prompt,
                    "original_request": original_request,
                    "original_response": original_response,
                    "original_fingerprint": original_fingerprint,
                    "replication_request": request,
                    "error": str(e),
                }
            )

    # Return both the results and the server process for cleanup
    return replication_results, server_process


def write_to_file(args, replication_results):
    """
    Write replication results to the output file.
    """
    # Create replications directory if it doesn't exist
    replications_dir = os.path.join(SCRIPT_DIR, "replications")
    os.makedirs(replications_dir, exist_ok=True)

    if args.output_file is None:
        first_result = replication_results[0]
        model = first_result["replication_request"]["model"]
        model_prefix = model.replace("/", "_")
        args.output_file = (
            model_prefix + "_replications_for_" + os.path.basename(args.input_file)
        )

    output_filepath = os.path.join(replications_dir, args.output_file)
    with open(output_filepath, "w") as f:
        json.dump(replication_results, f, indent=4)
    print(f"Replication results written to {output_filepath}")


def main():
    args = parse_args()
    fingerprints = load_fingerprints(args)
    print(f"Loaded {len(fingerprints)} fingerprints, preparing to replicate...")

    try:
        # Server will be started within perform_replications when processing the first fingerprint
        replication_results, server_process = perform_replications(
            fingerprints, args.machine, args
        )
        write_to_file(args, replication_results)

        # Print a summary of replication results
        print(f"\nReplication summary:")
        print(f"Total replications: {len(replication_results)}")
        success_count = sum(
            1 for result in replication_results if "error" not in result
        )
        print(f"Successful replications: {success_count}")
        print(f"Failed replications: {len(replication_results) - success_count}")
    finally:
        if server_process:
            print("Terminating server...")
            terminate_process(server_process)
            print("Server terminated.")


if __name__ == "__main__":
    main()
