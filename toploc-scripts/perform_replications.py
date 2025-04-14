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
        "--override-model",
        type=str,
        required=False,
        default=None,
        help="Override model to use when replicating",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="JSON filename containing inferences to replicate (from the inferences_to_replicate directory)",
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
        help="Limit number of inferences to replicate (whole file if not supplied)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Number of inferences to replicate (whole file if not supplied)",
    )
    parser.add_argument("--skip-write", action="store_true")
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


def load_inferences(args):
    """
    Load inferences from the input file.
    """
    input_filepath = args.input_file
    if not os.path.isabs(input_filepath):
        input_filepath = os.path.join(
            SCRIPT_DIR, "inferences_to_replicate", input_filepath
        )

    print(f"Loading inferences from {input_filepath}")
    with open(input_filepath, "r") as f:
        inferences = json.load(f)

    if args.limit is not None and args.limit > 0:
        inferences = inferences[: args.limit]
        print(f"Limited to first {args.limit} inferences")

    return inferences


def perform_replications(inferences, machine_name, args):
    """
    Rerun the prompts from the inferences and collect new responses.
    """
    replication_results = []
    server_process = None
    port = None
    client = None

    for i, item in enumerate(tqdm(inferences)):
        if args.N is not None and i >= args.N:
            break
        prompt = item["prompt"]
        model_from_inference = item.get("model")
        original_request = item["complete_request"]
        original_response = item["complete_response"]

        # Start server with the model from the first inference if not already running
        if port is None:
            model_to_use = args.override_model or model_from_inference
            print(f"Starting server with model {model_to_use}...")
            kill_gpu_processes()
            server_process, port = start_server(args, model_to_use)
            client = openai.Client(
                base_url=f"http://127.0.0.1:{port}/v1", api_key="None"
            )

        # Copy all parameters from the original request
        request = dict(original_request)

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
                "replication_request": request,
                "replication_response": response_dump,
            }

            replication_results.append(replication_result)

            original_response_text = original_response["choices"][0]["message"][
                "content"
            ]
            replication_response_text = response_dump["choices"][0]["message"][
                "content"
            ]

            if original_response_text != replication_response_text:
                prefix_match_len = (
                    calculate_prefix_match_length(
                        original_response_text, replication_response_text
                    )
                    or 0
                )
                prefix_match_percent = (
                    prefix_match_len / len(original_response_text) * 100
                )
                print(
                    f"   >>> Prompt {i} did not match original response (prefix %: {prefix_match_percent:.2f}, response lengths: {len(original_response_text)} : {len(replication_response_text)})"
                )
                print(
                    f"Divergence:\n\t{original_response_text[prefix_match_len:prefix_match_len+10]}\n\t{replication_response_text[prefix_match_len:prefix_match_len+10]}"
                )
            else:
                print(f"   >>> Prompt {i} matched original response")

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
                    "prefix_match_length": prefix_match_len,
                    "prefix_match_percent": prefix_match_percent,
                    "original_response_length": len(original_response_text),
                    "replication_response_length": len(replication_response_text),
                }
            )

    # Return both the results and the server process for cleanup
    return replication_results, server_process


def write_to_file(args, replication_results):
    """
    Write replication results to the output file.
    """
    if args.output_file is None:
        first_result = replication_results[0]
        model = first_result["replication_request"]["model"]
        model_prefix = model.replace("/", "_")
        args.output_file = (
            model_prefix + "_replications_for_" + os.path.basename(args.input_file)
        )

    replications_dir = os.path.join(SCRIPT_DIR, "replications")
    os.makedirs(replications_dir, exist_ok=True)
    output_filepath = os.path.join(replications_dir, args.output_file)
    with open(output_filepath, "w") as f:
        json.dump(replication_results, f, indent=4)
    print(f"Replication results written to {output_filepath}")


def main():
    args = parse_args()
    inferences = load_inferences(args)
    print(f"Loaded {len(inferences)} inferences, preparing to replicate...")

    server_process = None
    try:
        # Server will be started within perform_replications when processing the first inference
        replication_results, server_process = perform_replications(
            inferences, args.machine, args
        )

        if not args.skip_write:
            write_to_file(args, replication_results)

        # Print a summary of replication results
        print(f"\nReplication summary:")
        print(f"Total replications: {len(replication_results)}")
        no_error_count = sum(
            1 for result in replication_results if "error" not in result
        )
        print(f"Replications with no errors: {no_error_count}")
        print(f"Replications with errors: {len(replication_results) - no_error_count}")
    finally:
        if server_process:
            print("Terminating server...")
            terminate_process(server_process)
            print("Server terminated.")


def calculate_prefix_match_length(original, replication):
    length = max(len(original), len(replication))
    original = original + " " * (length - len(original))
    replication = replication + " " * (length - len(replication))
    return sum(1 for o, r in zip(original, replication) if o == r)


if __name__ == "__main__":
    main()
