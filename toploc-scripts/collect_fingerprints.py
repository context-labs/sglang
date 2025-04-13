import json
import os
import signal
import subprocess
import sys
import time
from argparse import ArgumentParser

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
    raise ValueError("HF_TOKEN not found in environment variables")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--machine", type=str, required=True, help="Machine name")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model to use",
    )
    parser.add_argument(
        "--ultrachat_file", type=str, default="train_0.jsonl", help="ultrachat filename"
    )
    parser.add_argument(
        "--N", type=int, required=True, help="Number of requests to process"
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


def collect_N_fingerprints(port, args):
    client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")
    fingerprints = []
    ultrachat_filepath = os.path.join(SCRIPT_DIR, "ultrachat", args.ultrachat_file)

    with open(ultrachat_filepath, "r") as f:
        for i, line in enumerate(tqdm(f)):
            if i >= args.N:
                break

            data = json.loads(line)
            prompt = data["data"][0]  # Assuming the first element is the user prompt

            request = dict(
                model=args.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                temperature=args.temperature,
                seed=args.seed,
                extra_body={"return_verification_proofs": True},
            )

            response = client.chat.completions.create(**request)
            response_dump = response.model_dump()
            fingerprint = response_dump["choices"][0]["message"][
                "toploc_verification_fingerprints"
            ][-1]

            fingerprints.append(
                {
                    "machine": args.machine,
                    "prompt": prompt,
                    "complete_request": request,
                    "complete_response": response_dump,
                    "model": args.model,
                    "fingerprint": fingerprint,
                }
            )

    return fingerprints


def write_to_file(args, fingerprints):
    if args.output_filename is None:
        ultrachat_no_ext = os.path.splitext(args.ultrachat_file)[0]
        args.output_filename = args.model.replace("/", "_") + "_for_" + ultrachat_no_ext
    output_filepath = os.path.join(SCRIPT_DIR, "fingerprints", args.output_filename)
    with open(output_filepath, "w") as f:
        json.dump(fingerprints, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    kill_gpu_processes()
    server_process, port = start_server(args)
    fingerprints = collect_N_fingerprints(port, args)
    write_to_file(args, fingerprints)
    server_process.terminate()
    print("Server terminated.")
