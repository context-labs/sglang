import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import List, Tuple

import openai
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api.adapter import v1_chat_generate_request
from sglang.srt.openai_api.protocol import ChatCompletionRequest
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.utils import (
    launch_server_cmd,
    print_highlight,
    terminate_process,
    wait_for_server,
)

load_dotenv()

# Import the API for performing inferences. Adjust the import if necessary for your codebase.
import sglang.api

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect logprobs for inferences.")
    parser.add_argument("--model", type=str, required=True, help="Model to use")
    parser.add_argument("--input-file", type=str, required=True, help="Input file")
    parser.add_argument("--machine", type=str, required=True, help="Machine name")
    parser.add_argument("--output-file", type=str, required=True, help="Output file")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Number of inferences to collect logprobs for (all if not supplied)",
    )
    return parser.parse_args()


def kill_gpu_processes():
    cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
    pids = [line.decode().strip() for line in result.stdout.splitlines()]
    for pid in pids:
        print(f"Killing process {pid}")
        os.kill(int(pid), signal.SIGKILL)


def load_inferences(args):
    input_filename = args.input_file
    print(f"Loading inferences from {input_filename}")
    input_filepath = os.path.join(ROOT_DIR, "inferences_to_replicate", input_filename)
    with open(input_filepath, "r") as f:
        return json.load(f)


def collect_logprobs(port, args):
    inferences = load_inferences(args)
    logprobs = []

    if args.N is None:
        args.N = len(inferences)

    # Create a tokenizer manager so that we can exactly replicate the openai API codepath for tokenization
    model, *_ = args.model.split(";")
    server_args = ServerArgs(model_path=model)
    port_args = PortArgs.init_new(server_args)
    tokenizer_manager = TokenizerManager(server_args, port_args)

    for i, inference in enumerate(
        tqdm(inferences[: args.N], desc="Collecting logprobs")
    ):

        # Create a request for the purpose of getting the token IDs of just the prompt
        request = make_prompt_request(inference, model)
        prompt_token_ids = get_token_ids(
            tokenizer_manager, args.model, request, add_eos_id=False
        )

        # Create a pre-fill for the purpose of getting the token IDs of both the prompt & response
        prefill_request = make_prefill_request(inference, model)
        prompt_response_token_ids = get_token_ids(
            tokenizer_manager, model, prefill_request, add_eos_id=True
        )

        # Sanity Check.  The tokenizer is complicated. This passes for the first 100 prompts from ultrachat.
        assert is_prefix(
            prompt_token_ids, prompt_response_token_ids
        ), "Prefix check failed"

        # Now we have a safe way of extracting from the logprobs which region of token IDs belongs to the prompt and the response
        # So, let's get the log-probs associated with the prompt+response prefill sequence
        logprobs_request = make_logprobs_request(
            prefill_request, prompt_response_token_ids
        )
        logprobs = get_logprobs(logprobs_request, prompt_response_token_ids)

        # Get the logprobs for the entire pre-fill
        logprobs = get_logprobs(args.model, prefill_request)

    return logprobs


def is_prefix(prefix_ids, ids):
    for a, b in zip(prefix_ids, ids):
        if a != b:
            return False
    return True


def write_to_file(args, logprobs):
    output_filename = args.output_file
    os.makedirs(os.path.join(ROOT_DIR, "response_logprobs"), exist_ok=True)
    output_filepath = os.path.join(ROOT_DIR, "response_logprobs", output_filename)
    with open(output_filepath, "w") as f:
        json.dump(logprobs, f)


def make_prompt_request(inference, model):
    original_messages = inference["complete_request"]["messages"]
    original_response = inference["complete_response"]["choices"][0]["message"][
        "content"
    ]

    prefill_messages = original_messages
    original_request = inference["complete_request"]
    prefill_request = original_request.copy()
    prefill_request["messages"] = prefill_messages
    prefill_request["max_tokens"] = 0
    prefill_request["extra_body"] = {"input_token_ids": True}
    prefill_request["model"] = model
    return prefill_request


def make_prefill_request(inference, model):
    original_messages = inference["complete_request"]["messages"]
    original_response = inference["complete_response"]["choices"][0]["message"][
        "content"
    ]

    prefill_messages = [
        *original_messages,
        {"role": "assistant", "content": original_response},
    ]
    original_request = inference["complete_request"]
    prefill_request = original_request.copy()
    prefill_request["messages"] = prefill_messages
    prefill_request["max_tokens"] = 0
    prefill_request["extra_body"] = {"input_token_ids": True}
    prefill_request["model"] = model
    return prefill_request


def get_token_ids(tokenizer_manager, model, request, add_eos_id=True):
    # Create an OpenAI API style request
    chat_request = ChatCompletionRequest(**request)

    # Use the same function that is used for OpenAI style API requests to gen token ids
    adapted_request, original_request = v1_chat_generate_request(
        [chat_request], tokenizer_manager
    )

    # The token IDs will be in adapted_request
    token_ids = adapted_request.input_ids

    # We need to add the end of sequence token because this is a prefill, so we won't have that the END-OF-SEQUENCE token
    if add_eos_id:
        tokenizer = tokenizer_manager.tokenizer
        eos_token_id = tokenizer.eos_token_id
        token_ids.append(eos_token_id)

    return token_ids


def make_logprobs_request(request, token_ids):
    request = copy(request)
    request["input_token_ids"] = token_ids
    return request


def get_logprobs(model, request) -> List[Tuple[int, float]]:
    # engine = Engine(model)
    return None


def copy(x):
    return json.loads(json.dumps(x))


def start_server(args):
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

    model_path = args.model

    print(f"Starting server with model {model_path}...")

    model, *quantization = model_path.split(";")
    if quantization:
        quantization = quantization[0]
        print(f"Quantization: {quantization}")
        MAYBE_QUANTIZATION = f"--quantization {quantization}"
    else:
        MAYBE_QUANTIZATION = ""

    server_process, port = launch_server_cmd(
        f"""
        python -m sglang.launch_server --model-path {model} {MAYBE_QUANTIZATION} --host 0.0.0.0 {MAYBE_NOISY} {MAYBE_DISABLE_CUDA_GRAPH}
        """
    )

    print(f"Starting on port {port}...")

    # Wait for the server to start
    wait_for_server(f"http://localhost:{port}")

    # Add additional delay to ensure server is fully initialized
    print("Waiting 3 more seconds for server to be fully initialized...")
    time.sleep(3)

    print(
        f"Started server with model {model} on port {port} {MAYBE_QUANTIZATION}, {MAYBE_NOISY}, {MAYBE_DISABLE_CUDA_GRAPH}"
    )
    if args.interactive:
        input("Press Enter when ready to continue...")

    return server_process, port


def main():

    args = parse_args()
    kill_gpu_processes()
    server_process, port = start_server(args)
    logprobs = collect_logprobs(port, args)
    write_to_file(args, logprobs)
    server_process.terminate()
    print("Server terminated.")


if __name__ == "__main__":
    main()
