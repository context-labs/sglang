import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import List, Tuple

import numpy as np
import openai
import torch
from dotenv import load_dotenv
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from scipy.stats import chi2
from tqdm import tqdm

from sglang.srt.entrypoints.engine import _launch_subprocesses
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

DEBUGGING = True

load_dotenv()

# Import the API for performing inferences. Adjust the import if necessary for your codebase.
import sglang.api

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)


_global_state = {}


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


async def collect_logprobs(port, args):
    inferences = load_inferences(args)
    all_logprobs_results = []

    if args.N is None:
        args.N = len(inferences)

    # Create a tokenizer manager so that we can exactly replicate the openai API codepath for tokenization
    model, *_ = args.model.split(";")
    tokenizer_manager = _global_state["tokenizer_manager"]

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

        # Now we can isolate the prompt IDs from the reponse IDs by removing the prompt ID prefix
        assert is_prefix(
            prompt_token_ids, prompt_response_token_ids
        ), "Prefix check failed"

        # Let's get the top log-probs associated with the prompt+response prefill sequence
        top_logprob_num = 200
        model, *_ = model.split(";")
        logprobs_request = make_logprobs_request(
            prefill_request, prompt_response_token_ids, top_logprob_num
        )
        ret = await get_top_logprobs_from_LLM(
            model, logprobs_request, prompt_response_token_ids, top_logprob_num
        )
        with open("ret.json", "w") as f:
            json.dump(ret, f, indent=2)
        M = gather_logprobs(ret, top_logprob_num)

        input_token_logprobs_from_ret = [
            item[1] for item in ret["meta_info"]["input_token_logprobs"][1:]
        ]

        # Get the response token IDs
        response_token_ids = prompt_response_token_ids[len(prompt_token_ids) + 1 :]
        M_response = M[len(prompt_token_ids) :,]
        assert M_response.shape[0] == len(
            response_token_ids
        ), f"{M_response.shape[0]} != {len(response_token_ids)}"
        assert (
            input_token_logprobs_from_ret[len(prompt_token_ids) :] == response_token_ids
        )

        p_value, test_statistic, num_unlikely_tokens_detected = (
            calculate_p_value_from_sparse_matrix(
                M_response, response_token_ids, top_logprob_num
            )
        )

        print(p_value, test_statistic, num_unlikely_tokens_detected)

        # Store the logprobs for this inference
        logprobs_results = {
            "inference_id": i,
            "prompt": inference["complete_request"]["messages"],
            "response": inference["complete_response"]["choices"][0]["message"][
                "content"
            ],
            "p_value": float(p_value),
            "test_statistic": float(test_statistic),
            "num_unlikely_tokens_detected": int(num_unlikely_tokens_detected),
        }

        # Add to the collection
        all_logprobs_results.append(logprobs_results)

    return all_logprobs_results


async def get_top_logprobs_from_LLM(model, request, token_ids, top_logprob_num):
    for key, value in request["extra_body"].items():
        request[key] = value
    del request["extra_body"]

    assert (
        request["max_tokens"] == 0
    ), f"Not a prefill request, had max_tokens = {request['max_tokens']}"
    assert (
        request["logprobs"] == True
    ), f"Not a logprobs request, had logprobs = {request['logprobs']}"
    assert (
        request["top_logprobs"] >= 0
    ), f"Not a top_logprobs request, had top_logprobs = {request['top_logprobs']}"
    assert (
        request["logprob_start_len"] == 0
    ), f"logprob_start_len must be GT 0, had logprob_start_len = {request['logprob_start_len']}"

    # Use the same codepaths used when sending requests through the API (this is important for reproducibility)
    all_requests = [ChatCompletionRequest(**request)]
    tokenizer_manager = _global_state["tokenizer_manager"]
    adapted_request, request = v1_chat_generate_request(all_requests, tokenizer_manager)
    adapted_request.input_ids = [token_ids]
    try:
        ret = await tokenizer_manager.generate_request(adapted_request).__anext__()
    except ValueError as e:
        raise e
    if isinstance(ret, list):
        ret = ret[0]
    return ret


def gather_logprobs(ret, top_logprob_num):

    # Get the Top N logprobs for each token in the sequence into arrays of [T,N]

    print("len input_top_logprobs[1:]", len(ret["meta_info"]["input_top_logprobs"][1:]))

    seq_logprobs, seq_token_ids = [], []
    for top_tokens in tqdm(
        ret["meta_info"]["input_top_logprobs"][1:], desc="Gathering logprobs"
    ):
        logprobs, token_ids = [], []
        for nth_token in top_tokens:
            logprobs.append(nth_token[0])
            token_ids.append(nth_token[1])
        seq_logprobs.append(logprobs)
        seq_token_ids.append(token_ids)

    # Get the size of the vocabulary dimension from the tokenizer
    tokenizer_manager = _global_state["tokenizer_manager"]
    VOCAB_DIM = len(
        tokenizer_manager.tokenizer
    )  # see: https://github.com/huggingface/transformers/blob/5f4ecf2d9f867a1255131d2461d75793c0cf1db2/src/transformers/tokenization_utils_fast.py#L275

    # Construct the sparse matrix M
    # We need to prepare the COO format: data, (row_ind, col_ind)
    data = []  # Values (logprobs)
    row_indices = []  # Row indices
    col_indices = []  # Column indices (token IDs)

    # Populate the coordinates and data
    for i in range(len(seq_logprobs)):
        for j in range(len(seq_logprobs[i])):
            data.append(seq_logprobs[i][j])
            row_indices.append(i)
            col_indices.append(seq_token_ids[i][j])

    # Construct the sparse matrix with dimensions [T, VOCAB_DIM]
    M = coo_matrix(
        (data, (row_indices, col_indices)), shape=(len(seq_logprobs), VOCAB_DIM)
    )

    # Transform to CSR format for better efficiency
    M = M.tocsr()

    return M


def calculate_p_value_from_dense_matrix(M: np.ndarray, token_ids):
    assert isinstance(M, np.ndarray)
    # Transform to probabilities
    M_p = np.exp(M)
    # Get the observed probabilities for each token id at each position in the token sequence
    P_obs = M_p[np.arange(M_p.shape[0]), token_ids]
    # Calculate the tail mass for each position
    tail_mass = (M_p < P_obs.reshape([-1, 1])).sum(axis=1)
    # Apply mid-rank correction
    tail_mass -= 0.5 * P_obs
    # Simulate uniform distribution over the frequently quite large P_obs when temperature is low
    tail_mass += P_obs * (np.random.rand(len(P_obs)) - 0.5)
    # Clip for safety
    np.clip(tail_mass, 1e-323, 1.0, out=tail_mass)
    # Compute test statistic
    F = -2.0 * np.log(tail_mass).sum()
    # Compute p-value using chi-squared distribution (Fisher's method)
    p_value = chi2.sf(F, df=2 * M_p.shape[0])
    return p_value


def calculate_p_value_from_sparse_matrix(M: csr_matrix, token_ids, top_logprob_num):
    assert isinstance(M, csr_matrix)
    T = M.shape[0]
    VOCAB_DIM = M.shape[1]

    # Transform the sparse matrix from log-probs to probs (but only on the defined entries)
    M_p = M.copy()
    M_p.data = np.exp(M_p.data)
    assert isinstance(M_p, csr_matrix)

    # Get the observed probabilities for each token id at each position in the token sequence
    # Annoyingly, scipy.sparse matrix indexing returns a np.matrix, not an ndarray
    # The .A1 gets the flattened representation of the matrix, but I still ravel() just in case scipy.sparse changes its return type
    P_obs = M_p[np.arange(T), token_ids]
    if isinstance(P_obs, np.matrix):
        P_obs = P_obs.A1
    P_obs = P_obs.ravel()
    assert P_obs.shape == (T,), f"{P_obs.shape} != ({T},)"

    # P_obs will be zero if the token is not in the top_logprob_num (let's just say 200)
    # So, we have to substitute P_obs at those indices with a reasonable value
    # 1. Calculate the remaining probability mass after the top 200 tokens
    unlikely_tokens_total_mass = 1.0 - M_p.sum(axis=1)
    print(type(unlikely_tokens_total_mass), unlikely_tokens_total_mass.shape)
    if isinstance(unlikely_tokens_total_mass, np.matrix):
        unlikely_tokens_total_mass = unlikely_tokens_total_mass.A1
    unlikely_tokens_total_mass = unlikely_tokens_total_mass.ravel()
    assert unlikely_tokens_total_mass.shape == (
        T,
    ), f"{unlikely_tokens_total_mass.shape} != ({T},)"

    # 2. Spread the remaining probability mass out equally over the remaining non-top-200 tokens
    num_unlikely_tokens = VOCAB_DIM - top_logprob_num
    default_unlikely_token_prob = unlikely_tokens_total_mass / num_unlikely_tokens
    assert default_unlikely_token_prob.shape == (
        T,
    ), f"{default_unlikely_token_prob.shape} != ({T},)"

    # 3. Create a mask representing where a token was not part of the top-200
    dok = M_p.todok()
    unlikely_token_mask = np.asarray(
        [(i, token_id) not in dok for i, token_id in enumerate(token_ids)], dtype=bool
    )
    assert unlikely_token_mask.shape == (T,), f"{unlikely_token_mask.shape} != ({T},)"

    # 4. Substitute in our default values at these positions
    P_obs[unlikely_token_mask] = default_unlikely_token_prob[unlikely_token_mask]

    # Next, we need to measure the tail mass (sum of all probabilities in M_p[i] less than P_obs[i])
    # If we do the naive M_p < P_obs.reshape([-1,1]), we'll get a dense matrix
    # So instead, we'll process each row of the sparse matrix efficiently
    tail_mass = np.zeros(T)
    token_ranks = []
    for i in range(T):
        row_start = M_p.indptr[i]
        row_end = M_p.indptr[i + 1]
        # Get the data and indices for this row
        data = M_p.data[row_start:row_end]
        # what position is this token in the sorted list of token probabilities?
        token_ranks.append(np.sum(data > P_obs[i]) + 1)
        # Sum the values that are less than or equal to P_obs[i]
        tail_mass[i] = np.sum(data[data <= P_obs[i]])

    # Manually add the total mass of the unlikely tokens to each row, since that isn't included in the sparse matrix
    tail_mass += unlikely_tokens_total_mass

    # But if we are an unlikely token, choose a random tail mass between 0 and unlikely_tokens_total_mass[i]
    num_unlikely_tokens_detected = np.sum(unlikely_token_mask)
    if num_unlikely_tokens_detected > 0:
        tail_mass[unlikely_token_mask] = unlikely_tokens_total_mass[
            unlikely_token_mask
        ] * np.random.rand(num_unlikely_tokens_detected)

    # Ok, now we have an accurate tail_mass for top-200 tokens, and plausible (but randomized) for non-top-200 tokens
    # Now we can proceed normally.

    # Mid-rank correction
    # U = tail_mass - 0.5 * P_obs

    # Uniform distribution correction
    U = tail_mass - P_obs * (np.random.rand(len(P_obs)))

    # Spot-check - should give test statistic around 2*T-2
    # U = np.random.rand(len(P_obs))

    # Clipping for safety
    np.clip(U, 1e-323, 1.0, out=U)
    # Test statistic
    F = -2.0 * np.log(U).sum()
    # p-value
    p_value = chi2.sf(F, df=2 * T)

    chi2(df=2 * T).mean()

    if DEBUGGING:
        for i in range(T):
            print(
                f"""Token {i}:
    Token ID: {token_ids[i]},
    P_obs = {P_obs[i]:.4f},
    tail_mass = {tail_mass[i]:.4f},
    U = {U[i]:.4f},
    Fi = {-2*np.log(U[i]):.4f},
    Token Rank: {token_ranks[i]},
    Is Unlikely Token: {unlikely_token_mask[i]},
    Unlikely Token Total mass: {unlikely_tokens_total_mass[i]:.4f},
    Default Unlikely Token Prob: {default_unlikely_token_prob[i]:.4f}"""
            )

        print(f"F = {F:.4f}")
        print(f"chi2 mode: {2*T - 2}")
        print(f"chi2 stdev: {(2 * 2*T)**0.5:.4f}")

    return p_value, F, num_unlikely_tokens_detected


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
        eos_token_id = tokenizer_manager.tokenizer.eos_token_id
        token_ids.append(eos_token_id)

    # Omit the begin-of-sequence token
    if token_ids[0] == tokenizer_manager.tokenizer.bos_token_id:
        return token_ids[1:]
    return token_ids


def make_logprobs_request(prefill_request, token_ids, top_logprob_num):
    request = copy(prefill_request)
    # request["messages"] = token_ids
    request["extra_body"] = request.get("extra_body", {})
    request["logprobs"] = True
    request["top_logprobs"] = top_logprob_num
    request["extra_body"]["logprob_start_len"] = 0  # Start from beginning
    # request["extra_body"]["return_text_in_logprobs"] = True
    return request


def get_logprobs(client, model, request):
    """
    Get the full token distribution for each token in the input_ids.

    Args:
        model (str): The model to use for inference
        request (dict): The request object with parameters for the API call

    Returns:
        List: A list of dictionaries containing the token distributions for each token
    """
    # Initialize OpenAI client with base URL pointing to the local SGL server

    try:
        # Make the API call using OpenAI-compatible API
        response = client.chat.completions.create(**request)

        # Convert the response to a dictionary for easier handling
        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else response
        )

        print(json.dumps(response_dict, indent=2))

        # Extract token distributions from the response
        token_distributions = []

        # If we reach here, neither format matched
        print("Warning: Could not extract token distributions from the response")
        return []

    except Exception as e:
        print(f"Error when trying to get logprobs: {e}")
        return []


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


def launch_engine(args):
    model, *_ = args.model.split(";")
    server_args = get_server_args(args)
    server_args = ServerArgs(
        **server_args
    )  # TODO: the other flags like disable cuda graph
    tokenizer_manager, scheduler_info = _launch_subprocesses(server_args=server_args)
    _global_state["tokenizer_manager"] = tokenizer_manager
    _global_state["scheduler_info"] = scheduler_info


def shut_down_engine():
    tokenizer_manager = _global_state["tokenizer_manager"]
    scheduler_info = _global_state["scheduler_info"]
    _global_state["tokenizer_manager"] = None
    _global_state["scheduler_info"] = None
    tokenizer_manager.shutdown()
    scheduler_info.shutdown()


def get_server_args(args):
    # other things to consider in future: grammar backend, etc.
    model, *quantization = args.model.split(";")
    disable_cuda_graph = args.disable_cuda_graph
    return {
        "model_path": model,
        "quantization": quantization[0] if quantization else None,
        "disable_cuda_graph": disable_cuda_graph,
        "log_level": "debug" if not args.quiet else None,
    }


async def main():

    args = parse_args()

    # Dump the args
    print(args)
    if args.interactive:
        input("Press Enter when ready to continue...")

    kill_gpu_processes()
    launch_engine(args)
    port = 0
    logprobs = await collect_logprobs(port, args)

    print("writing to disk...")
    write_to_file(args, logprobs)

    print("Finished!")
    return


if __name__ == "__main__":
    asyncio.run(main())
