"""Patched server that imports and runs the functionality from sglang.launch_server with a spy on v1_chat_generate_request"""

import os
import sys
import json
import functools
import time
import base64
import logging
from typing import List, Any, Optional, Dict, Tuple
from toploc import build_proofs_base64
import torch
import numpy as np
# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s] [SPY] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("sglang_spy")

# Patch the OpenAI API models to support extra fields
from sglang.srt.openai_api.protocol import ChatCompletionResponse, ChatCompletionResponseChoice
from pydantic import ConfigDict, BaseModel, create_model, Field

# Backup the original classes
OriginalChatCompletionResponseChoice = ChatCompletionResponseChoice

# Create a new message class with an extra 'hidden_states' field
class ExtendedChatCompletionResponseChoice(OriginalChatCompletionResponseChoice):
    fingerprint: Optional[str] = None

# Apply the patch
import sglang.srt.openai_api.protocol
sglang.srt.openai_api.protocol.ChatCompletionResponseChoice = ExtendedChatCompletionResponseChoice

ChatCompletionResponse.model_rebuild()

# Import the original functions 
from sglang.srt.openai_api.adapter import v1_chat_generate_request as original_v1_chat_generate_request
from sglang.srt.openai_api.adapter import v1_chat_generate_response as original_v1_chat_generate_response

# Create a spy wrapper for the function - force every request to return hidden states
def v1_chat_generate_request_spy(*args, **kwargs):
    logger.info("v1_chat_generate_request_spy called")
    result = original_v1_chat_generate_request(*args, **kwargs)
    adapted_request, all_requests = result
    logger.info(f"Setting return_hidden_states=True on adapted_request")
    adapted_request.return_hidden_states = True
    return adapted_request, all_requests


# Create a spy wrapper for the function - add hidden states directly
def v1_chat_generate_response_spy(*args, **kwargs):
    logger.info("v1_chat_generate_response_spy called")
    
    ret = args[1]

    # Call the original function
    logger.info("Calling original v1_chat_generate_response")
    result = original_v1_chat_generate_response(*args, **kwargs)
    logger.info(f"Got result with type: {type(result).__name__}")

    # Add hidden states directly to the __dict__ of each choice
    choices = result.choices if hasattr(result, "choices") else None
    logger.info(f"Result has {len(choices) if choices else 0} choices")
    
    if choices is None:
        logger.info("No choices found, returning original result")
        return result
    
    for idx, (choice, ret_item) in enumerate(zip(choices, ret)):
        logger.info(f"Setting fingerprint on choice {idx}")                
        fingerprint = encode_fingerprint(ret_item["meta_info"]["hidden_states"])
        choice.fingerprint = fingerprint
    
    logger.info("Returning result from v1_chat_generate_response_spy")
    return result

# Import stuff we need to start the server
from sglang.launch_server import launch_server, prepare_server_args, kill_process_tree
from sglang.srt.managers.io_struct import GenerateReqInput

# Function to encode hidden states into a base64 string
def encode_fingerprint(hidden_states):
    logger.info("Encoding hidden states to fingerprint")

    # TODO: use dtype of model
    last_token_hidden_state = torch.tensor(hidden_states[-1], dtype=torch.float16, device = torch.device("cpu"))

    return build_proofs_base64(
            [last_token_hidden_state],
            decode_batching_size=1,
            topk=128,
            skip_prefill=True,
        )[0]

# Patch the functions in the module
logger.info("Patching v1_chat_generate_request and v1_chat_generate_response")
import sglang.srt.openai_api.adapter
sglang.srt.openai_api.adapter.v1_chat_generate_request = v1_chat_generate_request_spy
sglang.srt.openai_api.adapter.v1_chat_generate_response = v1_chat_generate_response_spy

if __name__ == "__main__":
    logger.info("Starting server")
    server_args = prepare_server_args(sys.argv[1:])
    
    try:
        logger.info("Launching server")
        launch_server(server_args)
    finally:
        logger.info("Cleaning up")
        kill_process_tree(os.getpid(), include_parent=False)