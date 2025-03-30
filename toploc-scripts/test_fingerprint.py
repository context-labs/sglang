import json
import os
import time

from dotenv import load_dotenv

# load from .env in the same directory as this script
load_dotenv()

# Set the HF_TOKEN environment variable
print("hf_token", os.getenv("HF_TOKEN", ""))

from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server

# Launch the server
if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

print_highlight("Starting server with TopLoc fingerprint verification enabled...")
server_process, port = launch_server_cmd(
    """
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --toploc-fingerprint
"""
)

# Wait for the server to start
wait_for_server(f"http://localhost:{port}")

# Send an inference request using the OpenAI client
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

print_highlight("Sending request with verification_proofs=True...")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0,
    extra_body={"return_verification_proofs": True},  # Request verification proofs
)

# Print the response
print_highlight("Response received:")
response_dump = response.model_dump()
print(json.dumps(response_dump, indent=4))

# Check if verification proofs are in the response
if "choices" in response_dump and len(response_dump["choices"]) > 0:
    message = response_dump["choices"][0].get("message", {})
    if "verification_proofs" in message:
        if message["verification_proofs"] is not None:
            print_highlight(
                "SUCCESS: Verification proofs are included in the response!"
            )
            print(f"Number of proof sets: {len(message['verification_proofs'])}")
        else:
            print_highlight(
                "PARTIAL SUCCESS: verification_proofs field exists but is null",
                color="yellow",
            )
            print("We need to debug why proofs aren't being generated")
    else:
        print_highlight(
            "ERROR: No verification_proofs field in the response", color="red"
        )
        print("Available message fields:", list(message.keys()))

# Terminate the server
print_highlight("Terminating server...")
terminate_process(server_process)
