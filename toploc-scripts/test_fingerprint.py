import json
import os
import signal
import sys
from pathlib import Path

from dotenv import load_dotenv

DISABLE_CUDA_GRAPH = True

FIXED_SEED = 42
MAYBE_DISABLE_CUDA_GRAPH = "--disable-cuda-graph" if DISABLE_CUDA_GRAPH else ""

# load from .env in the same directory as this script

script_dir = Path(__file__).parent.absolute()
env_path = script_dir / ".env"

# determine if any processes are running on the card and if so kill them (nvidia-smi)
import subprocess

cmd = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
result = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE)
pids = [line.decode().strip() for line in result.stdout.splitlines()]
for pid in pids:
    print(f"Killing process {pid}")
    os.kill(int(pid), signal.SIGKILL)


if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    raise Exception(".env file not found")

hf_token = os.getenv("HF_TOKEN", "")
if not hf_token:
    raise Exception("HF_TOKEN environment variable not set")

from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server

# Launch the server
if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

print("Starting server with TopLoc fingerprint verification enabled...")
server_process, port = launch_server_cmd(
    f"""
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --toploc-fingerprint --log-level debug {MAYBE_DISABLE_CUDA_GRAPH}
"""
)

# Wait for the server to start
wait_for_server(f"http://localhost:{port}")

# Add an additional delay to ensure server is fully initialized
import time

print("Waiting 3 more seconds for server to be fully initialized...")
time.sleep(3)

# Send an inference request using the OpenAI client
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

params = {
    "temperature": 0,
    "seed": FIXED_SEED,
}

print("Sending request with toploc-fingerprint enabled on server...")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    **params,
    extra_body={"return_input_ids": True, "return_output_ids": True},
)

# Print the response
print("Response received:")
response_dump = response.model_dump()
print(json.dumps(response_dump, indent=4))

original_content = response_dump["choices"][0]["message"]["content"]
last_token_proof = response_dump["choices"][0]["message"]["verification_proofs"][-1]

# Do a prefill
prefill_response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": original_content},
    ],
    max_tokens=0,
    **params,
    extra_body={
        "verification_proof_to_validate": last_token_proof,
        "return_input_ids": True,
        "return_output_ids": True,
    },
)

prefill_dump = prefill_response.model_dump()
print("Prefill response received:")
print(json.dumps(prefill_dump, indent=4))


# Terminate the server
print("Terminating server...")
terminate_process(server_process)
