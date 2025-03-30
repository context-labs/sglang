import json
import os
import signal
from pathlib import Path

from dotenv import load_dotenv


def print_highlight(text, color="green"):
    print(f"\033[38;5;{color}m{text}\033[0m")


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

print_highlight("Starting server with TopLoc fingerprint verification enabled...")
server_process, port = launch_server_cmd(
    """
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --toploc-fingerprint --log-level debug
"""
)

# Wait for the server to start
wait_for_server(f"http://localhost:{port}")

# Add an additional delay to ensure server is fully initialized
import time

print_highlight("Waiting 5 more seconds for server to be fully initialized...")
time.sleep(5)

# Send an inference request using the OpenAI client
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

print_highlight("Sending request with toploc-fingerprint enabled on server...")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0,
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
