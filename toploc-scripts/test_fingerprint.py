import json
import time

from dotenv import load_dotenv

load_dotenv()

from sglang.test.test_utils import is_in_ci
from sglang.utils import print_highlight, terminate_process, wait_for_server

# Launch the server
if is_in_ci():
    from patch import launch_server_cmd
else:
    from sglang.utils import launch_server_cmd

server_process, port = launch_server_cmd(
    """
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0
"""
)

# Wait for the server to start
wait_for_server(f"http://localhost:{port}")

# Send an inference request using the OpenAI client
import openai

client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0,
)

# Print the response
print(json.dumps(response.model_dump(), indent=4))

# Terminate the server
terminate_process(server_process)
