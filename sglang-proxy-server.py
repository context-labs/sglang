#!/usr/bin/env python3
from fastapi import FastAPI, Request, Response, Depends
from fastapi.responses import StreamingResponse
import httpx
from typing import AsyncGenerator, Callable, Optional, Dict, List, Any
import json
import uvicorn
import argparse
import sys
from sglang.srt.openai_api.adapter import v1_chat_generate_request
from sglang.srt.openai_api.protocol import ChatCompletionRequest

app = FastAPI()

SGLANG_SERVER_URL = None
TOKENIZER = None

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_endpoint(request: Request, path: str):
    # Intercept and log request
    body = await request.body()
    
    url = f"{SGLANG_SERVER_URL}/{path}"
    if request.query_params:
        url = f"{url}?{request.query_params}"
    
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method=request.method,
            url=url,
            headers=request.headers,
            content=body,
            timeout=None
        )

        if "text/event-stream" in response.headers.get("content-type", ""):
            log_message(f"Starting streaming response for [{path}]")
            return StreamingResponse(
                proxy_stream(response, path),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )

# Simple request/response logging to stdout
def log_message(message):
    print(f"\n{message}", file=sys.stdout, flush=True)

def maybe_parse_chunk(chunk_text: str) -> Optional[Dict[str, Any]]:
    if chunk_text.startswith("data: "):
        chunk_text = chunk_text[6:]
    if chunk_text == "[DONE]":
        return None
    try:
        return json.loads(chunk_text)
    except:
        print(f"Error parsing chunk: {chunk_text}")
        return None

async def proxy_stream(response: httpx.Response, path: str) -> AsyncGenerator[bytes, None]:
    chunk_index, parsed_chunks, template_chunk = 0, [], None
    async for chunk in response.aiter_bytes():
        chunk_text = chunk.decode('utf-8').strip()
        chunk_texts = chunk_text.split("\n")
        for chunk_text in chunk_texts:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            parsed_chunk = maybe_parse_chunk(chunk_text)
            print(f"Parsed chunk: {parsed_chunk}")
            if parsed_chunk is not None:
                parsed_chunks.append(parsed_chunk)        
            if chunk_text == "[DONE]" and len(parsed_chunks) > 0:
                toploc_fingerprint = make_toploc_fingerprint_from_chunks(parsed_chunks)
                fingerprint_chunk = create_toploc_fingerprint_chunk(parsed_chunks[-1], toploc_fingerprint)
                yield json.dumps(fingerprint_chunk).encode() + b"\n"
            yield chunk_text.encode('utf-8') + b"\n"

def make_toploc_fingerprint_from_chunks(original_request: Dict[str, Any], parsed_chunks: List[Dict[str, Any]]) -> str:
    response_text = get_response_text_from_chunks(parsed_chunks)
    chat_completion_request = make_chat_completion_request(original_request, response_text)
    generation_request = convert_to_generation_request(chat_completion_request)
    generation_request.sampling_params["max_new_tokens"] = 0
    generation_request.return_hidden_states = True
    hidden_states = get_hidden_states(generation_request)
    toploc_fingerprint = do_toploc_encoding(hidden_states)
    return toploc_fingerprint

def make_chat_completion_request(original_request: Dict[str, Any], response_text: str) -> Dict[str, Any]:
    response_augmented_messages = original_request['messages'] + [{"role": "assistant", "content": response_text}]
    return {
        **original_request,
        "messages": response_augmented_messages
    }

def convert_to_generation_request(chat_completion_request: Dict[str, Any]) -> List[int]:
    v1_chat_completion = ChatCompletion(**chat_completion_request)
    # TokenizerManager does a lot more than just manage the tokenizer (it news up an entire LLM), so we just fake it
    tokenizer_manager = FakeTokenizerManager(_global_state["tokenizer"])
    adapted_request, _ = v1_chat_generate_request([v1_chat_completion], tokenizer_manager)
    return adapted_request

class FakeTokenizerManager:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

def tokenize_chat_completion(tokenization_request: Dict[str, Any]) -> TokenizationInfo:
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    
    


def get_response_text_from_chunks(parsed_chunks: List[Dict[str, Any]]) -> str:
    response_text = ""
    for chunk in parsed_chunks:
        chunk_text += chunk['choices'][0]['message']['content']
        if chunk_text:
            response_text += chunk_text
    return response_text

def tokenize_chat_completion()

def tokenize_chat_completion(messages: List[Dict[str, Any]]) -> List[int]:
    tokenizer = AutoTokenizer.from_pretrained("Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    token_ids = tokenizer.encode(messages)
    return token_ids

def create_toploc_fingerprint_chunk(template_chunk, toploc_fingerprint: str) -> Dict[str, Any]:
    template_chunk = json.loads(json.dumps(template_chunk))
    template_chunk['choices'][0]['message']['toploc_fingerprint'] = toploc_fingerprint
    return template_chunk



def main():
    parser = argparse.ArgumentParser(description="SGLang Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, required = True, help="Port to bind to")
    parser.add_argument("--sglang-server", default="http://localhost:30000", 
                      help="Address of the SGLang server to proxy to")
    
    args = parser.parse_args()
    
    # Set the global SGLang server URL
    global SGLANG_SERVER_URL
    SGLANG_SERVER_URL = args.sglang_server
    
    print(f"Starting proxy server on {args.host}:{args.port}")
    print(f"Proxying to SGLang server at {SGLANG_SERVER_URL}")
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main() 