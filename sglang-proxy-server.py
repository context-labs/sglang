#!/usr/bin/env python3
from fastapi import FastAPI, Request, Response, Depends
from fastapi.responses import StreamingResponse
import httpx
from typing import AsyncGenerator, Callable, Optional, Dict, List, Any
import json
import uvicorn
import argparse
import sys

app = FastAPI()

# Global variable to store SGLang server URL
SGLANG_SERVER_URL = "http://localhost:30000"

# Simple request/response logging to stdout
def log_message(message):
    print(f"\n{message}", file=sys.stdout, flush=True)


async def proxy_stream(response: httpx.Response, path: str) -> AsyncGenerator[bytes, None]:
    chunk_index, parsed_chunks = 0, []
    async for chunk in response.aiter_bytes():
        try:
            chunk_text = chunk.decode('utf-8').strip()
            parsed_chunk = maybe_parse_chunk(chunk_text)
            print(f"Parsed chunk: {parsed_chunk}")
            if parsed_chunk is not None:
                parsed_chunks.append(parsed_chunk)
        except Exception as e:
            log_message(f"Error logging stream chunk: {str(e)}")
        
        # if we're done, yield one more chunk with the fingerprint
        
        try:
            if chunk_text == "[DONE]":
                toploc_fingerprint = make_toploc_fingerprint_from_chunks(parsed_chunks)
                fingerprint_chunk = create_toploc_fingerprint_chunk(toploc_fingerprint)
                yield json.dumps(fingerprint_chunk).encode() + b"\n"
        except Exception as e:
            log_message(f"Error logging stream chunk: {str(e)}")

        yield chunk

def make_toploc_fingerprint_from_chunks(chunks: List[str]) -> str:
    return "ABCD123"

def create_toploc_fingerprint_chunk(toploc_fingerprint: str) -> Dict[str, Any]:
    return {
        "delta": {
            "content": toploc_fingerprint
        }
    }

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