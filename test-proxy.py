import requests
import json
import sys
import time

def test_chat_completion():
    # Proxy server endpoint
    url = "http://localhost:4001/v1/chat/completions"
    
    # OpenAI-style chat completion request
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Send request to proxy server
        response = requests.post(url, json=payload, headers=headers)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        
        # Print the response for inspection
        print("\nResponse from proxy server:")
        print(json.dumps(result, indent=2))
        
        # Verify response structure
        assert "choices" in result, "Response missing 'choices' field"
        assert len(result["choices"]) > 0, "No choices in response"
        assert "message" in result["choices"][0], "Choice missing 'message' field"
        assert "content" in result["choices"][0]["message"], "Message missing 'content' field"
        
        print("\nTest passed! Response structure is valid.")
        print("Generated response:", result["choices"][0]["message"]["content"])
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError making request: {e}")
        return False
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

def test_streaming_chat_completion():
    # Proxy server endpoint
    url = "http://localhost:4001/v1/chat/completions"
    
    # OpenAI-style chat completion request with streaming
    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": True
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }
    
    try:
        # Send request to proxy server with stream=True to get streaming response
        response = requests.post(url, json=payload, headers=headers, stream=True)
        
        # Check if request was successful
        response.raise_for_status()
        
        print("\nStreaming response from proxy server:")
        
        # Process the streaming response
        full_content = ""
        for line in response.iter_lines():
            # Skip empty lines
            if not line:
                continue
                
            # Remove 'data: ' prefix and parse JSON
            line = line.decode('utf-8')
            
            if line.startswith('data: '):
                line = line[6:]  # Remove 'data: ' prefix
                
                # Check for [DONE] marker which indicates end of stream
                if line == "[DONE]":
                    print("\n[DONE] Stream completed")
                    break
                    
                try:
                    chunk = json.loads(line)
                    
                    # Extract and display content delta if present
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        choice = chunk["choices"][0]
                        if "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                            content_chunk = choice["delta"]["content"]
                            print(content_chunk, end='', flush=True)
                            full_content += content_chunk
                except json.JSONDecodeError:
                    print(f"Could not parse JSON: {line}")
        
        print("\n\nTest passed! Streaming response was received successfully.")
        print("Full assembled content:", full_content)
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"\nError making streaming request: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error in streaming test: {e}")
        return False

if __name__ == "__main__":
    print("Testing proxy server chat completion...")
    success = test_chat_completion()
    
    if success:
        print("\n\nTesting streaming chat completion...")
        success = test_streaming_chat_completion()
    
    sys.exit(0 if success else 1) 