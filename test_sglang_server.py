#!/usr/bin/env python3
"""
Test script to debug SGLang server startup issues.
Run this on your server to test if SGLang can start with your model.
"""

import subprocess
import time
import requests
import sys

def test_sglang_server():
    """Test if SGLang server can start successfully."""
    
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B"
    port = 12345
    
    print(f"Testing SGLang server startup with model: {model_path}")
    print(f"Using port: {port}")
    print("-" * 60)
    
    # Start SGLang server with minimal configuration
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tokenizer-path", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--device", "cuda",
        "--dtype", "bfloat16",
        "--context-length", "8192",  # Reduced from 32768
        "--mem-fraction-static", "0.8",  # Reduced from 0.9
        "--log-level", "info",  # Changed from warning to info for more details
        "--trust-remote-code",
    ]
    
    print("Starting server with command:")
    print(" ".join(cmd))
    print("-" * 60)
    
    # Start the server process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor output for 60 seconds
    start_time = time.time()
    timeout = 60
    server_ready = False
    
    print("Server output:")
    print("-" * 60)
    
    while time.time() - start_time < timeout:
        # Read output
        line = process.stdout.readline()
        if line:
            print(line.strip())
            
            # Check for success indicators
            if "Server started" in line or "Uvicorn running" in line:
                server_ready = True
                print("\n✅ Server started successfully!")
                break
            
            # Check for error indicators
            if "CUDA out of memory" in line:
                print("\n❌ ERROR: Out of GPU memory!")
                break
            if "Error" in line or "Failed" in line:
                print(f"\n❌ ERROR detected: {line}")
        
        # Check if process has terminated
        if process.poll() is not None:
            print(f"\n❌ Process terminated with code: {process.returncode}")
            break
        
        # Try to check health endpoint
        if time.time() - start_time > 10:  # Wait 10 seconds before checking
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=1)
                if response.status_code == 200:
                    server_ready = True
                    print("\n✅ Health check passed! Server is ready.")
                    break
            except:
                pass
        
        time.sleep(0.1)
    
    # Clean up
    if process.poll() is None:
        print("\nTerminating server process...")
        process.terminate()
        time.sleep(2)
        if process.poll() is None:
            process.kill()
    
    print("-" * 60)
    if server_ready:
        print("✅ SGLang server can start successfully!")
        print("\nSuggestions:")
        print("1. The server works with reduced settings")
        print("2. Try reducing context_length in your config from 32768 to 8192")
        print("3. Try reducing mem_fraction_static from 0.9 to 0.8")
    else:
        print("❌ SGLang server failed to start within timeout")
        print("\nPossible issues:")
        print("1. Not enough GPU memory - try reducing batch size or context length")
        print("2. Model loading issues - check if model files are accessible")
        print("3. CUDA/driver issues - check nvidia-smi output")
        print("4. Port conflicts - check if ports are already in use")
    
    return server_ready

if __name__ == "__main__":
    success = test_sglang_server()
    sys.exit(0 if success else 1)