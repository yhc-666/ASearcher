#!/usr/bin/env python3
"""
Minimal test to see if SGLang server can actually start
"""

import subprocess
import time
import os
import signal
import sys

def test_minimal_sglang():
    """Test SGLang with minimal configuration."""
    
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B"
    
    # Use minimal settings to reduce chance of failure
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tokenizer-path", model_path,
        "--host", "0.0.0.0",
        "--port", "30000",
        "--device", "cuda",
        "--dtype", "bfloat16",
        "--context-length", "2048",  # Very small context
        "--mem-fraction-static", "0.5",  # Only use half the memory
        "--trust-remote-code",
        "--log-level", "debug",  # Debug level for more info
        "--tp-size", "1",
        "--max-running-requests", "1",  # Minimal concurrency
    ]
    
    env = os.environ.copy()
    # Explicitly set CUDA device
    env['CUDA_VISIBLE_DEVICES'] = '0'
    # Disable any vLLM if it's interfering
    env['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    print("Starting SGLang with minimal configuration...")
    print("Command:", " ".join(cmd))
    print("-" * 60)
    
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        preexec_fn=os.setsid  # Create new process group
    )
    
    # Give it 30 seconds to start
    start_time = time.time()
    server_started = False
    
    try:
        while time.time() - start_time < 30:
            line = process.stdout.readline()
            if line:
                print(line.strip())
                
                # Check for successful start indicators
                if any(x in line for x in ["Uvicorn running", "Application startup complete", "Started server"]):
                    server_started = True
                    print("\n✅ SUCCESS: Server started!")
                    break
                
                # Check for common errors
                if "CUDA out of memory" in line:
                    print("\n❌ Out of memory error detected")
                    break
                if "RuntimeError" in line or "Error" in line:
                    print(f"\n⚠️ Error detected: {line}")
                
            # Check if process died
            if process.poll() is not None:
                print(f"\n❌ Process terminated with code: {process.returncode}")
                # Get any remaining output
                remaining = process.stdout.read()
                if remaining:
                    print("Remaining output:", remaining)
                break
                
            time.sleep(0.1)
    
    finally:
        # Clean up
        if process.poll() is None:
            print("\nTerminating server...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            time.sleep(2)
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    
    return server_started

if __name__ == "__main__":
    print("=" * 60)
    print("MINIMAL SGLANG SERVER TEST")
    print("=" * 60)
    
    success = test_minimal_sglang()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ SGLang CAN start with minimal settings")
        print("\nNext steps:")
        print("1. Gradually increase context_length")
        print("2. Increase mem_fraction_static")
        print("3. Add more concurrent requests")
    else:
        print("❌ SGLang failed to start even with minimal settings")
        print("\nTroubleshooting:")
        print("1. Check if vLLM is interfering: pip uninstall vllm")
        print("2. Reinstall SGLang: pip install --upgrade --force-reinstall sglang")
        print("3. Check GPU driver: nvidia-smi")
        print("4. Try with CPU: --device cpu")
    
    sys.exit(0 if success else 1)