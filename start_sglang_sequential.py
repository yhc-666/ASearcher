#!/usr/bin/env python3
"""
Start SGLang servers sequentially to avoid initialization conflicts on A100s
"""

import subprocess
import time
import sys
import os

def start_sglang_server(gpu_id, port, model_path):
    """Start a single SGLang server and wait for it to be ready."""
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    cmd = [
        'python3', '-m', 'sglang.launch_server',
        '--model-path', model_path,
        '--tokenizer-path', model_path,
        '--host', '0.0.0.0',
        '--port', str(port),
        '--device', 'cuda',
        '--dtype', 'bfloat16',
        '--context-length', '32768',  # Full 32K for A100s
        '--mem-fraction-static', '0.9',
        '--attention-backend', 'fa3',
        '--trust-remote-code',
        '--log-level', 'info',
    ]
    
    print(f"Starting SGLang server on GPU {gpu_id}, port {port}...")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    # Wait for server to start
    start_time = time.time()
    while time.time() - start_time < 60:
        line = process.stdout.readline()
        if line:
            print(f"  GPU{gpu_id}: {line.strip()}")
            if "Uvicorn running" in line or "Application startup complete" in line:
                print(f"✓ Server on GPU {gpu_id} is ready!")
                return process
    
    print(f"✗ Server on GPU {gpu_id} failed to start")
    process.terminate()
    return None

def main():
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B"
    
    servers = []
    ports = [13028, 23425, 28655, 36397]
    
    print("Starting SGLang servers sequentially...")
    print("=" * 60)
    
    for i, port in enumerate(ports):
        process = start_sglang_server(i, port, model_path)
        if process:
            servers.append(process)
            print(f"Waiting 5 seconds before starting next server...")
            time.sleep(5)
        else:
            print(f"Failed to start server {i}")
            # Clean up already started servers
            for p in servers:
                p.terminate()
            sys.exit(1)
    
    print("=" * 60)
    print(f"✓ All {len(servers)} SGLang servers started successfully!")
    print("Servers are running on ports:", ports)
    print("\nPress Ctrl+C to stop all servers...")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all servers...")
        for p in servers:
            p.terminate()
        print("Done!")

if __name__ == "__main__":
    main()