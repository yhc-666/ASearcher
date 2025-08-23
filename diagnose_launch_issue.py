#!/usr/bin/env python3
"""
Diagnose why the training launcher is failing quickly
"""

import os
import subprocess
import socket

def check_ports(ports):
    """Check if ports are already in use."""
    print("Checking if ports are available...")
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('localhost', port))
        if result == 0:
            print(f"  ✗ Port {port} is already in use!")
        else:
            print(f"  ✓ Port {port} is available")
        sock.close()

def check_model_path():
    """Check if model path exists and is accessible."""
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B"
    print(f"\nChecking model path: {model_path}")
    
    if os.path.exists(model_path):
        print("  ✓ Model path exists")
        
        # Check for key model files
        expected_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        for file in expected_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                print(f"  ✓ Found {file}")
            else:
                print(f"  ✗ Missing {file}")
        
        # Check for model weights
        import glob
        safetensors = glob.glob(os.path.join(model_path, "*.safetensors"))
        if safetensors:
            print(f"  ✓ Found {len(safetensors)} safetensors files")
        else:
            print("  ✗ No safetensors files found!")
    else:
        print("  ✗ Model path does not exist!")

def check_gpu_processes():
    """Check if there are any hanging GPU processes."""
    print("\nChecking for GPU processes...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if "No running processes found" in result.stdout:
            print("  ✓ No GPU processes running")
        else:
            print("  ⚠ Found running GPU processes:")
            # Parse and show processes
            lines = result.stdout.split('\n')
            for line in lines:
                if 'python' in line.lower() or 'sglang' in line.lower():
                    print(f"    {line.strip()}")
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")

def check_disk_space():
    """Check available disk space."""
    print("\nChecking disk space...")
    try:
        result = subprocess.run(['df', '-h', '/mnt/dolphinfs'], capture_output=True, text=True)
        print(result.stdout)
    except:
        pass

def check_python_imports():
    """Check if required Python packages are importable."""
    print("\nChecking Python imports...")
    packages = ['sglang', 'transformers', 'torch', 'ddgs']
    
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg} is importable")
        except ImportError as e:
            print(f"  ✗ {pkg} import failed: {e}")

def main():
    print("=" * 60)
    print("DIAGNOSING LAUNCH ISSUES")
    print("=" * 60)
    
    # Check ports that SGLang will use
    ports = [13051, 16406, 22483, 44367]
    check_ports(ports)
    
    check_model_path()
    check_gpu_processes()
    check_disk_space()
    check_python_imports()
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
    
    print("\nCommon solutions:")
    print("1. Kill hanging processes: pkill -f sglang")
    print("2. Clear ports: sudo fuser -k PORT/tcp")
    print("3. Check logs in experiments/logs/")
    print("4. Ensure model files are complete")

if __name__ == "__main__":
    main()