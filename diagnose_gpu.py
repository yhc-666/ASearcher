#!/usr/bin/env python3
"""
Diagnostic script to check GPU resources and SGLang compatibility
"""

import torch
import subprocess
import os

def diagnose_gpu_environment():
    print("=" * 60)
    print("GPU ENVIRONMENT DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Check PyTorch CUDA availability
    print("\n1. PyTorch CUDA Status:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
        
        # Check each GPU
        print("\n2. GPU Details:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n   GPU {i}: {props.name}")
            print(f"   - Total memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"   - Memory allocated: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
            print(f"   - Memory reserved: {torch.cuda.memory_reserved(i) / 1024**3:.1f} GB")
            print(f"   - Compute capability: {props.major}.{props.minor}")
    
    # 2. Check nvidia-smi
    print("\n3. NVIDIA-SMI Output:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"   Error running nvidia-smi: {e}")
    
    # 3. Calculate memory requirements for 32K context
    print("\n4. Memory Requirements for Qwen3-8B with 32K context:")
    
    # Model parameters
    model_params = 8e9  # 8B parameters
    dtype_size = 2  # bfloat16
    model_memory = model_params * dtype_size / 1024**3
    print(f"   Model weights: {model_memory:.1f} GB")
    
    # KV cache per GPU (assuming split across GPUs)
    batch_size = 64
    context_length = 32768
    hidden_dim = 4096
    num_layers = 32
    num_heads = 32
    
    # For 4 SGLang servers (each on 1 GPU)
    batch_per_gpu = batch_size // 4
    kv_cache_per_gpu = (
        batch_per_gpu * context_length * hidden_dim * num_layers * 2 * dtype_size
    ) / 1024**3
    
    print(f"   KV cache per GPU (batch={batch_per_gpu}): {kv_cache_per_gpu:.1f} GB")
    print(f"   Total per GPU: {model_memory + kv_cache_per_gpu:.1f} GB")
    print(f"   Available per A100-80GB: 80 GB")
    print(f"   Should fit? {'YES ✓' if (model_memory + kv_cache_per_gpu) < 70 else 'NO ✗'}")
    
    # 4. Check environment variables
    print("\n5. Relevant Environment Variables:")
    env_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH', 
                'PYTORCH_CUDA_ALLOC_CONF', 'CUDA_LAUNCH_BLOCKING']
    for var in env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"   {var}: {value}")
    
    # 5. Test memory allocation
    print("\n6. Testing Large Memory Allocation:")
    try:
        device = torch.device('cuda:0')
        # Try to allocate 60GB on first GPU
        test_size_gb = 60
        test_tensor = torch.zeros(
            (int(test_size_gb * 1024**3 / 4),), 
            dtype=torch.float32, 
            device=device
        )
        print(f"   ✓ Successfully allocated {test_size_gb}GB on GPU 0")
        del test_tensor
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ Failed to allocate large tensor: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_gpu_environment()