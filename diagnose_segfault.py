#!/usr/bin/env python3
"""
Diagnose segmentation fault in SGLang
"""

import sys
import subprocess

print("=" * 60)
print("DIAGNOSING SEGMENTATION FAULT")
print("=" * 60)

# 1. Check Python version
print("\n1. Python version:")
print(f"   {sys.version}")

# 2. Check PyTorch and CUDA
print("\n2. PyTorch and CUDA versions:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version (PyTorch built with): {torch.version.cuda}")
        print(f"   cuDNN version: {torch.backends.cudnn.version()}")
        print(f"   Number of GPUs: {torch.cuda.device_count()}")
except Exception as e:
    print(f"   Error checking PyTorch: {e}")

# 3. Check system CUDA version
print("\n3. System CUDA version:")
try:
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"   {line.strip()}")
except:
    print("   nvcc not found in PATH")

# 4. Check SGLang version
print("\n4. SGLang version:")
try:
    import sglang
    print(f"   SGLang version: {sglang.__version__}")
except Exception as e:
    print(f"   Error importing SGLang: {e}")

# 5. Check for conflicting libraries
print("\n5. Checking LD_LIBRARY_PATH:")
import os
ld_path = os.environ.get('LD_LIBRARY_PATH', 'Not set')
if ld_path != 'Not set':
    paths = ld_path.split(':')
    for path in paths[:5]:  # Show first 5 paths
        print(f"   - {path}")
else:
    print(f"   {ld_path}")

# 6. Test minimal torch operations
print("\n6. Testing PyTorch CUDA operations:")
try:
    import torch
    if torch.cuda.is_available():
        # Try to create a tensor on GPU
        x = torch.randn(10, 10).cuda()
        print("   ✓ Can create CUDA tensors")
        
        # Try a simple operation
        y = x @ x
        print("   ✓ Can perform CUDA operations")
        
        # Try to import flash attention (often causes segfaults)
        try:
            import flash_attn
            print("   ✓ flash_attn imports successfully")
        except ImportError:
            print("   ✗ flash_attn not installed or import failed")
    else:
        print("   CUDA not available for testing")
except Exception as e:
    print(f"   ✗ Error during CUDA operations: {e}")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)
print("""
Common causes of segmentation faults in SGLang:

1. **CUDA version mismatch**: PyTorch compiled for different CUDA than system
2. **Flash Attention issues**: Incompatible flash-attn version
3. **Library conflicts**: Multiple CUDA installations or LD_LIBRARY_PATH issues
4. **SGLang compilation**: SGLang C++ extensions compiled for different environment

Recommended fixes:
1. Reinstall SGLang: pip uninstall sglang && pip install sglang
2. Check CUDA compatibility: Ensure PyTorch CUDA version matches system CUDA
3. Try without flash attention: Set --attention-backend triton instead of fa3
4. Clear LD_LIBRARY_PATH: unset LD_LIBRARY_PATH (temporarily)
""")