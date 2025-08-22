#!/usr/bin/env python3
"""
Debug SGLang import and initialization issues
"""

import sys
import traceback

print("=" * 60)
print("DEBUGGING SGLANG INITIALIZATION")
print("=" * 60)

# 1. Test basic imports
print("\n1. Testing basic imports...")
try:
    import torch
    print(f"   ✓ PyTorch imported successfully (version: {torch.__version__})")
except Exception as e:
    print(f"   ✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"   ✓ Transformers imported successfully (version: {transformers.__version__})")
except Exception as e:
    print(f"   ✗ Transformers import failed: {e}")

# 2. Test SGLang import
print("\n2. Testing SGLang import...")
try:
    import sglang
    print(f"   ✓ SGLang imported successfully")
    if hasattr(sglang, '__version__'):
        print(f"   Version: {sglang.__version__}")
except Exception as e:
    print(f"   ✗ SGLang import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# 3. Test SGLang server components
print("\n3. Testing SGLang server components...")
try:
    from sglang.srt.server import Server
    print("   ✓ SGLang Server class imported")
except Exception as e:
    print(f"   ✗ Server import failed: {e}")
    traceback.print_exc()

# 4. Test model loading
print("\n4. Testing model loading capabilities...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B"
    
    print(f"   Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        use_fast=True
    )
    print("   ✓ Tokenizer loaded successfully")
    
    # Don't actually load the model (too heavy for debug)
    print("   Checking model config...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"   ✓ Model config loaded: {config.model_type}")
    
except Exception as e:
    print(f"   ✗ Model loading test failed: {e}")
    traceback.print_exc()

# 5. Test SGLang server launch (without actually starting)
print("\n5. Testing SGLang server initialization...")
try:
    from sglang.srt.server_args import ServerArgs
    args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        host="0.0.0.0",
        port=12345,
        device="cuda",
        dtype="bfloat16",
        context_length=8192,
        mem_fraction_static=0.8,
        trust_remote_code=True,
    )
    print("   ✓ ServerArgs created successfully")
    print(f"   Args: {args}")
except Exception as e:
    print(f"   ✗ ServerArgs creation failed: {e}")
    traceback.print_exc()

# 6. Check for version conflicts
print("\n6. Checking for known issues...")
try:
    import vllm
    print(f"   ⚠ vLLM is installed (version: {vllm.__version__})")
    print("     This might conflict with SGLang")
except ImportError:
    print("   ✓ No vLLM conflict detected")

try:
    import flash_attn
    print(f"   ✓ FlashAttention installed (version: {flash_attn.__version__})")
except ImportError:
    print("   ⚠ FlashAttention not found - SGLang might fall back to slower attention")

# 7. Test CUDA operations
print("\n7. Testing CUDA operations...")
if torch.cuda.is_available():
    try:
        device = torch.device("cuda:0")
        test_tensor = torch.ones(1000, 1000, device=device)
        result = torch.matmul(test_tensor, test_tensor)
        print(f"   ✓ CUDA operations working (result shape: {result.shape})")
        del test_tensor, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"   ✗ CUDA operation failed: {e}")
else:
    print("   ✗ CUDA not available")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)

print("\nRECOMMENDATIONS:")
print("1. If SGLang import fails, try: pip install --upgrade sglang")
print("2. If model loading fails, check file permissions and paths")
print("3. If CUDA operations fail, check driver compatibility")
print("4. Check SGLang GitHub issues for your specific error"