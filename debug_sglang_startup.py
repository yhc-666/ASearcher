#!/usr/bin/env python3
"""
调试SGLang启动问题 - 捕获详细错误信息
运行方式: python debug_sglang_startup.py
"""

import sys
import traceback
import subprocess
import os

print("=" * 80)
print("SGLang 启动调试工具")
print("=" * 80)

# 1. 检查环境信息
print("\n1. 环境信息:")
try:
    import torch
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
except Exception as e:
    print(f"   PyTorch检查失败: {e}")

try:
    import sglang
    print(f"   SGLang: {sglang.__version__}")
except Exception as e:
    print(f"   SGLang检查失败: {e}")

try:
    import torchao
    print(f"   TorchAO: {torchao.__version__}")
except Exception as e:
    print(f"   TorchAO检查失败: {e}")

# 2. 测试SGLang导入
print("\n2. 测试SGLang关键模块导入:")
sglang_modules = [
    "sglang.srt.server",
    "sglang.srt.model_executor.model_runner", 
    "sglang.srt.layers.torchao_utils",
    "sglang.launch_server",
]

for module in sglang_modules:
    try:
        __import__(module)
        print(f"   ✓ {module} - 导入成功")
    except Exception as e:
        print(f"   ✗ {module} - 导入失败: {e}")
        if "torchao" in module:
            print(f"      详细错误:")
            traceback.print_exc()

# 3. 测试TorchAO导入
print("\n3. 测试TorchAO关键导入:")
torchao_imports = [
    "torchao.quantization.int4_weight_only",
    "torchao.quantization.int8_weight_only", 
    "torchao.quantization.int8_dynamic_activation_int8_weight",
]

for imp in torchao_imports:
    try:
        exec(f"from {imp.rsplit('.', 1)[0]} import {imp.split('.')[-1]}")
        print(f"   ✓ {imp} - 导入成功")
    except Exception as e:
        print(f"   ✗ {imp} - 导入失败: {e}")

# 4. 测试最小SGLang服务器启动
print("\n4. 测试最小SGLang服务器启动:")
print("   尝试启动SGLang服务器（仅测试导入，不实际启动）...")

# 构造最小启动命令但不执行
model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B"
test_cmd = [
    sys.executable, "-m", "sglang.launch_server",
    "--help"  # 只显示帮助，不实际启动
]

try:
    result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print("   ✓ SGLang命令行接口正常")
    else:
        print(f"   ✗ SGLang命令行接口异常:")
        print(f"      stdout: {result.stdout}")
        print(f"      stderr: {result.stderr}")
except Exception as e:
    print(f"   ✗ SGLang命令行测试失败: {e}")

# 5. 模拟实际启动参数（但不启动）
print("\n5. 验证启动参数:")
startup_args = [
    "--host", "localhost",
    "--port", "12345", 
    "--tokenizer-path", model_path,
    "--model-path", model_path,
    "--dtype", "bfloat16",
    "--tp-size", "1",
    "--context-length", "32768",
    "--mem-fraction-static", "0.9",
    "--attention-backend", "fa3",
    "--dry-run"  # 如果支持的话
]

print(f"   启动参数: {' '.join(startup_args)}")

# 6. 检查模型路径
print("\n6. 检查模型路径:")
if os.path.exists(model_path):
    print(f"   ✓ 模型路径存在: {model_path}")
    # 检查关键文件
    key_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    for file in key_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            print(f"   ✓ {file} 存在")
        else:
            print(f"   ✗ {file} 不存在")
else:
    print(f"   ✗ 模型路径不存在: {model_path}")

print("\n" + "=" * 80)
print("建议调试步骤:")
print("=" * 80)
print("""
1. 如果TorchAO导入失败，尝试：
   pip uninstall torchao -y
   
2. 如果SGLang模块导入失败，检查版本兼容性：
   pip list | grep -E "(sglang|torchao|torch)"
   
3. 如果需要降级TorchAO，尝试：
   pip install torchao==0.6.1
   
4. 运行实际的SGLang启动命令并查看完整错误:
   python -m sglang.launch_server [你的参数] --log-level debug
""")
