#!/usr/bin/env python3
"""
诊断SGLang段错误问题
"""

import subprocess
import sys
import os

def check_kernel_compatibility():
    """检查sgl-kernel与CUDA/PyTorch的兼容性"""
    
    print("=" * 60)
    print("检查内核扩展兼容性")
    print("=" * 60)
    
    # 1. 检查sgl-kernel版本
    print("\n1. sgl-kernel版本:")
    try:
        result = subprocess.run([sys.executable, "-c", "import sgl_kernel; print(sgl_kernel.__version__)"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✓ sgl-kernel: {result.stdout.strip()}")
        else:
            print(f"   ✗ sgl-kernel导入失败: {result.stderr}")
    except Exception as e:
        print(f"   ✗ 检查失败: {e}")
    
    # 2. 检查CUDA扩展
    print("\n2. CUDA扩展测试:")
    cuda_tests = [
        "import torch; print('CUDA compiled:', torch.version.cuda)",
        "import torch; print('CUDA available:', torch.cuda.is_available())",
        "import torch; print('CUDA arch list:', torch.cuda.get_arch_list() if torch.cuda.is_available() else 'N/A')",
    ]
    
    for test in cuda_tests:
        try:
            result = subprocess.run([sys.executable, "-c", test], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"   ✓ {result.stdout.strip()}")
            else:
                print(f"   ✗ 测试失败: {result.stderr}")
        except Exception as e:
            print(f"   ✗ 执行失败: {e}")
    
    # 3. 检查Flash Attention
    print("\n3. Flash Attention测试:")
    fa_tests = [
        "import flash_attn; print('Flash Attention version:', flash_attn.__version__)",
        "from flash_attn import flash_attn_func; print('Flash Attention func available')",
    ]
    
    for test in fa_tests:
        try:
            result = subprocess.run([sys.executable, "-c", test], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"   ✓ {result.stdout.strip()}")
            else:
                print(f"   ✗ {result.stderr.strip()}")
        except Exception as e:
            print(f"   ✗ 执行失败: {e}")

def test_minimal_sglang():
    """测试最小化SGLang配置"""
    
    print("\n" + "=" * 60)
    print("测试最小化SGLang配置")
    print("=" * 60)
    
    # 测试不同的后端配置
    minimal_configs = [
        {
            "name": "禁用Flash Attention",
            "args": ["--attention-backend", "triton"]
        },
        {
            "name": "减小内存占用", 
            "args": ["--mem-fraction-static", "0.3"]
        },
        {
            "name": "最小上下文长度",
            "args": ["--context-length", "1024"]
        },
        {
            "name": "禁用编译优化",
            "args": ["--disable-flashinfer", "--disable-triton-attention"]
        }
    ]
    
    base_cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--host", "localhost",
        "--port", "12346",  # 避免端口冲突
        "--model-path", "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B",
        "--dtype", "bfloat16",
        "--tp-size", "1",
        "--help"  # 只测试参数解析，不实际启动
    ]
    
    for config in minimal_configs:
        print(f"\n测试配置: {config['name']}")
        cmd = base_cmd[:-1] + config['args'] + ["--help"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                print(f"   ✓ 参数解析成功")
            elif result.returncode == -11:
                print(f"   ✗ 段错误 (仍然存在)")
            else:
                print(f"   ✗ 退出码: {result.returncode}")
                if result.stderr:
                    print(f"   错误: {result.stderr[:200]}")
        except Exception as e:
            print(f"   ✗ 执行异常: {e}")

def check_environment_issues():
    """检查环境相关问题"""
    
    print("\n" + "=" * 60)
    print("环境问题检查")
    print("=" * 60)
    
    # 1. 检查共享库
    print("\n1. 关键共享库检查:")
    libs_to_check = [
        "libcuda.so.1",
        "libcudart.so.12", 
        "libcublas.so.12",
        "libcurand.so.10",
    ]
    
    for lib in libs_to_check:
        try:
            result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
            if lib in result.stdout:
                print(f"   ✓ {lib} 可用")
            else:
                print(f"   ✗ {lib} 未找到")
        except Exception as e:
            print(f"   ? 无法检查 {lib}: {e}")
    
    # 2. 检查Python扩展编译
    print("\n2. Python扩展编译检查:")
    try:
        result = subprocess.run([sys.executable, "-c", 
            "import torch; print('PyTorch CUDA compiled with:', torch.version.cuda); "
            "print('Current CUDA runtime:', torch.cuda.get_device_properties(0).major if torch.cuda.is_available() else 'N/A')"],
            capture_output=True, text=True)
        print(f"   {result.stdout}")
        if result.stderr:
            print(f"   警告: {result.stderr}")
    except Exception as e:
        print(f"   ✗ 检查失败: {e}")
    
    # 3. 内存检查
    print("\n3. GPU内存状态:")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                free, total = line.split(', ')
                print(f"   GPU {i}: {free}MB free / {total}MB total")
        else:
            print(f"   ✗ nvidia-smi失败: {result.stderr}")
    except Exception as e:
        print(f"   ✗ 检查失败: {e}")

def suggest_solutions():
    """提供解决方案建议"""
    
    print("\n" + "=" * 60)
    print("解决方案建议")
    print("=" * 60)
    
    solutions = [
        {
            "问题": "sgl-kernel 与 CUDA 12.4 不兼容",
            "解决方案": [
                "pip uninstall sgl-kernel -y",
                "pip install sgl-kernel --no-cache-dir --force-reinstall",
                "或者尝试: pip install sgl-kernel==0.2.6"
            ]
        },
        {
            "问题": "Flash Attention 兼容性问题", 
            "解决方案": [
                "pip uninstall flash-attn -y",
                "pip install flash-attn --no-build-isolation --no-cache-dir",
                "或在启动时使用: --attention-backend triton"
            ]
        },
        {
            "问题": "PyTorch 2.6.0 与扩展不兼容",
            "解决方案": [
                "降级PyTorch: pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124",
                "或使用SGLang推荐的PyTorch版本"
            ]
        },
        {
            "问题": "CUDA 架构不匹配",
            "解决方案": [
                "设置环境变量: export TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0'",
                "重新编译所有CUDA扩展"
            ]
        }
    ]
    
    for i, solution in enumerate(solutions, 1):
        print(f"\n{i}. {solution['问题']}:")
        for step in solution['解决方案']:
            print(f"   - {step}")

if __name__ == "__main__":
    check_kernel_compatibility()
    test_minimal_sglang()
    check_environment_issues()
    suggest_solutions()
