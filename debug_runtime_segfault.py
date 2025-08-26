#!/usr/bin/env python3
"""
调试运行时段错误 - 逐步模拟SGLang启动过程
"""

import sys
import os
import traceback
import torch

def test_step_by_step_initialization():
    """逐步测试SGLang初始化过程"""
    
    print("=" * 60)
    print("逐步测试SGLang初始化")
    print("=" * 60)
    
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B"
    
    steps = [
        {
            "name": "1. 导入基础模块",
            "code": """
import sglang
import sglang.srt.model_executor.model_runner
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
print("基础模块导入成功")
"""
        },
        {
            "name": "2. 创建启动参数",
            "code": f"""
from sglang.srt.server_args import ServerArgs
args = ServerArgs(
    model_path="{model_path}",
    tokenizer_path="{model_path}",
    host="localhost",
    port=12345,
    dtype="bfloat16",
    tp_size=1,
    context_length=4096,
    mem_fraction_static=0.5,
    attention_backend="fa3",
)
print("启动参数创建成功:", type(args))
"""
        },
        {
            "name": "3. 测试TorchAO配置应用",
            "code": """
# 模拟apply_torchao_config调用
dummy_model = torch.nn.Linear(10, 10)
try:
    apply_torchao_config_to_model(dummy_model, "")  # 空配置
    print("TorchAO配置应用成功")
except Exception as e:
    print(f"TorchAO配置应用失败: {e}")
    if "torchao" in str(e).lower():
        raise e
"""
        },
        {
            "name": "4. 测试模型加载器创建",
            "code": f"""
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import ServerArgs

args = ServerArgs(
    model_path="{model_path}",
    tokenizer_path="{model_path}",
    dtype="bfloat16",
    tp_size=1,
    context_length=4096,
    mem_fraction_static=0.5,
)

# 只创建ModelRunner对象，不加载模型
try:
    model_runner = ModelRunner(
        model_config=None,  # 先用None测试
        mem_fraction_static=0.5,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        nccl_port=None,
        server_args=args,
    )
    print("ModelRunner创建成功")
except Exception as e:
    print(f"ModelRunner创建失败: {e}")
    raise e
"""
        },
        {
            "name": "5. 测试CUDA内存分配",
            "code": """
import torch
print(f"GPU可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

# 测试基本CUDA操作
try:
    x = torch.randn(100, 100).cuda()
    y = torch.matmul(x, x.t())
    print("基本CUDA操作成功")
except Exception as e:
    print(f"CUDA操作失败: {e}")
    raise e
"""
        },
        {
            "name": "6. 测试sgl_kernel函数调用",
            "code": """
import sgl_kernel
# 测试sgl_kernel的一些基本函数
try:
    # 检查sgl_kernel是否有常用的操作函数
    attrs = [attr for attr in dir(sgl_kernel) if not attr.startswith('_')]
    print(f"sgl_kernel可用函数: {len(attrs)}个")
    
    # 尝试调用一个简单的函数（如果存在）
    if hasattr(sgl_kernel, 'common_ops'):
        print("sgl_kernel.common_ops可用")
        # 不实际调用，避免段错误
    else:
        print("sgl_kernel.common_ops不可用")
        
except Exception as e:
    print(f"sgl_kernel测试失败: {e}")
    raise e
"""
        }
    ]
    
    for step in steps:
        print(f"\n{step['name']}:")
        try:
            # 在子进程中执行，避免段错误影响主进程
            import subprocess
            result = subprocess.run([
                sys.executable, "-c", step['code']
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"   ✓ 成功")
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        print(f"     {line}")
            elif result.returncode == -11:
                print(f"   ✗ 段错误 - 问题出现在这一步!")
                return step['name']
            else:
                print(f"   ✗ 失败 (退出码: {result.returncode})")
                if result.stderr:
                    for line in result.stderr.strip().split('\n')[:3]:
                        print(f"     错误: {line}")
                        
        except subprocess.TimeoutExpired:
            print(f"   ✗ 超时")
        except Exception as e:
            print(f"   ✗ 执行异常: {e}")
    
    return None

def test_torchao_specific_issue():
    """专门测试TorchAO相关的问题"""
    
    print("\n" + "=" * 60)
    print("TorchAO 特定问题测试")
    print("=" * 60)
    
    # 测试有无TorchAO时的差异
    test_cases = [
        {
            "name": "禁用TorchAO时的启动",
            "env": {"SGLANG_DISABLE_TORCHAO": "1"},
            "args": ["--help"]
        },
        {
            "name": "启用TorchAO时的启动", 
            "env": {},
            "args": ["--help"]
        },
        {
            "name": "最小参数启动测试",
            "env": {},
            "args": [
                "--model-path", "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B",
                "--host", "localhost",
                "--port", "12345",
                "--dtype", "bfloat16", 
                "--dry-run"  # 如果支持的话
            ]
        }
    ]
    
    for case in test_cases:
        print(f"\n测试: {case['name']}")
        
        cmd = [sys.executable, "-m", "sglang.launch_server"] + case['args']
        env = os.environ.copy()
        env.update(case.get('env', {}))
        
        try:
            import subprocess
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=20,
                env=env
            )
            
            if result.returncode == 0:
                print("   ✓ 成功")
            elif result.returncode == -11:
                print("   ✗ 段错误")
            else:
                print(f"   ? 退出码: {result.returncode}")
                
            if result.stderr and "torchao" in result.stderr.lower():
                print("   >> 发现TorchAO相关错误:")
                for line in result.stderr.split('\n')[:3]:
                    if line.strip():
                        print(f"      {line}")
                        
        except Exception as e:
            print(f"   执行异常: {e}")

def suggest_next_steps():
    """建议下一步调试方案"""
    
    print("\n" + "=" * 60)
    print("下一步调试建议")
    print("=" * 60)
    
    suggestions = [
        "1. 如果段错误出现在模型加载阶段，尝试:",
        "   - 使用更小的测试模型",
        "   - 减小context_length和mem_fraction_static",
        "",
        "2. 如果段错误出现在TorchAO配置阶段，尝试:",
        "   - 完全卸载TorchAO: pip uninstall torchao -y",
        "   - 设置环境变量: export SGLANG_DISABLE_TORCHAO=1",
        "",
        "3. 如果段错误出现在sgl_kernel调用阶段，尝试:",
        "   - 降级PyTorch: pip install torch==2.4.0",
        "   - 或者暂时禁用sgl_kernel优化",
        "",
        "4. 最终方案 - 使用已知稳定的版本组合:",
        "   - PyTorch 2.4.0 + SGLang 0.4.8 + sgl-kernel 0.2.6"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

if __name__ == "__main__":
    failed_step = test_step_by_step_initialization()
    test_torchao_specific_issue()
    suggest_next_steps()
    
    if failed_step:
        print(f"\n🎯 关键发现: 问题出现在 '{failed_step}'")
    else:
        print(f"\n🤔 奇怪: 所有步骤都成功了，但实际启动失败")
        print("这可能是多线程或GPU初始化相关的问题")
