#!/usr/bin/env python3
"""
简单直接的SGLang启动调试
"""

import subprocess
import sys
import os

def test_direct_scenarios():
    """直接测试几种关键场景"""
    
    print("=" * 60)
    print("直接场景测试")
    print("=" * 60)
    
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B"
    
    scenarios = [
        {
            "name": "1. 卸载TorchAO后测试",
            "prep_cmd": ["pip3", "uninstall", "torchao", "-y"],
            "test_cmd": [sys.executable, "-m", "sglang.launch_server", "--help"]
        },
        {
            "name": "2. 设置禁用环境变量测试",
            "prep_cmd": None,
            "test_cmd": [sys.executable, "-m", "sglang.launch_server", "--help"],
            "env": {"SGLANG_DISABLE_TORCHAO": "1"}
        },
        {
            "name": "3. 最小参数测试",
            "prep_cmd": None,
            "test_cmd": [
                sys.executable, "-m", "sglang.launch_server",
                "--model-path", model_path,
                "--host", "localhost",
                "--port", "12345", 
                "--dtype", "bfloat16",
                "--help"
            ]
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        
        # 执行准备命令
        if scenario.get('prep_cmd'):
            print("   执行准备命令...")
            try:
                prep_result = subprocess.run(
                    scenario['prep_cmd'],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                if prep_result.returncode == 0:
                    print("   ✓ 准备命令成功")
                else:
                    print("   ? 准备命令可能失败，继续测试...")
            except Exception as e:
                print(f"   ? 准备命令异常: {e}")
        
        # 执行测试命令
        print("   执行测试命令...")
        try:
            env = os.environ.copy()
            if scenario.get('env'):
                env.update(scenario['env'])
                
            result = subprocess.run(
                scenario['test_cmd'],
                capture_output=True,
                text=True,
                timeout=30,
                env=env
            )
            
            if result.returncode == 0:
                print("   ✓ 测试成功 - 没有段错误!")
                return scenario['name']
            elif result.returncode == -11:
                print("   ✗ 仍然段错误")
            else:
                print(f"   ? 其他错误 (退出码: {result.returncode})")
                if result.stderr:
                    error_lines = result.stderr.strip().split('\n')[:2]
                    for line in error_lines:
                        print(f"     {line}")
                        
        except subprocess.TimeoutExpired:
            print("   ✗ 超时")
        except Exception as e:
            print(f"   ✗ 执行异常: {e}")
    
    return None

def check_versions():
    """检查当前版本状态"""
    
    print("\n" + "=" * 60)
    print("当前版本状态")
    print("=" * 60)
    
    version_checks = [
        ("PyTorch", "import torch; print(torch.__version__)"),
        ("SGLang", "import sglang; print(sglang.__version__)"),
        ("TorchAO", "import torchao; print(torchao.__version__)"),
        ("sgl-kernel", "import sgl_kernel; print(sgl_kernel.__version__)"),
        ("Flash Attention", "import flash_attn; print(flash_attn.__version__)"),
    ]
    
    for name, check_code in version_checks:
        try:
            result = subprocess.run(
                [sys.executable, "-c", check_code],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"   {name}: {version}")
            else:
                print(f"   {name}: 未安装或导入失败")
        except Exception:
            print(f"   {name}: 检查失败")

def quick_fix_suggestions():
    """提供快速修复建议"""
    
    print("\n" + "=" * 60)
    print("快速修复建议")
    print("=" * 60)
    
    suggestions = [
        "如果场景1成功（卸载TorchAO后）:",
        "  → 问题确实是TorchAO引起的，保持TorchAO卸载状态",
        "",
        "如果场景2成功（环境变量禁用）:",
        "  → 在启动脚本中添加: export SGLANG_DISABLE_TORCHAO=1",
        "",
        "如果所有场景都失败:",
        "  → 尝试降级PyTorch: pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124",
        "",
        "临时绕过方案:",
        "  → 使用其他推理框架如vLLM或HuggingFace Transformers"
    ]
    
    for suggestion in suggestions:
        print(suggestion)

if __name__ == "__main__":
    check_versions()
    success_scenario = test_direct_scenarios()
    quick_fix_suggestions()
    
    if success_scenario:
        print(f"\n🎉 找到工作方案: {success_scenario}")
        print("请使用这个方案继续您的训练!")
    else:
        print(f"\n😞 所有测试场景都失败了")
        print("建议降级PyTorch或使用替代方案")
