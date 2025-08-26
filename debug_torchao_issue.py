#!/usr/bin/env python3
"""
专门调试TorchAO兼容性问题
"""

import sys
import traceback

def check_torchao_compatibility():
    """检查TorchAO兼容性"""
    
    print("=" * 60)
    print("TorchAO 兼容性诊断")
    print("=" * 60)
    
    # 1. 基本导入测试
    print("\n1. 测试TorchAO基本导入:")
    try:
        import torchao
        print(f"   ✓ TorchAO版本: {torchao.__version__}")
    except Exception as e:
        print(f"   ✗ TorchAO导入失败: {e}")
        return
    
    # 2. 测试quantization模块
    print("\n2. 测试quantization模块:")
    try:
        import torchao.quantization
        print("   ✓ torchao.quantization 导入成功")
        
        # 列出所有可用的函数
        attrs = dir(torchao.quantization)
        public_attrs = [attr for attr in attrs if not attr.startswith('_')]
        print(f"   可用函数/类: {len(public_attrs)}个")
        for attr in sorted(public_attrs):
            print(f"     - {attr}")
            
    except Exception as e:
        print(f"   ✗ quantization模块导入失败: {e}")
        traceback.print_exc()
        return
    
    # 3. 测试SGLang需要的具体导入
    print("\n3. 测试SGLang需要的具体函数:")
    required_functions = [
        "int4_weight_only",
        "int8_weight_only", 
        "int8_dynamic_activation_int8_weight",
    ]
    
    for func_name in required_functions:
        try:
            func = getattr(torchao.quantization, func_name, None)
            if func is not None:
                print(f"   ✓ {func_name} - 可用")
            else:
                print(f"   ✗ {func_name} - 不存在")
        except Exception as e:
            print(f"   ✗ {func_name} - 访问失败: {e}")
    
    # 4. 尝试模拟SGLang的导入方式
    print("\n4. 模拟SGLang的导入方式:")
    try:
        # 这是SGLang通常的导入方式
        exec_code = """
from torchao.quantization import (
    int4_weight_only,
    int8_weight_only, 
    int8_dynamic_activation_int8_weight,
)
"""
        exec(exec_code)
        print("   ✓ SGLang式导入成功")
    except Exception as e:
        print(f"   ✗ SGLang式导入失败: {e}")
        traceback.print_exc()
    
    # 5. 测试SGLang的torchao_utils模块
    print("\n5. 测试SGLang的torchao_utils模块:")
    try:
        from sglang.srt.layers import torchao_utils
        print("   ✓ SGLang torchao_utils 导入成功")
        
        # 测试apply_torchao_config函数
        if hasattr(torchao_utils, 'apply_torchao_config'):
            print("   ✓ apply_torchao_config 函数存在")
        else:
            print("   ✗ apply_torchao_config 函数不存在")
            
    except Exception as e:
        print(f"   ✗ SGLang torchao_utils 导入失败: {e}")
        traceback.print_exc()

def test_manual_sglang_import():
    """手动测试SGLang关键模块导入"""
    
    print("\n" + "=" * 60)
    print("SGLang 关键模块导入测试")
    print("=" * 60)
    
    modules_to_test = [
        "sglang",
        "sglang.srt",
        "sglang.srt.model_executor",
        "sglang.srt.model_executor.model_runner",
        "sglang.srt.layers",
        "sglang.srt.layers.torchao_utils",
        "sglang.launch_server",
    ]
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"   ✓ {module}")
        except Exception as e:
            print(f"   ✗ {module} - {e}")
            if "torchao" in str(e).lower():
                print(f"      >> TorchAO相关错误!")
                traceback.print_exc()

if __name__ == "__main__":
    check_torchao_compatibility()
    test_manual_sglang_import()
    
    print("\n" + "=" * 60)
    print("调试建议:")
    print("=" * 60)
    print("""
1. 如果TorchAO 0.9.0不兼容，尝试降级：
   pip uninstall torchao -y
   pip install torchao==0.6.1
   
2. 如果完全不需要量化，卸载TorchAO：
   pip uninstall torchao -y
   
3. 检查SGLang版本兼容性：
   pip show sglang
   
4. 查看SGLang的requirements：
   pip show sglang | grep Requires
""")
