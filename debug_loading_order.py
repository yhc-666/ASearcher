#!/usr/bin/env python3
"""
分析SGLang模块加载顺序，解释为什么TorchAO影响启动
"""

import sys
import traceback

def test_module_loading_order():
    """测试不同模块的加载顺序"""
    
    print("=" * 60)
    print("SGLang 模块加载顺序分析")
    print("=" * 60)
    
    # 记录加载顺序
    loading_steps = [
        ("1. SGLang基础模块", "import sglang"),
        ("2. SGLang SRT", "import sglang.srt"),
        ("3. 模型执行器", "import sglang.srt.model_executor"),
        ("4. 模型运行器", "import sglang.srt.model_executor.model_runner"),
        ("5. TorchAO工具", "import sglang.srt.layers.torchao_utils"),
        ("6. sgl-kernel", "import sgl_kernel"),
        ("7. 启动服务器", "import sglang.launch_server"),
    ]
    
    for step, import_code in loading_steps:
        print(f"\n{step}:")
        try:
            exec(import_code)
            print(f"   ✓ 成功")
        except Exception as e:
            print(f"   ✗ 失败: {e}")
            if "sgl_kernel" in import_code or "torchao" in str(e):
                print(f"   >> 关键错误点!")
                traceback.print_exc()
                break

def analyze_sglang_initialization():
    """分析SGLang初始化过程"""
    
    print("\n" + "=" * 60)
    print("SGLang 初始化过程分析") 
    print("=" * 60)
    
    print("\n1. 检查TorchAO集成代码:")
    try:
        # 查看SGLang如何使用TorchAO
        from sglang.srt.layers import torchao_utils
        
        # 检查torchao_utils模块的函数
        attrs = [attr for attr in dir(torchao_utils) if not attr.startswith('_')]
        print(f"   torchao_utils 可用函数: {attrs}")
        
        # 查看是否有apply_torchao_config函数
        if hasattr(torchao_utils, 'apply_torchao_config'):
            print("   ✓ apply_torchao_config 存在")
        else:
            print("   ✗ apply_torchao_config 不存在")
            
    except Exception as e:
        print(f"   torchao_utils 导入失败: {e}")
    
    print("\n2. 检查model_runner中的TorchAO调用:")
    try:
        # 检查model_runner是否调用了torchao
        import sglang.srt.model_executor.model_runner as mr
        
        # 查看模块源码中是否包含torchao相关代码
        import inspect
        source = inspect.getsource(mr)
        
        torchao_lines = [line.strip() for line in source.split('\n') 
                        if 'torchao' in line.lower() and not line.strip().startswith('#')]
        
        if torchao_lines:
            print("   model_runner中的TorchAO相关代码:")
            for line in torchao_lines[:5]:  # 只显示前5行
                print(f"     {line}")
        else:
            print("   model_runner中未发现TorchAO相关代码")
            
    except Exception as e:
        print(f"   model_runner检查失败: {e}")

def test_torchao_dependency_chain():
    """测试TorchAO依赖链"""
    
    print("\n" + "=" * 60)
    print("TorchAO 依赖链分析")
    print("=" * 60)
    
    # 测试不同的导入组合
    test_cases = [
        {
            "name": "只导入SGLang基础",
            "imports": ["import sglang"]
        },
        {
            "name": "导入SGLang + TorchAO utils",
            "imports": ["import sglang", "import sglang.srt.layers.torchao_utils"]
        },
        {
            "name": "导入模型运行器",
            "imports": ["import sglang", "import sglang.srt.model_executor.model_runner"]
        },
        {
            "name": "导入sgl-kernel",
            "imports": ["import sgl_kernel"]
        },
        {
            "name": "完整导入链",
            "imports": [
                "import sglang",
                "import sglang.srt.layers.torchao_utils", 
                "import sglang.srt.model_executor.model_runner",
                "import sgl_kernel"
            ]
        }
    ]
    
    for case in test_cases:
        print(f"\n测试: {case['name']}")
        success = True
        for import_stmt in case['imports']:
            try:
                exec(import_stmt)
                print(f"   ✓ {import_stmt}")
            except Exception as e:
                print(f"   ✗ {import_stmt} - {e}")
                success = False
                break
        
        if success:
            print(f"   >> {case['name']} - 全部成功")
        else:
            print(f"   >> {case['name']} - 在此步骤失败")

def explain_torchao_effect():
    """解释TorchAO对启动的影响"""
    
    print("\n" + "=" * 60)
    print("为什么TorchAO影响SGLang启动")
    print("=" * 60)
    
    explanations = [
        {
            "场景": "没有TorchAO时",
            "过程": [
                "1. SGLang启动，尝试导入torchao_utils",
                "2. torchao_utils内部检测到没有TorchAO，跳过量化配置",
                "3. 继续加载sgl_kernel",
                "4. sgl_kernel ABI不兼容 -> 段错误",
                "5. 但错误可能出现在稍后的阶段"
            ]
        },
        {
            "场景": "有TorchAO时", 
            "过程": [
                "1. SGLang启动，导入torchao_utils",
                "2. torchao_utils调用TorchAO函数",
                "3. TorchAO可能触发了某些CUDA内核或C++扩展的早期加载",
                "4. 这些扩展与sgl_kernel的ABI冲突",
                "5. 更早地暴露了ABI不兼容问题 -> 段错误"
            ]
        }
    ]
    
    for exp in explanations:
        print(f"\n{exp['场景']}:")
        for step in exp['过程']:
            print(f"   {step}")
    
    print(f"\n结论:")
    print(f"   TorchAO不是根本原因，而是触发器")
    print(f"   真正原因: sgl-kernel 0.2.7 与 PyTorch 2.6.0 ABI不兼容")
    print(f"   TorchAO的存在改变了模块加载时机，更早暴露了ABI问题")

if __name__ == "__main__":
    test_module_loading_order()
    analyze_sglang_initialization()
    test_torchao_dependency_chain()
    explain_torchao_effect()
