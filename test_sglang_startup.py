#!/usr/bin/env python3
"""
测试SGLang服务器启动 - 捕获完整错误信息
"""

import subprocess
import sys
import time
import signal
import os

def test_sglang_startup():
    """测试SGLang启动并捕获错误"""
    
    print("=" * 60)
    print("SGLang 启动测试")
    print("=" * 60)
    
    # 设置模型路径（根据您的配置文件）
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtsearch-assistant/ai-search/deepsearch_files/LLMbasemodels/huggingface.co/Qwen/Qwen3-8B"
    
    # 最简单的启动命令
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--host", "localhost",
        "--port", "12345",  # 使用不同的端口避免冲突
        "--tokenizer-path", model_path,
        "--model-path", model_path,
        "--dtype", "bfloat16",
        "--tp-size", "1",
        "--context-length", "8192",  # 较小的上下文长度
        "--mem-fraction-static", "0.8",  # 较小的内存占用
        "--attention-backend", "fa3",
        "--log-level", "debug",  # 详细日志
        "--log-level-http", "debug",
    ]
    
    print("启动命令:")
    print(" ".join(cmd))
    print("\n" + "="*60)
    print("启动日志 (Ctrl+C 停止):")
    print("="*60)
    
    process = None
    try:
        # 启动进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并stderr到stdout
            universal_newlines=True,
            bufsize=1  # 行缓冲
        )
        
        # 实时输出日志
        startup_timeout = 120  # 2分钟超时
        start_time = time.time()
        
        while True:
            # 检查超时
            if time.time() - start_time > startup_timeout:
                print("\n[TIMEOUT] 启动超时，终止进程...")
                break
                
            # 读取输出
            line = process.stdout.readline()
            if line:
                print(f"[SGLang] {line.rstrip()}")
                
                # 检查启动成功标志
                if "Server is ready" in line or "Uvicorn running" in line:
                    print("\n[SUCCESS] SGLang服务器启动成功！")
                    time.sleep(2)  # 等待一下确保稳定
                    break
                    
                # 检查错误标志
                error_keywords = ["Error", "Exception", "Traceback", "ImportError", "ModuleNotFoundError"]
                if any(keyword in line for keyword in error_keywords):
                    print(f"\n[ERROR DETECTED] 发现错误: {line.rstrip()}")
                    
            # 检查进程是否结束
            if process.poll() is not None:
                print(f"\n[PROCESS ENDED] 进程退出，返回码: {process.returncode}")
                break
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 用户中断")
    except Exception as e:
        print(f"\n[EXCEPTION] 执行异常: {e}")
    finally:
        # 清理进程
        if process and process.poll() is None:
            print("\n[CLEANUP] 终止SGLang进程...")
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()
                process.wait()
        
        # 获取剩余输出
        if process:
            remaining_output = process.stdout.read()
            if remaining_output:
                print("\n[FINAL OUTPUT]")
                print(remaining_output)

if __name__ == "__main__":
    test_sglang_startup()
