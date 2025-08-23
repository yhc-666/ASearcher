#!/usr/bin/env python3
"""
Check how SGLang decides to use torchao
"""

import os

# Path to the problematic file
torchao_utils_path = "/home/hadoop-ai-search/.local/lib/python3.10/site-packages/sglang/srt/layers/torchao_utils.py"
model_runner_path = "/home/hadoop-ai-search/.local/lib/python3.10/site-packages/sglang/srt/model_executor/model_runner.py"

print("=" * 60)
print("Checking SGLang's torchao usage logic")
print("=" * 60)

# Check torchao_utils.py
print("\n1. Checking torchao_utils.py:")
print(f"   Path: {torchao_utils_path}")
if os.path.exists(torchao_utils_path):
    with open(torchao_utils_path, 'r') as f:
        lines = f.readlines()
    
    print("\n   First 100 lines of torchao_utils.py:")
    for i, line in enumerate(lines[:100], 1):
        if 'def apply_torchao_config' in line:
            print(f"   Line {i}: {line.strip()}")
            # Show the function definition
            for j in range(i, min(i+20, len(lines))):
                print(f"   Line {j}: {lines[j-1].rstrip()}")
            break
else:
    print("   File not found!")

# Check model_runner.py around line 317
print("\n2. Checking model_runner.py around line 317:")
print(f"   Path: {model_runner_path}")
if os.path.exists(model_runner_path):
    with open(model_runner_path, 'r') as f:
        lines = f.readlines()
    
    # Show lines around 317
    start = max(0, 317 - 10)
    end = min(len(lines), 317 + 10)
    
    print(f"\n   Lines {start+1} to {end}:")
    for i in range(start, end):
        line_num = i + 1
        marker = ">>>" if line_num == 317 else "   "
        print(f"   {marker} Line {line_num}: {lines[i].rstrip()}")
else:
    print("   File not found!")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)
print("""
Look for:
1. How 'torchao_config' is checked (is it checking for empty string?)
2. Is there a way to skip the apply_torchao_config call?
3. What triggers the torchao import?
""")