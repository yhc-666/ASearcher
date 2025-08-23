#!/usr/bin/env python3
"""
Check torchao version compatibility with SGLang
"""

import subprocess
import sys

print("=" * 60)
print("TORCHAO VERSION COMPATIBILITY CHECK")
print("=" * 60)

# 1. Check installed torchao version
print("\n1. Installed torchao version:")
try:
    import torchao
    version = getattr(torchao, '__version__', 'Unknown')
    print(f"   torchao version: {version}")
except ImportError:
    print("   torchao is NOT installed")
    sys.exit(0)

# 2. Test the specific imports SGLang needs
print("\n2. Testing SGLang's required imports from torchao:")
print("   (from torchao.quantization import ...)")

imports_to_test = [
    "int4_weight_only",
    "int8_weight_only", 
    "int8_dynamic_activation_int8_weight",
]

for import_name in imports_to_test:
    try:
        exec(f"from torchao.quantization import {import_name}")
        print(f"   ✓ {import_name} - available")
    except ImportError:
        print(f"   ✗ {import_name} - NOT available")

# 3. Check what's actually in torchao.quantization
print("\n3. What's actually available in torchao.quantization:")
try:
    import torchao.quantization as tq
    all_items = dir(tq)
    
    # Filter for likely function/class names (not private)
    public_items = [x for x in all_items if not x.startswith('_')]
    
    print(f"   Found {len(public_items)} public items:")
    for item in sorted(public_items)[:20]:  # Show first 20
        print(f"   - {item}")
    
    if len(public_items) > 20:
        print(f"   ... and {len(public_items) - 20} more")
        
    # Check for similar names
    print("\n4. Checking for similar/renamed functions:")
    similar_patterns = ['int4', 'int8', 'weight', 'dynamic']
    for pattern in similar_patterns:
        matches = [x for x in public_items if pattern.lower() in x.lower()]
        if matches:
            print(f"   Functions containing '{pattern}':")
            for match in matches[:5]:
                print(f"   - {match}")
                
except Exception as e:
    print(f"   Error examining torchao.quantization: {e}")

# 4. Check PyPI for version history
print("\n5. Known torchao versions (from PyPI):")
print("   Checking available versions...")
try:
    result = subprocess.run(
        ["pip", "index", "versions", "torchao"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        output = result.stdout
        # Parse versions
        if "Available versions:" in output:
            versions_line = output.split("Available versions:")[1].split("\n")[0]
            print(f"   Available: {versions_line.strip()}")
    else:
        print("   Could not fetch version info from PyPI")
except Exception as e:
    print(f"   Error checking PyPI: {e}")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)

print("""
If the imports are failing, this confirms a version incompatibility.

Possible solutions:
1. Downgrade torchao to an older version that SGLang expects
2. Uninstall torchao (SGLang will skip it with a warning)
3. Update SGLang to a version compatible with torchao 0.11.0

To check which torchao version SGLang needs, look at:
- SGLang's GitHub repository for requirements
- The SGLang version release notes
""")