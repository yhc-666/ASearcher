#!/usr/bin/env python3
"""
Test torchao compatibility with SGLang
"""

import sys

print("Testing torchao import and compatibility...")
print("=" * 60)

# 1. Test basic torchao import
try:
    import torchao
    print(f"✓ torchao imported successfully")
    print(f"  Version: {torchao.__version__ if hasattr(torchao, '__version__') else 'Unknown'}")
except ImportError as e:
    print(f"✗ torchao import failed: {e}")
    sys.exit(1)

# 2. Test the specific import that SGLang uses
try:
    from torchao.quantization import (
        int4_weight_only,
        int8_weight_only,
        int8_dynamic_activation_int8_weight,
    )
    print("✓ SGLang's required torchao imports work")
except ImportError as e:
    print(f"✗ SGLang's torchao imports failed: {e}")
    print("\nThis is the issue! The torchao API has changed.")

# 3. Check what's actually available in torchao
print("\n" + "=" * 60)
print("Available in torchao.quantization:")
try:
    import torchao.quantization as tq
    available = [attr for attr in dir(tq) if not attr.startswith('_')]
    for item in available[:10]:  # Show first 10
        print(f"  - {item}")
    if len(available) > 10:
        print(f"  ... and {len(available) - 10} more")
except Exception as e:
    print(f"  Error exploring torchao: {e}")

# 4. Test if we can disable torchao usage
print("\n" + "=" * 60)
print("Workaround: Set torchao_config to empty string")
print("This should make SGLang skip torchao entirely")