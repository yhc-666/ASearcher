#!/usr/bin/env python3
"""Test script to diagnose DuckDuckGo search issues"""

import asyncio
import sys
from ddgs import DDGS

def test_basic_search():
    """Test basic synchronous search"""
    print("Testing basic DuckDuckGo search...")
    try:
        ddgs = DDGS()
        results = list(ddgs.text("test query", max_results=2))
        print(f"Success! Got {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"  Result {i}: {result.get('title', 'No title')[:50]}")
        return True
    except Exception as e:
        print(f"Error in basic search: {type(e).__name__}: {e}")
        return False

async def test_async_search():
    """Test async search wrapper"""
    print("\nTesting async search wrapper...")
    try:
        ddgs = DDGS()
        # Use asyncio.to_thread for Python 3.9+
        if sys.version_info >= (3, 9):
            results = await asyncio.to_thread(
                lambda: list(ddgs.text("test async", max_results=2))
            )
        else:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: list(ddgs.text("test async", max_results=2))
            )
        print(f"Async success! Got {len(results)} results")
        return True
    except Exception as e:
        print(f"Error in async search: {type(e).__name__}: {e}")
        return False

def test_ddgs_with_proxy():
    """Test with explicit proxy settings"""
    print("\nTesting with proxy settings...")
    try:
        # Try with explicit timeout and proxy settings
        ddgs = DDGS(timeout=20)
        results = list(ddgs.text("proxy test", max_results=1))
        print(f"Proxy test success! Got {len(results)} results")
        return True
    except Exception as e:
        print(f"Error with proxy: {type(e).__name__}: {e}")
        return False

def main():
    print("=" * 50)
    print("DuckDuckGo Search Diagnostic Test")
    print("=" * 50)
    
    # Run tests
    basic_ok = test_basic_search()
    
    # Run async test
    async_ok = asyncio.run(test_async_search())
    
    # Test with proxy
    proxy_ok = test_ddgs_with_proxy()
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"  Basic search: {'✓ PASS' if basic_ok else '✗ FAIL'}")
    print(f"  Async search: {'✓ PASS' if async_ok else '✗ FAIL'}")
    print(f"  Proxy search: {'✓ PASS' if proxy_ok else '✗ FAIL'}")
    print("=" * 50)
    
    if not (basic_ok or async_ok or proxy_ok):
        print("\nAll tests failed. Possible issues:")
        print("1. Network/firewall blocking DuckDuckGo")
        print("2. DNS configuration issues")
        print("3. Curl/libcurl configuration problems")
        print("4. Corporate proxy blocking requests")
        print("\nTry setting HTTP_PROXY and HTTPS_PROXY environment variables")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())