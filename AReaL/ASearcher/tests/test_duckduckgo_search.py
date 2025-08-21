#!/usr/bin/env python3
"""
Comprehensive test suite for AsyncDuckDuckGoClient
Tests search functionality, content extraction, and integration with SearchToolBox
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ASearcher.utils.search_utils import AsyncDuckDuckGoClient, make_search_client


class TestAsyncDuckDuckGoClient:
    """Test suite for AsyncDuckDuckGoClient"""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
    
    async def run_all_tests(self):
        """Run all test cases"""
        print("=" * 80)
        print("AsyncDuckDuckGoClient Comprehensive Test Suite")
        print("=" * 80)
        
        # Test 1: Basic Search Functionality
        await self.test_basic_search()
        
        # Test 2: Multiple Concurrent Queries
        await self.test_concurrent_queries()
        
        # Test 3: Content Extraction
        await self.test_content_extraction()
        
        # Test 4: Cache Functionality
        await self.test_cache_functionality()
        
        # Test 5: Error Handling
        await self.test_error_handling()
        
        # Test 6: Factory Function
        await self.test_factory_function()
        
        # Test 7: Format Compatibility
        await self.test_format_compatibility()
        
        # Test 8: Special Characters and Edge Cases
        await self.test_edge_cases()
        
        # Print summary
        self.print_summary()
    
    async def test_basic_search(self):
        """Test basic search functionality"""
        print("\n" + "-" * 40)
        print("Test 1: Basic Search Functionality")
        print("-" * 40)
        
        try:
            client = AsyncDuckDuckGoClient(wrapper_format=True)
            
            # Test single query
            req_meta = {
                "queries": ["artificial intelligence"],
                "topk": 5
            }
            
            results = await client.query_async(req_meta)
            
            # Verify results structure
            assert results, "No results returned"
            assert isinstance(results, list), "Results should be a list"
            assert len(results) == 1, "Should return one result set for one query"
            
            result = results[0]
            assert "documents" in result, "Result should contain 'documents'"
            assert "urls" in result, "Result should contain 'urls'"
            assert "server_type" in result, "Result should contain 'server_type'"
            assert result["server_type"] == "async-duckduckgo-search", "Server type should be correct"
            
            # Verify content
            assert len(result["urls"]) > 0, "Should return at least one URL"
            assert len(result["documents"]) == len(result["urls"]), "Documents and URLs count should match"
            
            print(f"✅ Basic search test passed")
            print(f"   Found {len(result['urls'])} results for 'artificial intelligence'")
            print(f"   First URL: {result['urls'][0][:60]}...")
            
            self.passed_tests += 1
            self.test_results.append(("Basic Search", True, None))
            
        except Exception as e:
            print(f"❌ Basic search test failed: {e}")
            self.failed_tests += 1
            self.test_results.append(("Basic Search", False, str(e)))
    
    async def test_concurrent_queries(self):
        """Test multiple concurrent queries"""
        print("\n" + "-" * 40)
        print("Test 2: Multiple Concurrent Queries")
        print("-" * 40)
        
        try:
            client = AsyncDuckDuckGoClient(wrapper_format=False)
            
            # Test multiple queries
            req_meta = {
                "queries": [
                    "machine learning",
                    "deep learning",
                    "neural networks"
                ],
                "topk": 3
            }
            
            start_time = time.time()
            results = await client.query_async(req_meta)
            elapsed_time = time.time() - start_time
            
            # Verify results
            assert len(results) == 3, f"Should return 3 result sets, got {len(results)}"
            
            for i, (query, query_results) in enumerate(zip(req_meta["queries"], results)):
                assert isinstance(query_results, list), f"Query {i} results should be a list"
                print(f"   Query '{query}': {len(query_results)} results")
            
            print(f"✅ Concurrent queries test passed")
            print(f"   Completed 3 queries in {elapsed_time:.2f} seconds")
            
            self.passed_tests += 1
            self.test_results.append(("Concurrent Queries", True, None))
            
        except Exception as e:
            print(f"❌ Concurrent queries test failed: {e}")
            self.failed_tests += 1
            self.test_results.append(("Concurrent Queries", False, str(e)))
    
    async def test_content_extraction(self):
        """Test webpage content extraction using free Jina Reader"""
        print("\n" + "-" * 40)
        print("Test 3: Content Extraction (Free Jina)")
        print("-" * 40)
        
        try:
            client = AsyncDuckDuckGoClient(wrapper_format=True)
            
            # Test URLs
            test_urls = [
                "https://www.python.org",
                "https://www.wikipedia.org"
            ]
            
            results = await client.access_async(test_urls)
            
            # Verify results
            assert len(results) == len(test_urls), "Should return results for all URLs"
            
            for i, (url, result) in enumerate(zip(test_urls, results)):
                assert "page" in result, f"Result {i} should contain 'page'"
                assert "type" in result, f"Result {i} should contain 'type'"
                assert "server_type" in result, f"Result {i} should contain 'server_type'"
                
                if result["page"]:
                    print(f"   ✅ {url}: {len(result['page'])} chars extracted")
                    print(f"      Type: {result['type']}")
                else:
                    print(f"   ⚠️ {url}: No content extracted")
            
            print(f"✅ Content extraction test passed")
            
            self.passed_tests += 1
            self.test_results.append(("Content Extraction", True, None))
            
        except Exception as e:
            print(f"❌ Content extraction test failed: {e}")
            self.failed_tests += 1
            self.test_results.append(("Content Extraction", False, str(e)))
    
    async def test_cache_functionality(self):
        """Test webpage caching functionality"""
        print("\n" + "-" * 40)
        print("Test 4: Cache Functionality")
        print("-" * 40)
        
        try:
            # Create client with specific cache file
            cache_file = "/tmp/test_ddg_cache.json"
            client = AsyncDuckDuckGoClient(
                enable_cache=True,
                cache_file=cache_file,
                wrapper_format=True
            )
            
            test_url = "https://www.example.com"
            
            # Clear cache first
            client.clear_cache()
            initial_stats = client.get_cache_stats()
            print(f"   Initial cache stats: {initial_stats}")
            
            # First access (should miss cache)
            result1 = await client.access_async([test_url])
            stats1 = client.get_cache_stats()
            
            # Second access (should hit cache)
            result2 = await client.access_async([test_url])
            stats2 = client.get_cache_stats()
            
            # Verify cache behavior
            assert stats1["cache_size"] > 0, "Cache should have content after first access"
            assert stats2["hits"] > stats1.get("hits", 0), "Cache hits should increase"
            
            # Verify same content returned
            assert result1[0]["page"] == result2[0]["page"], "Cached content should be identical"
            
            print(f"✅ Cache functionality test passed")
            print(f"   Cache hits: {stats2.get('hits', 0)}")
            print(f"   Hit rate: {stats2.get('hit_rate', 0):.2%}")
            
            # Clean up
            client.clear_cache()
            
            self.passed_tests += 1
            self.test_results.append(("Cache Functionality", True, None))
            
        except Exception as e:
            print(f"❌ Cache functionality test failed: {e}")
            self.failed_tests += 1
            self.test_results.append(("Cache Functionality", False, str(e)))
    
    async def test_error_handling(self):
        """Test error handling for invalid inputs"""
        print("\n" + "-" * 40)
        print("Test 5: Error Handling")
        print("-" * 40)
        
        try:
            client = AsyncDuckDuckGoClient(wrapper_format=True)
            
            # Test 1: Empty query
            print("   Testing empty query...")
            empty_result = await client.query_async({"queries": [], "topk": 5})
            assert empty_result == [], "Empty query should return empty list"
            print("   ✅ Empty query handled correctly")
            
            # Test 2: Invalid URL
            print("   Testing invalid URL...")
            invalid_url_result = await client.access_async(["not-a-valid-url"])
            assert invalid_url_result[0]["page"] == "" or "Failed" in invalid_url_result[0]["page"], \
                "Invalid URL should return empty or error message"
            print("   ✅ Invalid URL handled correctly")
            
            # Test 3: Very long query
            print("   Testing very long query...")
            long_query = "python " * 500  # Create a very long query
            long_result = await client.query_async({"queries": [long_query], "topk": 1})
            # Should not crash, query will be truncated
            print("   ✅ Long query handled correctly")
            
            print(f"✅ Error handling test passed")
            
            self.passed_tests += 1
            self.test_results.append(("Error Handling", True, None))
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.failed_tests += 1
            self.test_results.append(("Error Handling", False, str(e)))
    
    async def test_factory_function(self):
        """Test the make_search_client factory function"""
        print("\n" + "-" * 40)
        print("Test 6: Factory Function")
        print("-" * 40)
        
        try:
            # Create client using factory
            client = make_search_client("async-duckduckgo-search")
            
            # Verify it's the correct type
            assert isinstance(client, AsyncDuckDuckGoClient), "Factory should create AsyncDuckDuckGoClient"
            
            # Test basic functionality
            req_meta = {"queries": ["test query"], "topk": 2}
            results = await client.query_async(req_meta)
            
            assert results, "Factory-created client should work"
            
            print(f"✅ Factory function test passed")
            print(f"   Successfully created and used client via factory")
            
            self.passed_tests += 1
            self.test_results.append(("Factory Function", True, None))
            
        except Exception as e:
            print(f"❌ Factory function test failed: {e}")
            self.failed_tests += 1
            self.test_results.append(("Factory Function", False, str(e)))
    
    async def test_format_compatibility(self):
        """Test output format compatibility with SearchToolBox"""
        print("\n" + "-" * 40)
        print("Test 7: Format Compatibility")
        print("-" * 40)
        
        try:
            # Test with wrapper_format=True (for SearchToolBox)
            client_wrapped = AsyncDuckDuckGoClient(wrapper_format=True)
            req_meta = {"queries": ["data science"], "topk": 3}
            
            wrapped_results = await client_wrapped.query_async(req_meta)
            
            # Verify wrapped format
            assert len(wrapped_results) == 1, "Wrapped format should return single element list"
            assert "documents" in wrapped_results[0], "Should have documents field"
            assert "urls" in wrapped_results[0], "Should have urls field"
            assert "server_type" in wrapped_results[0], "Should have server_type field"
            
            # Test with wrapper_format=False
            client_unwrapped = AsyncDuckDuckGoClient(wrapper_format=False)
            unwrapped_results = await client_unwrapped.query_async(req_meta)
            
            # Verify unwrapped format
            assert isinstance(unwrapped_results, list), "Unwrapped should return list"
            if unwrapped_results:
                assert "title" in unwrapped_results[0], "Results should have title"
                assert "url" in unwrapped_results[0], "Results should have url"
                assert "snippet" in unwrapped_results[0], "Results should have snippet"
            
            print(f"✅ Format compatibility test passed")
            print(f"   Wrapped format: {type(wrapped_results)}")
            print(f"   Unwrapped format: {type(unwrapped_results)}")
            
            self.passed_tests += 1
            self.test_results.append(("Format Compatibility", True, None))
            
        except Exception as e:
            print(f"❌ Format compatibility test failed: {e}")
            self.failed_tests += 1
            self.test_results.append(("Format Compatibility", False, str(e)))
    
    async def test_edge_cases(self):
        """Test special characters and edge cases"""
        print("\n" + "-" * 40)
        print("Test 8: Edge Cases & Special Characters")
        print("-" * 40)
        
        try:
            client = AsyncDuckDuckGoClient(wrapper_format=True)
            
            # Test queries with special characters
            special_queries = [
                "C++ programming",
                "What is 2+2?",
                "Python 3.11",
                "AI & ML",
                "搜索测试"  # Chinese characters
            ]
            
            for query in special_queries:
                req_meta = {"queries": [query], "topk": 1}
                try:
                    results = await client.query_async(req_meta)
                    if results and results[0]["urls"]:
                        print(f"   ✅ '{query}': Success ({len(results[0]['urls'])} results)")
                    else:
                        print(f"   ⚠️ '{query}': No results")
                except Exception as query_error:
                    print(f"   ❌ '{query}': Failed - {query_error}")
            
            print(f"✅ Edge cases test completed")
            
            self.passed_tests += 1
            self.test_results.append(("Edge Cases", True, None))
            
        except Exception as e:
            print(f"❌ Edge cases test failed: {e}")
            self.failed_tests += 1
            self.test_results.append(("Edge Cases", False, str(e)))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("Test Summary")
        print("=" * 80)
        
        total_tests = self.passed_tests + self.failed_tests
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        
        if total_tests > 0:
            success_rate = (self.passed_tests / total_tests) * 100
            print(f"Success Rate: {success_rate:.1f}%")
        
        print("\nDetailed Results:")
        for test_name, passed, error in self.test_results:
            status = "✅" if passed else "❌"
            error_msg = f" - {error}" if error else ""
            print(f"  {status} {test_name}{error_msg}")
        
        print("\n" + "=" * 80)
        if self.failed_tests == 0:
            print("🎉 All tests passed successfully!")
        else:
            print(f"⚠️ {self.failed_tests} test(s) failed. Please review the errors above.")
        print("=" * 80)


async def main():
    """Main test runner"""
    tester = TestAsyncDuckDuckGoClient()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())