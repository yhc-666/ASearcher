"""
搜索引擎客户端工具模块

提供两种搜索客户端实现：
1. AsyncSearchBrowserClient: 连接本地RAG服务器（如local_retrieval_server.py）
2. AsyncOnlineSearchClient: 连接在线搜索引擎（Serper API + Jina Reader）

核心特性：
- 全异步操作，高并发性能
- 内置重试机制和错误处理
- 网页缓存支持（减少重复访问）
- 统一的API接口（query_async, access_async）
"""
import requests
import random
import time
import json
import asyncio
import html
import os
from typing import Dict, Any, List

import aiohttp
import asyncio
from typing import Dict, List, Any

try:
    from .web_browser import WebPageCache
    WEBPAGECACHE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] import web browser error: {e}")
    WEBPAGECACHE_AVAILABLE = False
    WebPageCache = None



SERPER_STATS = dict(
    num_requests = 0
)

class AsyncSearchBrowserClient:
    """
    本地RAG服务器客户端
    
    连接到local_retrieval_server.py启动的FAISS/BM25检索服务
    支持多服务器负载均衡和故障转移
    
    主要用于RAG场景，从local知识库检索文档
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = None
        self.server_list = self.get_server_list()
        self.server_addr = random.choice(self.server_list)     

    def get_server_list(self):
        """
        获取可用的RAG服务器地址列表
        
        从环境变量RAG_SERVER_ADDR_DIR指定的目录读取服务器地址
        文件格式：Host*_IP*.txt，每行一个地址
        """
        import glob
        rag_server_addr_dir = os.environ.get("RAG_SERVER_ADDR_DIR", "")

        server_list = []

        filenames = glob.glob(rag_server_addr_dir + "/Host*_IP*.txt")
        for filename in filenames:
            try:
                server_list.extend(open(filename).readlines())
            except:
                continue
        return server_list
        
    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
        """
        异步检索查询
        
        发送检索请求到RAG服务器的/retrieve端点
        支持自动重试和服务器切换
        
        Args:
            req_meta: 包含"queries"、"topk"等参数的请求字典
            
        Returns:
            检索结果列表，包含documents和urls
        """
        cnt = 0
        last_exception = None
        while cnt < 10:  # 最多重试10次
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{self.server_addr}/retrieve",
                        json=req_meta,
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120)
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            dict(
                                documents=[r["contents"] for r in result],
                                urls=[r["url"] for r in result],
                                server_type="async-search-browser",
                            ) for result in res["result"]
                        ]
            except Exception as e:
                print("query_async", self.server_list, e.__class__.__name__, e.__cause__)
                last_exception = e
                self.server_list = self.get_server_list()
                self.server_addr = random.choice(self.server_list)
                print(f"Search Engine switched to {self.server_addr}")
                cnt += 1
                await asyncio.sleep(10)
        
        raise RuntimeError("Fail to post search query to RAG server") from last_exception
        
    async def access_async(self, urls: List[str]) -> List[Dict]:
        """
        异步访问文档内容
        
        发送访问请求到RAG服务器的/access端点
        获取指定URL/文档ID的完整内容
        
        Args:
            urls: 要访问的URL或文档ID列表
            
        Returns:
            文档内容列表
        """
        cnt = 0
        last_exception = None        
        while cnt < 10:  # 最多重试10次
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{self.server_addr}/access",
                        json={"urls": urls},
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120)
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            dict(
                                page=result["contents"] if result is not None else "",
                                type="access",
                                server_type="async-search-browser",
                            ) for result in res["result"]
                        ]
            except Exception as e:
                print("access_async", self.server_list, e)
                last_exception = e
                self.server_list = self.get_server_list()
                self.server_addr = random.choice(self.server_list)
                print(f"Search Engine switched to {self.server_addr}")
                cnt += 1
                await asyncio.sleep(10)
        
        raise RuntimeError("Fail to post access request to RAG server") from last_exception

class AsyncOnlineSearchClient:
    """
    在线搜索引擎客户端
    
    使用Serper API进行网络搜索，使用Jina Reader读取网页内容
    内置网页缓存、重试机制、并发控制
    
    核心功能：
    1. Google搜索（通过Serper API）
    2. 网页内容读取（通过Jina Reader API）
    3. 结果缓存和并发控制
    """

    _search_semaphore = None  # 搜索并发控制
    _access_semaphore = None  # 网页访问并发控制
    
    @classmethod
    def _get_search_semaphore(cls):
        """获取搜索信号量，控制并发搜索数量（最多20个）"""
        if cls._search_semaphore is None:
            cls._search_semaphore = asyncio.Semaphore(20)
        return cls._search_semaphore
    
    @classmethod
    def _get_access_semaphore(cls):
        """获取网页访问信号量，控制并发访问数量（最多10个）"""
        if cls._access_semaphore is None:
            cls._access_semaphore = asyncio.Semaphore(10)  
        return cls._access_semaphore
    
    def __init__(self, enable_cache: bool = True, cache_size: int = 10000, cache_file: str = "../webpage_cache.json",
                 use_jina: bool = False, jina_api_key: str = None, wrapper_format: bool = True):
        # Serper API
        self.serper_server_addr = "https://google.serper.dev"
        self.serper_api_key = os.environ.get('SERPER_API_KEY', '')
        if not self.serper_api_key:
            raise RuntimeError("Serper API key is not set. Please configure it in config.yaml or set the SERPER_API_KEY environment variable.")
        self.serper_headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        self.max_workers = 10
        
        self.max_retries = 3
        self.retry_delay = 1.0
        self.backoff_factor = 2.0

        self.wrapper_format = wrapper_format

        self.use_jina = use_jina

        self.jina_api_key = jina_api_key or os.environ.get('JINA_API_KEY', '')
        if self.use_jina and not self.jina_api_key:
            raise RuntimeError("Jina is enabled but the API key is not set. Please configure it in config.yaml or set the JINA_API_KEY environment variable.")

        if enable_cache and WEBPAGECACHE_AVAILABLE:
            self.webpage_cache = WebPageCache(cache_size, cache_file, save_interval=5)
        else:
            self.webpage_cache = None
    
    async def _jina_readpage_async(self, session, url: str) -> str:
        """
        Read webpage content using Jina service asynchronously.
        
        Args:
            session: aiohttp ClientSession
            url: The URL to read
            
        Returns:
            str: The webpage content or error message
        """
        try:
            headers = {
                'Authorization': f'Bearer {self.jina_api_key}',
                'Content-Type': 'application/json',
            }
            
            async with session.get(f'https://r.jina.ai/{url}', headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content = await response.text()
                    return content
                else:
                    return f"[visit] Failed to read page. Status code: {response.status}"
        
        except Exception as e:
            return f"[visit] Failed to read page. Error: {str(e)}"
    
    async def query_async(self, req_meta):
        """
        异步搜索查询
        
        使用Serper API进行Google搜索
        支持批量查询、自动重试、结果格式化
        
        Args:
            req_meta: 包含"queries"和"topk"的请求字典
            
        Returns:
            格式化的搜索结果，包含documents和urls
        """
        import aiohttp
        
        queries = req_meta.get("queries", [])
        topk = req_meta.get("topk", 5)
        
        if not queries:
            return []
        
        async def single_serper_query_async(session, query: str, topk: int) -> dict:
            """
            单个搜索查询的内部函数
            
            包含重试逻辑、错误处理、速率限制
            """
            query = query[:2000]  # 限制查询长度
            async with self._get_search_semaphore():  # 并发控制
                payload = {
                    "q": query,
                    "num": topk
                }
                
                for attempt in range(4):
                    try:
                        if attempt > 0:
                            delay = 1.0 * (2 ** (attempt - 1))  # 1s, 2s, 4s
                            await asyncio.sleep(delay)
                        
                        
                        await asyncio.sleep(0.1)
                        
                        SERPER_STATS["num_requests"] += 1
                        print("Serper Stats: ", json.dumps(SERPER_STATS))

                        async with session.post(
                            f"{self.serper_server_addr}/search",
                            headers=self.serper_headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30)
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                if attempt > 0:
                                    print(f"[INFO] AsyncOnlineSearchClient: Query succeeded on retry {attempt}")
                                return {
                                    "success": True,
                                    "data": data
                                }
                            else:

                                response_text = await response.text()
                                error_msg = f"HTTP {response.status}: {response_text[:100]}"
                                print(f"[WARNING] AsyncOnlineSearchClient: HTTP error (attempt {attempt + 1}): {error_msg}")
                                if attempt == 3: 
                                    return {
                                        "success": False,
                                        "error": error_msg
                                    }
                            
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)[:100]}"
                        print(f"[WARNING] AsyncOnlineSearchClient: Error (attempt {attempt + 1}): {error_msg}")
                        if attempt == 3: 
                            return {
                                "success": False,
                                "error": error_msg
                            }
                
                return {
                    "success": False,
                    "error": "Unknown error after all retries"
                }
        
        async with aiohttp.ClientSession() as session:
            tasks = [single_serper_query_async(session, query, topk) for query in queries]
            serper_results = await asyncio.gather(*tasks)
        
        formatted_results = []
        for query, serper_result in zip(queries, serper_results):
            query_results = []
            
            if serper_result and serper_result.get("success", False):
                data = serper_result.get("data", {})
                organic_results = data.get("organic", [])[:topk]
                
                for result in organic_results:
                    query_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("link", ""),
                        "snippet": result.get("snippet", ""),
                        "server_type": "async-online-search",
                    })
            else:
                error = serper_result.get("error", "Unknown error") if serper_result else "No response"
                print(f"[ERROR] AsyncOnlineSearchClient: Search failed for '{query}': {error}")
            
            formatted_results.append(query_results)

        if self.wrapper_format:
            first_query_results = formatted_results[0] if formatted_results else []
            return [{
                "documents": [result.get("title", "") + " " + result.get("snippet", "") for result in first_query_results],
                "urls": [result.get("url", "") for result in first_query_results],
                "server_type": "async-online-search"
            }]
        else:
            if len(queries) == 1:
                return formatted_results[0]  # return [{...}, {...}] rather than [[{...}, {...}]]
            else:
                return formatted_results  # return [[...], [...]]

    async def access_async(self, urls):
        """
        异步访问网页内容
        
        优先使用缓存，如果未命中则使用Jina Reader API读取
        支持批量访问和结果缓存
        
        Args:
            urls: 要访问的URL列表
            
        Returns:
            网页内容列表
        """
        if not urls:
            return []
        
        results = []
        urls_to_fetch = []
        
        for url in urls:
            if self.webpage_cache and self.webpage_cache.has(url):
                cached_content = self.webpage_cache.get(url)
                if cached_content:
                    results.append(dict(page=cached_content, type="access"))
                else:
                    urls_to_fetch.append(url)
                    results.append(None)
            else:
                urls_to_fetch.append(url)
                results.append(None)
        
        if urls_to_fetch:
            if self.use_jina and self.jina_api_key:
                try:
                    async with self._get_access_semaphore():
                        fetched_results = await self._access_urls_jina_async(urls_to_fetch)
                    
                    fetch_index = 0
                    for i, result in enumerate(results):
                        if result is None:
                            if fetch_index < len(fetched_results):
                                fetched_result = fetched_results[fetch_index]
                                results[i] = fetched_result
                                
                                if self.webpage_cache and fetched_result.get("page"):
                                    self.webpage_cache.put(urls[i], fetched_result["page"])
                                
                                fetch_index += 1
                            else:
                                results[i] = dict(page="", type="access")
                                
                except Exception as e:
                    for i, result in enumerate(results):
                        if result is None:
                            results[i] = dict(page="", type="access")
            else:
                for i, result in enumerate(results):
                    if result is None:
                        results[i] = dict(page="", type="access")
        
        for result in results:
            if result is not None:
                result["server_type"] = "async-online-search"
        return results

    async def _access_urls_jina_async(self, urls):
        results = []
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for url in urls:
                    content = await self._jina_readpage_async(session, url)
                    if content and not content.startswith("[visit] Failed"):
                        results.append(dict(page=content, type="access"))
                    else:
                        results.append(dict(page="", type="access"))
                        
        except Exception as e:
            results = [dict(page="", type="access") for _ in urls]
        
        for r in results:
            if len(r["page"]) > 0:
                r["type"] = "jina"
            
        return results



    def get_cache_stats(self):
        if self.webpage_cache:
            return self.webpage_cache.get_stats()
        else:
            return {"cache_disabled": True}
    
    def clear_cache(self):
        if self.webpage_cache:
            self.webpage_cache.clear()
    
    def force_save_cache(self):
        if self.webpage_cache:
            self.webpage_cache.force_save()



class AsyncDuckDuckGoClient:
    """
    免费搜索引擎客户端 - 使用 DuckDuckGo + 免费 Jina Reader
    
    完全免费的搜索方案：
    1. DuckDuckGo 搜索 - 无需 API 密钥
    2. Jina Reader - 免费网页内容提取 (r.jina.ai)
    
    核心特性：
    - 无需任何 API 密钥
    - 异步操作，高并发性能
    - 内置重试机制和错误处理
    - 网页缓存支持
    - 与现有接口完全兼容
    """
    
    _search_semaphore = None  # 搜索并发控制
    _access_semaphore = None  # 网页访问并发控制
    
    @classmethod
    def _get_search_semaphore(cls):
        """获取搜索信号量，控制并发搜索数量（最多20个）"""
        if cls._search_semaphore is None:
            cls._search_semaphore = asyncio.Semaphore(20)
        return cls._search_semaphore
    
    @classmethod
    def _get_access_semaphore(cls):
        """获取网页访问信号量，控制并发访问数量（最多10个）"""
        if cls._access_semaphore is None:
            cls._access_semaphore = asyncio.Semaphore(10)
        return cls._access_semaphore
    
    def __init__(self, enable_cache: bool = True, cache_size: int = 10000, 
                 cache_file: str = "../webpage_cache.json", wrapper_format: bool = True):
        """
        初始化 DuckDuckGo 搜索客户端
        
        Args:
            enable_cache: 是否启用网页缓存
            cache_size: 缓存大小
            cache_file: 缓存文件路径
            wrapper_format: 是否使用包装格式（兼容 SearchToolBox）
        """
        self.max_workers = 10
        self.max_retries = 3
        self.retry_delay = 1.0
        self.backoff_factor = 2.0
        self.wrapper_format = wrapper_format
        
        # 初始化 DuckDuckGo 搜索（无需 API 密钥）
        try:
            # 尝试使用新版 ddgs 包
            from ddgs import DDGS
            self.ddgs = DDGS()
        except ImportError:
            try:
                # 回退到旧版 duckduckgo_search
                from duckduckgo_search import DDGS
                self.ddgs = DDGS()
            except ImportError:
                raise RuntimeError("Please install ddgs: pip install ddgs")
        
        # 网页缓存
        if enable_cache and WEBPAGECACHE_AVAILABLE:
            self.webpage_cache = WebPageCache(cache_size, cache_file, save_interval=5)
        else:
            self.webpage_cache = None
    
    async def _jina_readpage_async(self, session, url: str) -> str:
        """
        使用免费 Jina Reader 服务异步读取网页内容
        
        无需 API 密钥！只需在 URL 前添加 r.jina.ai
        
        Args:
            session: aiohttp ClientSession
            url: 要读取的 URL
            
        Returns:
            str: 网页内容或错误消息
        """
        try:
            # 使用免费的 Jina Reader - 无需 API 密钥
            jina_url = f'https://r.jina.ai/{url}'
            
            async with session.get(jina_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    content = await response.text()
                    return content
                else:
                    return f"[visit] Failed to read page. Status code: {response.status}"
        
        except Exception as e:
            return f"[visit] Failed to read page. Error: {str(e)}"
    
    async def query_async(self, req_meta):
        """
        异步搜索查询 - 使用 DuckDuckGo
        
        将同步的 DDGS 方法包装为异步操作
        支持批量查询、自动重试、结果格式化
        
        Args:
            req_meta: 包含 "queries" 和 "topk" 的请求字典
            
        Returns:
            格式化的搜索结果，包含 documents 和 urls
        """
        import aiohttp
        
        queries = req_meta.get("queries", [])
        topk = req_meta.get("topk", 5)
        
        if not queries:
            return []
        
        async def single_ddg_query_async(query: str, topk: int) -> dict:
            """
            单个 DuckDuckGo 搜索查询
            
            使用 asyncio.to_thread 将同步调用转为异步
            """
            query = query[:2000]  # 限制查询长度
            
            async with self._get_search_semaphore():  # 并发控制
                for attempt in range(self.max_retries):
                    try:
                        if attempt > 0:
                            delay = self.retry_delay * (self.backoff_factor ** (attempt - 1))
                            await asyncio.sleep(delay)
                        
                        # 使用 asyncio.to_thread 将同步的 DDGS 调用转为异步
                        # 注意：Python 3.9+ 支持 asyncio.to_thread
                        import sys
                        if sys.version_info >= (3, 9):
                            results = await asyncio.to_thread(
                                lambda: list(self.ddgs.text(query, max_results=topk))
                            )
                        else:
                            # Python 3.8 及以下使用 run_in_executor
                            loop = asyncio.get_event_loop()
                            results = await loop.run_in_executor(
                                None, 
                                lambda: list(self.ddgs.text(query, max_results=topk))
                            )
                        
                        if attempt > 0:
                            print(f"[INFO] AsyncDuckDuckGoClient: Query succeeded on retry {attempt}")
                        
                        return {
                            "success": True,
                            "data": results
                        }
                        
                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)[:100]}"
                        print(f"[WARNING] AsyncDuckDuckGoClient: Error (attempt {attempt + 1}): {error_msg}")
                        if attempt == self.max_retries - 1:
                            return {
                                "success": False,
                                "error": error_msg
                            }
                
                return {
                    "success": False,
                    "error": "Unknown error after all retries"
                }
        
        # 并发执行所有查询
        tasks = [single_ddg_query_async(query, topk) for query in queries]
        ddg_results = await asyncio.gather(*tasks)
        
        # 格式化结果
        formatted_results = []
        for query, ddg_result in zip(queries, ddg_results):
            query_results = []
            
            if ddg_result and ddg_result.get("success", False):
                data = ddg_result.get("data", [])
                
                for result in data[:topk]:
                    query_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "server_type": "async-duckduckgo-search",
                    })
            else:
                error = ddg_result.get("error", "Unknown error") if ddg_result else "No response"
                print(f"[ERROR] AsyncDuckDuckGoClient: Search failed for '{query}': {error}")
            
            formatted_results.append(query_results)
        
        # 返回格式与 AsyncOnlineSearchClient 保持一致
        if self.wrapper_format:
            first_query_results = formatted_results[0] if formatted_results else []
            return [{
                "documents": [result.get("title", "") + " " + result.get("snippet", "") for result in first_query_results],
                "urls": [result.get("url", "") for result in first_query_results],
                "server_type": "async-duckduckgo-search"
            }]
        else:
            if len(queries) == 1:
                return formatted_results[0]
            else:
                return formatted_results
    
    async def access_async(self, urls):
        """
        异步访问网页内容 - 使用免费 Jina Reader
        
        优先使用缓存，如果未命中则使用免费 Jina Reader (r.jina.ai)
        无需 API 密钥！
        
        Args:
            urls: 要访问的 URL 列表
            
        Returns:
            网页内容列表
        """
        if not urls:
            return []
        
        results = []
        urls_to_fetch = []
        
        # 检查缓存
        for url in urls:
            if self.webpage_cache and self.webpage_cache.has(url):
                cached_content = self.webpage_cache.get(url)
                if cached_content:
                    results.append(dict(page=cached_content, type="access", server_type="async-duckduckgo-search"))
                else:
                    urls_to_fetch.append(url)
                    results.append(None)
            else:
                urls_to_fetch.append(url)
                results.append(None)
        
        # 获取未缓存的内容
        if urls_to_fetch:
            try:
                async with self._get_access_semaphore():
                    fetched_results = await self._access_urls_jina_async(urls_to_fetch)
                
                fetch_index = 0
                for i, result in enumerate(results):
                    if result is None:
                        if fetch_index < len(fetched_results):
                            fetched_result = fetched_results[fetch_index]
                            results[i] = fetched_result
                            
                            # 缓存成功获取的内容
                            if self.webpage_cache and fetched_result.get("page"):
                                self.webpage_cache.put(urls[i], fetched_result["page"])
                            
                            fetch_index += 1
                        else:
                            results[i] = dict(page="", type="access", server_type="async-duckduckgo-search")
                            
            except Exception as e:
                print(f"[ERROR] AsyncDuckDuckGoClient: Failed to fetch URLs: {e}")
                for i, result in enumerate(results):
                    if result is None:
                        results[i] = dict(page="", type="access", server_type="async-duckduckgo-search")
        
        return results
    
    async def _access_urls_jina_async(self, urls):
        """
        使用免费 Jina Reader 批量获取网页内容
        """
        results = []
        
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                for url in urls:
                    content = await self._jina_readpage_async(session, url)
                    if content and not content.startswith("[visit] Failed"):
                        results.append(dict(page=content, type="jina", server_type="async-duckduckgo-search"))
                    else:
                        results.append(dict(page="", type="access", server_type="async-duckduckgo-search"))
                        
        except Exception as e:
            print(f"[ERROR] AsyncDuckDuckGoClient: Jina Reader error: {e}")
            results = [dict(page="", type="access", server_type="async-duckduckgo-search") for _ in urls]
        
        return results
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        if self.webpage_cache:
            return self.webpage_cache.get_stats()
        else:
            return {"cache_disabled": True}
    
    def clear_cache(self):
        """清空缓存"""
        if self.webpage_cache:
            self.webpage_cache.clear()
    
    def force_save_cache(self):
        """强制保存缓存"""
        if self.webpage_cache:
            self.webpage_cache.force_save()


# 搜索客户端注册表
SEARCH_CLIENTS = {
    "async-search-access": AsyncSearchBrowserClient,  # 本地RAG服务器
    "async-online-search-access": AsyncOnlineSearchClient,  # 在线搜索引擎
    "async-duckduckgo-search": AsyncDuckDuckGoClient,  # 免费 DuckDuckGo + Jina Reader
}


def make_search_client(search_client_type: str, use_jina: bool = False, jina_api_key: str = None):
    """
    工厂函数：创建搜索客户端实例
    
    Args:
        search_client_type: 客户端类型
            - "async-search-access": 本地RAG服务器
            - "async-online-search-access": 在线搜索引擎 (Serper + Jina)
            - "async-duckduckgo-search": 免费搜索 (DuckDuckGo + 免费 Jina Reader)
        use_jina: 是否使用Jina Reader读取网页
        jina_api_key: Jina API密钥
    
    Returns:
        搜索客户端实例
    """
    if search_client_type == "async-online-search":
        return SEARCH_CLIENTS[search_client_type](use_jina=use_jina, jina_api_key=jina_api_key)
    elif search_client_type == "async-online-search-access":
        return SEARCH_CLIENTS[search_client_type](use_jina=use_jina, jina_api_key=jina_api_key, wrapper_format=True)
    elif search_client_type == "async-duckduckgo-search":
        # DuckDuckGo 客户端不需要 API 密钥
        return SEARCH_CLIENTS[search_client_type](wrapper_format=True)
    else:
        return SEARCH_CLIENTS[search_client_type]()


if __name__ == "__main__":
    import sys
    
    async def test_search_clients():
        """Test both AsyncOnlineSearchClient and AsyncDuckDuckGoClient"""
        
        print("=" * 80)
        print("Testing Search Clients")
        print("=" * 80)
        
        # Test configuration
        test_query = "Python programming"
        test_url = "https://www.python.org"
        
        # Test AsyncDuckDuckGoClient (Free)
        print("\n" + "=" * 40)
        print("Testing AsyncDuckDuckGoClient (Free)")
        print("=" * 40)
        
        try:
            ddg_client = AsyncDuckDuckGoClient(wrapper_format=True)
            
            # Test search
            print("\n1. Testing DuckDuckGo Search...")
            search_req = {"queries": [test_query], "topk": 3}
            ddg_search_results = await ddg_client.query_async(search_req)
            
            if ddg_search_results and ddg_search_results[0]['urls']:
                print(f"   ✅ Found {len(ddg_search_results[0]['urls'])} results")
                print(f"   First URL: {ddg_search_results[0]['urls'][0][:80]}...")
                test_url = ddg_search_results[0]['urls'][0]  # Use first result for content test
            else:
                print("   ⚠️ No search results returned")
            
            # Test content extraction
            print("\n2. Testing Free Jina Reader Content Extraction...")
            ddg_content_results = await ddg_client.access_async([test_url])
            
            if ddg_content_results and ddg_content_results[0]['page']:
                content_preview = ddg_content_results[0]['page'][:200].replace('\n', ' ')
                print(f"   ✅ Content extracted ({len(ddg_content_results[0]['page'])} chars)")
                print(f"   Preview: {content_preview}...")
            else:
                print("   ⚠️ No content extracted")
                
            # Test cache
            print("\n3. Testing Cache...")
            cache_stats = ddg_client.get_cache_stats()
            print(f"   Cache stats: {cache_stats}")
            
        except Exception as e:
            print(f"   ❌ Error testing AsyncDuckDuckGoClient: {e}")
            import traceback
            traceback.print_exc()
        
        # Test AsyncOnlineSearchClient (Requires API keys)
        print("\n" + "=" * 40)
        print("Testing AsyncOnlineSearchClient (API)")
        print("=" * 40)
        
        try:
            # Check if API keys are available
            serper_key = os.environ.get('SERPER_API_KEY', '')
            jina_key = os.environ.get('JINA_API_KEY', '')
            
            if not serper_key:
                print("   ⚠️ SERPER_API_KEY not found - skipping AsyncOnlineSearchClient test")
            else:
                online_client = AsyncOnlineSearchClient(use_jina=bool(jina_key))
                
                # Test search
                print("\n1. Testing Serper Search...")
                search_req = {"queries": [test_query], "topk": 3}
                online_search_results = await online_client.query_async(search_req)
                
                if online_search_results and online_search_results[0]['urls']:
                    print(f"   ✅ Found {len(online_search_results[0]['urls'])} results")
                    print(f"   First URL: {online_search_results[0]['urls'][0][:80]}...")
                else:
                    print("   ⚠️ No search results returned")
                
                # Test content extraction
                if jina_key:
                    print("\n2. Testing Jina Reader Content Extraction...")
                    online_content_results = await online_client.access_async([test_url])
                    
                    if online_content_results and online_content_results[0]['page']:
                        content_preview = online_content_results[0]['page'][:200].replace('\n', ' ')
                        print(f"   ✅ Content extracted ({len(online_content_results[0]['page'])} chars)")
                        print(f"   Preview: {content_preview}...")
                    else:
                        print("   ⚠️ No content extracted")
                else:
                    print("\n2. Jina Reader test skipped (no JINA_API_KEY)")
                    
        except Exception as e:
            print(f"   ❌ Error testing AsyncOnlineSearchClient: {e}")
        
        print("\n" + "=" * 80)
        print("Test Complete")
        print("=" * 80)
    
    # Run the tests
    asyncio.run(test_search_clients())
