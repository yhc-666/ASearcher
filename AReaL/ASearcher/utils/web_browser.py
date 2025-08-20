"""
网页缓存工具模块

提供高效的网页内容缓存机制，减少重复的网络请求
支持LRU淈汰策略、持久化存储、自动保存

核心特性：
1. LRU缓存策略（最近最少使用淈汰）
2. 线程安全的并发访问
3. 自动持久化到JSON文件
4. 统计信息跟踪（命中率、淈汰次数等）

来源：Microsoft Autogen团队的开源实现
https://github.com/microsoft/autogen/blob/gaia_multiagent_v01_march_1st/autogen/browser_utils.py
"""
import atexit
from collections import OrderedDict
import hashlib
import json
import os
import threading
import time
from typing import Any, Dict, Optional

       
class WebPageCache:
    """
    网页内容LRU缓存
    
    使用OrderedDict实现LRU缓存，支持持久化存储
    主要用于缓存Jina Reader或其他网页读取工具的结果
    
    Args:
        max_size: 最大缓存条目数（默认100000）
        cache_file: 缓存文件路径
        save_interval: 每隔多少次操作自动保存
    """
    
    def __init__(self, max_size: int = 100000, cache_file: str = "./webpage_cache.json", save_interval: int = 10):
        self.max_size = max_size
        self.cache_file = cache_file
        self.cache = OrderedDict()  # LRU缓存容器
        self.lock = threading.Lock()  # 线程锁
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}  # 统计信息
        self.save_interval = save_interval
        self.operations_since_save = 0
        
        self.load_from_file()  # 启动时加载历史缓存
        
        atexit.register(self.save_to_file)  # 程序退出时自动保存
    
    def _generate_cache_key(self, url: str) -> str:
        """生成URL的MD5哈希作为缓存键"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def put(self, url: str, content: str):
        """
        将网页内容加入缓存
        
        如果缓存已满，会淈汰最久未使用的条目（LRU）
        每隔save_interval次操作会触发后台保存
        """
        if not url or not content:
            return
            
        cache_key = self._generate_cache_key(url)
        
        with self.lock:
            # 如果已存在，先删除旧条目（用于更新LRU顺序）
            if cache_key in self.cache:
                del self.cache[cache_key]
            
            # LRU淈汰：如果缓存已满，删除最早的条目
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # 删除最旧的
                self.stats["evictions"] += 1
            
            # 添加新条目到末尾（最新）
            self.cache[cache_key] = {
                "url": url,
                "content": content,
                "timestamp": time.time()
            }
            
            # 触发自动保存
            self.operations_since_save += 1
            if self.operations_since_save >= self.save_interval:
                self.operations_since_save = 0
                import threading
                threading.Thread(target=self._background_save, daemon=True).start()
    
    def get(self, url: str) -> Optional[str]:
        """
        从URL获取缓存的网页内容
        
        命中时会更新LRU顺序（移动到末尾）
        
        Returns:
            缓存的内容，或None如果未命中
        """
        cache_key = self._generate_cache_key(url)
        
        with self.lock:
            if cache_key in self.cache:
                # LRU更新：移动到末尾（标记为最近使用）
                entry = self.cache.pop(cache_key)
                self.cache[cache_key] = entry
                self.stats["hits"] += 1
                return entry["content"]
            else:
                self.stats["misses"] += 1
                return None
    
    def has(self, url: str) -> bool:
        """检查URL是否在缓存中（不更新LRU顺序）"""
        cache_key = self._generate_cache_key(url)
        with self.lock:
            return cache_key in self.cache
    
    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats = {"hits": 0, "misses": 0, "evictions": 0}
            self.operations_since_save = 0
    
    def force_save(self):
        self.save_to_file()
        self.operations_since_save = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            包含命中率、淈汰次数等统计数据的字典
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    def _background_save(self):
        try:
            self.save_to_file()
        except Exception as e:
            print(f"[ERROR] WebPageCache: Background save failed: {e}")

    def save_to_file(self):
        """
        将缓存保存到JSON文件
        
        保存内容包括：
        - 有序的缓存条目（保留LRU顺序）
        - 统计信息
        - 保存时间戳
        """
        try:
            with self.lock:
                # 保留顺序：转换为列表以保存LRU顺序
                ordered_cache = []
                for key, value in self.cache.items():
                    ordered_cache.append((key, value))
                
                cache_data = {
                    "cache_ordered": ordered_cache,
                    "stats": self.stats,
                    "max_size": self.max_size,
                    "saved_at": time.time()
                }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] WebPageCache: Saved {len(self.cache)} entries to {self.cache_file}")
            
        except Exception as e:
            print(f"[ERROR] WebPageCache: Failed to save cache to {self.cache_file}: {e}")
    
    def load_from_file(self):
        """
        从JSON文件加载缓存
        
        支持两种格式：
        1. cache_ordered: 保留LRU顺序的新格式
        2. cache: 旧格式（不保留LRU顺序）
        """
        if not os.path.exists(self.cache_file):
            print(f"[DEBUG] WebPageCache: No existing cache file {self.cache_file}, starting fresh")
            return
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            with self.lock:
                if "cache_ordered" in cache_data:
                    ordered_cache = cache_data["cache_ordered"]
                    self.cache = OrderedDict(ordered_cache)
                    print(f"[DEBUG] WebPageCache: Loaded ordered cache format")
                else:
                    loaded_cache = cache_data.get("cache", {})
                    self.cache = OrderedDict(loaded_cache)
                    print(f"[DEBUG] WebPageCache: Loaded legacy cache format (LRU order may be lost)")
                
                self.stats = cache_data.get("stats", {"hits": 0, "misses": 0, "evictions": 0})
                
                while len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
                    self.stats["evictions"] += 1
            
            saved_at = cache_data.get("saved_at", 0)
            saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(saved_at))
            
            print(f"[DEBUG] WebPageCache: Loaded {len(self.cache)} entries from {self.cache_file} (saved at {saved_time})")
            
        except Exception as e:
            print(f"[ERROR] WebPageCache: Failed to load cache from {self.cache_file}: {e}")
            with self.lock:
                self.cache = OrderedDict()
                self.stats = {"hits": 0, "misses": 0, "evictions": 0}
