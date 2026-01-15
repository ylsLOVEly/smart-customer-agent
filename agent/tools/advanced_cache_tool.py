"""
高级缓存工具 - 多层缓存架构
支持内存缓存、磁盘缓存、Redis缓存和智能缓存策略
"""
import json
import pickle
import hashlib
import time
import threading
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
from collections import OrderedDict
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis未安装，将使用本地缓存")

@dataclass
class CacheEntry:
    """缓存条目数据类"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    access_count: int = 0
    last_access: float = 0
    size_bytes: int = 0
    metadata: Dict = None

class AdvancedCacheManager:
    """高级缓存管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # 缓存配置
        self.memory_max_size = self.config.get('memory_max_size', 100 * 1024 * 1024)  # 100MB
        self.disk_max_size = self.config.get('disk_max_size', 1024 * 1024 * 1024)    # 1GB
        self.default_ttl = self.config.get('default_ttl', 3600)  # 1小时
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # 5分钟
        
        # 缓存存储
        self.memory_cache = OrderedDict()  # LRU内存缓存
        self.cache_dir = Path(self.config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Redis连接（如果可用）
        self.redis_client = None
        self._init_redis()
        
        # 统计信息
        self.stats = {
            'hits': {'memory': 0, 'disk': 0, 'redis': 0},
            'misses': 0,
            'evictions': 0,
            'size_bytes': {'memory': 0, 'disk': 0},
            'operations': {'get': 0, 'set': 0, 'delete': 0}
        }
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 启动后台清理线程
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        
        logging.info(f"高级缓存管理器初始化完成，内存限制: {self.memory_max_size//1024//1024}MB")
    
    def _init_redis(self):
        """初始化Redis连接"""
        if not REDIS_AVAILABLE:
            return
            
        try:
            redis_config = self.config.get('redis', {})
            if redis_config.get('enabled', False):
                self.redis_client = redis.Redis(
                    host=redis_config.get('host', 'localhost'),
                    port=redis_config.get('port', 6379),
                    db=redis_config.get('db', 0),
                    decode_responses=False,
                    socket_timeout=2
                )
                # 测试连接
                self.redis_client.ping()
                logging.info("Redis缓存连接成功")
        except Exception as e:
            logging.warning(f"Redis连接失败，使用本地缓存: {e}")
            self.redis_client = None
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        with self.lock:
            self.stats['operations']['get'] += 1
            
            cache_key = self._normalize_key(key)
            
            # 1. 检查内存缓存
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if not self._is_expired(entry):
                    # 更新访问信息
                    entry.access_count += 1
                    entry.last_access = time.time()
                    
                    # LRU更新：移到最后
                    self.memory_cache.move_to_end(cache_key)
                    
                    self.stats['hits']['memory'] += 1
                    return entry.value
                else:
                    # 过期删除
                    del self.memory_cache[cache_key]
            
            # 2. 检查Redis缓存
            if self.redis_client:
                try:
                    redis_data = self.redis_client.get(cache_key)
                    if redis_data:
                        entry = pickle.loads(redis_data)
                        if not self._is_expired(entry):
                            # 提升到内存缓存
                            self._set_memory_cache(cache_key, entry)
                            self.stats['hits']['redis'] += 1
                            return entry.value
                        else:
                            # 过期删除
                            self.redis_client.delete(cache_key)
                except Exception as e:
                    logging.warning(f"Redis读取失败: {e}")
            
            # 3. 检查磁盘缓存
            disk_result = self._get_disk_cache(cache_key)
            if disk_result is not None:
                entry = disk_result
                if not self._is_expired(entry):
                    # 提升到内存缓存
                    self._set_memory_cache(cache_key, entry)
                    self.stats['hits']['disk'] += 1
                    return entry.value
            
            # 缓存未命中
            self.stats['misses'] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            priority: str = 'normal', metadata: Dict = None) -> bool:
        """设置缓存值"""
        with self.lock:
            self.stats['operations']['set'] += 1
            
            cache_key = self._normalize_key(key)
            ttl = ttl or self.default_ttl
            
            # 创建缓存条目
            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=time.time(),
                expires_at=time.time() + ttl if ttl > 0 else None,
                access_count=0,
                last_access=time.time(),
                size_bytes=self._estimate_size(value),
                metadata=metadata or {}
            )
            
            # 根据优先级和大小选择存储策略
            stored = False
            
            # 1. 尝试内存缓存
            if priority in ['high', 'normal'] and entry.size_bytes < self.memory_max_size // 10:
                stored = self._set_memory_cache(cache_key, entry)
            
            # 2. 尝试Redis缓存
            if self.redis_client and (priority == 'high' or not stored):
                try:
                    serialized = pickle.dumps(entry)
                    redis_ttl = int(ttl) if ttl > 0 else None
                    self.redis_client.set(cache_key, serialized, ex=redis_ttl)
                    stored = True
                except Exception as e:
                    logging.warning(f"Redis写入失败: {e}")
            
            # 3. 磁盘缓存
            if not stored or priority == 'persistent':
                self._set_disk_cache(cache_key, entry)
                stored = True
            
            return stored
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self.lock:
            self.stats['operations']['delete'] += 1
            
            cache_key = self._normalize_key(key)
            deleted = False
            
            # 从内存删除
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
                deleted = True
            
            # 从Redis删除
            if self.redis_client:
                try:
                    self.redis_client.delete(cache_key)
                    deleted = True
                except Exception as e:
                    logging.warning(f"Redis删除失败: {e}")
            
            # 从磁盘删除
            disk_file = self.cache_dir / f"{cache_key}.cache"
            if disk_file.exists():
                disk_file.unlink()
                deleted = True
            
            return deleted
    
    def clear(self, pattern: Optional[str] = None):
        """清空缓存"""
        with self.lock:
            if pattern:
                # 按模式清理
                keys_to_delete = []
                for key in self.memory_cache.keys():
                    if pattern in key:
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    self.delete(key)
            else:
                # 清空所有
                self.memory_cache.clear()
                
                if self.redis_client:
                    try:
                        self.redis_client.flushdb()
                    except Exception as e:
                        logging.warning(f"Redis清空失败: {e}")
                
                for cache_file in self.cache_dir.glob("*.cache"):
                    try:
                        cache_file.unlink()
                    except Exception:
                        pass
    
    def _set_memory_cache(self, key: str, entry: CacheEntry) -> bool:
        """设置内存缓存"""
        try:
            # 检查容量限制
            while (self.stats['size_bytes']['memory'] + entry.size_bytes > self.memory_max_size 
                   and self.memory_cache):
                # LRU淘汰最旧的条目
                oldest_key, oldest_entry = self.memory_cache.popitem(last=False)
                self.stats['size_bytes']['memory'] -= oldest_entry.size_bytes
                self.stats['evictions'] += 1
            
            # 存储新条目
            self.memory_cache[key] = entry
            self.stats['size_bytes']['memory'] += entry.size_bytes
            return True
            
        except Exception as e:
            logging.error(f"内存缓存写入失败: {e}")
            return False
    
    def _get_disk_cache(self, key: str) -> Optional[CacheEntry]:
        """获取磁盘缓存"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                return entry
        except Exception as e:
            logging.warning(f"磁盘缓存读取失败: {e}")
            # 删除损坏的缓存文件
            try:
                cache_file.unlink()
            except:
                pass
        return None
    
    def _set_disk_cache(self, key: str, entry: CacheEntry) -> bool:
        """设置磁盘缓存"""
        try:
            cache_file = self.cache_dir / f"{key}.cache"
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
            return True
        except Exception as e:
            logging.error(f"磁盘缓存写入失败: {e}")
            return False
    
    def _normalize_key(self, key: str) -> str:
        """标准化缓存键"""
        # 使用MD5确保键的一致性和长度限制
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查缓存条目是否过期"""
        if entry.expires_at is None:
            return False
        return time.time() > entry.expires_at
    
    def _estimate_size(self, obj: Any) -> int:
        """估算对象大小"""
        try:
            return len(pickle.dumps(obj))
        except:
            # 简单估算
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            else:
                return 1024  # 默认1KB
    
    def _cleanup_worker(self):
        """后台清理线程"""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
                self._manage_disk_size()
            except Exception as e:
                logging.error(f"缓存清理失败: {e}")
    
    def _cleanup_expired(self):
        """清理过期缓存"""
        with self.lock:
            # 清理内存缓存
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                entry = self.memory_cache[key]
                self.stats['size_bytes']['memory'] -= entry.size_bytes
                del self.memory_cache[key]
            
            # 清理磁盘缓存
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    with open(cache_file, 'rb') as f:
                        entry = pickle.load(f)
                    if self._is_expired(entry):
                        cache_file.unlink()
                except Exception:
                    # 删除损坏的文件
                    cache_file.unlink()
    
    def _manage_disk_size(self):
        """管理磁盘缓存大小"""
        try:
            cache_files = list(self.cache_dir.glob("*.cache"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            if total_size > self.disk_max_size:
                # 按访问时间排序，删除最旧的文件
                cache_files.sort(key=lambda f: f.stat().st_atime)
                
                while total_size > self.disk_max_size * 0.8 and cache_files:
                    file_to_delete = cache_files.pop(0)
                    file_size = file_to_delete.stat().st_size
                    file_to_delete.unlink()
                    total_size -= file_size
                    self.stats['evictions'] += 1
                    
        except Exception as e:
            logging.error(f"磁盘缓存管理失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total_hits = sum(self.stats['hits'].values())
            total_requests = total_hits + self.stats['misses']
            hit_rate = (total_hits / max(total_requests, 1)) * 100
            
            return {
                'hit_rate': round(hit_rate, 2),
                'hits': self.stats['hits'].copy(),
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'operations': self.stats['operations'].copy(),
                'size_info': {
                    'memory_entries': len(self.memory_cache),
                    'memory_bytes': self.stats['size_bytes']['memory'],
                    'memory_usage': f"{self.stats['size_bytes']['memory']//1024//1024}MB",
                    'memory_limit': f"{self.memory_max_size//1024//1024}MB"
                },
                'config': {
                    'redis_available': self.redis_client is not None,
                    'default_ttl': self.default_ttl,
                    'cleanup_interval': self.cleanup_interval
                }
            }

class SmartCacheDecorator:
    """智能缓存装饰器"""
    
    def __init__(self, cache_manager: AdvancedCacheManager, 
                 ttl: int = 3600, priority: str = 'normal'):
        self.cache_manager = cache_manager
        self.ttl = ttl
        self.priority = priority
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"func:{func.__name__}:{self._hash_args(args, kwargs)}"
            
            # 尝试从缓存获取
            result = self.cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 缓存结果
            metadata = {
                'function': func.__name__,
                'execution_time': execution_time,
                'cached_at': datetime.now().isoformat()
            }
            
            self.cache_manager.set(
                cache_key, result, ttl=self.ttl, 
                priority=self.priority, metadata=metadata
            )
            
            return result
        return wrapper
    
    def _hash_args(self, args, kwargs):
        """生成参数哈希"""
        content = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(content.encode()).hexdigest()

# 全局缓存实例
_global_cache = None

def get_cache_manager(config: Dict = None) -> AdvancedCacheManager:
    """获取全局缓存管理器"""
    global _global_cache
    if _global_cache is None:
        _global_cache = AdvancedCacheManager(config)
    return _global_cache

def cache(ttl: int = 3600, priority: str = 'normal'):
    """缓存装饰器"""
    cache_manager = get_cache_manager()
    return SmartCacheDecorator(cache_manager, ttl, priority)

# 测试函数
if __name__ == "__main__":
    # 测试缓存系统
    cache_config = {
        'memory_max_size': 10 * 1024 * 1024,  # 10MB
        'default_ttl': 300,  # 5分钟
        'redis': {'enabled': False}  # 禁用Redis进行测试
    }
    
    cache_manager = AdvancedCacheManager(cache_config)
    
    # 基本测试
    cache_manager.set("test_key", "test_value", ttl=60)
    result = cache_manager.get("test_key")
    print(f"缓存测试: {result}")
    
    # 装饰器测试
    @cache(ttl=120, priority='high')
    def expensive_function(n):
        time.sleep(0.1)  # 模拟耗时操作
        return f"结果: {n * n}"
    
    # 第一次调用 - 会执行函数
    print(expensive_function(10))
    
    # 第二次调用 - 从缓存获取
    print(expensive_function(10))
    
    # 统计信息
    print("\n=== 缓存统计 ===")
    stats = cache_manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
