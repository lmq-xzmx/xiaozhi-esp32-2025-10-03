#!/usr/bin/env python3
"""
Redis优化配置模块
支持连接池、集群模式、缓存策略优化
"""

import asyncio
import logging
import json
import time
from typing import Optional, Dict, Any, List
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from redis.asyncio.sentinel import Sentinel
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RedisConfig:
    """Redis配置类"""
    host: str = "localhost"
    port: int = 6379
    password: Optional[str] = None
    db: int = 0
    max_connections: int = 50  # 增加连接池大小
    retry_on_timeout: bool = True
    socket_keepalive: bool = True
    socket_keepalive_options: Dict = None
    health_check_interval: int = 30
    
    # 集群配置
    cluster_mode: bool = False
    cluster_nodes: List[Dict] = None
    
    # 哨兵配置
    sentinel_mode: bool = False
    sentinel_hosts: List[tuple] = None
    sentinel_service_name: str = "mymaster"

class OptimizedRedisClient:
    """优化的Redis客户端，支持连接池和高级缓存策略"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.client = None
        self.pool = None
        self.is_cluster = config.cluster_mode
        self.is_sentinel = config.sentinel_mode
        
    async def initialize(self):
        """初始化Redis连接"""
        try:
            if self.is_cluster:
                await self._init_cluster()
            elif self.is_sentinel:
                await self._init_sentinel()
            else:
                await self._init_single()
            
            logger.info("Redis客户端初始化成功")
            return True
        except Exception as e:
            logger.error(f"Redis初始化失败: {e}")
            return False
    
    async def _init_single(self):
        """初始化单机Redis"""
        # 优化的连接池配置
        self.pool = ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            password=self.config.password,
            db=self.config.db,
            max_connections=self.config.max_connections,
            retry_on_timeout=self.config.retry_on_timeout,
            socket_keepalive=self.config.socket_keepalive,
            socket_keepalive_options=self.config.socket_keepalive_options or {
                1: 1,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            },
            health_check_interval=self.config.health_check_interval,
            socket_connect_timeout=5,
            socket_timeout=5,
            encoding='utf-8',
            decode_responses=True
        )
        
        self.client = redis.Redis(connection_pool=self.pool)
        await self.client.ping()
    
    async def _init_cluster(self):
        """初始化Redis集群"""
        from redis.asyncio.cluster import RedisCluster
        
        startup_nodes = self.config.cluster_nodes or [
            {"host": "127.0.0.1", "port": "7000"},
            {"host": "127.0.0.1", "port": "7001"},
            {"host": "127.0.0.1", "port": "7002"},
        ]
        
        self.client = RedisCluster(
            startup_nodes=startup_nodes,
            password=self.config.password,
            decode_responses=True,
            skip_full_coverage_check=True,
            max_connections_per_node=20,
            retry_on_timeout=True,
            socket_keepalive=True,
            health_check_interval=30
        )
        
        await self.client.ping()
    
    async def _init_sentinel(self):
        """初始化Redis哨兵"""
        sentinel_hosts = self.config.sentinel_hosts or [
            ('localhost', 26379),
            ('localhost', 26380),
            ('localhost', 26381)
        ]
        
        sentinel = Sentinel(
            sentinel_hosts,
            socket_timeout=5,
            socket_connect_timeout=5,
            socket_keepalive=True,
            socket_keepalive_options={1: 1, 2: 3, 3: 5}
        )
        
        self.client = sentinel.master_for(
            self.config.sentinel_service_name,
            password=self.config.password,
            db=self.config.db,
            decode_responses=True
        )
        
        await self.client.ping()
    
    async def get_with_fallback(self, key: str, fallback_func=None, ttl: int = 3600) -> Optional[Any]:
        """带回退机制的缓存获取"""
        try:
            # 尝试从缓存获取
            cached_data = await self.client.get(key)
            if cached_data:
                return json.loads(cached_data)
            
            # 缓存未命中，执行回退函数
            if fallback_func:
                data = await fallback_func() if asyncio.iscoroutinefunction(fallback_func) else fallback_func()
                if data is not None:
                    await self.set_with_ttl(key, data, ttl)
                return data
            
            return None
        except Exception as e:
            logger.error(f"Redis获取失败 {key}: {e}")
            # 如果Redis失败，直接执行回退函数
            if fallback_func:
                return await fallback_func() if asyncio.iscoroutinefunction(fallback_func) else fallback_func()
            return None
    
    async def set_with_ttl(self, key: str, value: Any, ttl: int = 3600):
        """设置带TTL的缓存"""
        try:
            serialized_value = json.dumps(value, ensure_ascii=False)
            await self.client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Redis设置失败 {key}: {e}")
            return False
    
    async def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取缓存"""
        try:
            if not keys:
                return {}
            
            # 使用pipeline提升性能
            pipe = self.client.pipeline()
            for key in keys:
                pipe.get(key)
            
            results = await pipe.execute()
            
            # 解析结果
            parsed_results = {}
            for key, result in zip(keys, results):
                if result:
                    try:
                        parsed_results[key] = json.loads(result)
                    except json.JSONDecodeError:
                        parsed_results[key] = result
                else:
                    parsed_results[key] = None
            
            return parsed_results
        except Exception as e:
            logger.error(f"批量获取失败: {e}")
            return {key: None for key in keys}
    
    async def batch_set(self, data: Dict[str, Any], ttl: int = 3600):
        """批量设置缓存"""
        try:
            if not data:
                return True
            
            pipe = self.client.pipeline()
            for key, value in data.items():
                serialized_value = json.dumps(value, ensure_ascii=False)
                pipe.setex(key, ttl, serialized_value)
            
            await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"批量设置失败: {e}")
            return False
    
    async def delete_pattern(self, pattern: str):
        """删除匹配模式的键"""
        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern, count=100):
                keys.append(key)
                if len(keys) >= 1000:  # 批量删除
                    await self.client.delete(*keys)
                    keys = []
            
            if keys:
                await self.client.delete(*keys)
            
            return True
        except Exception as e:
            logger.error(f"删除模式失败 {pattern}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取Redis统计信息"""
        try:
            info = await self.client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
            }
        except Exception as e:
            logger.error(f"获取Redis统计失败: {e}")
            return {}
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis健康检查失败: {e}")
            return False
    
    async def close(self):
        """关闭连接"""
        try:
            if self.client:
                await self.client.close()
            if self.pool:
                await self.pool.disconnect()
        except Exception as e:
            logger.error(f"关闭Redis连接失败: {e}")

# 全局Redis客户端实例
_redis_client = None

async def get_redis_client() -> OptimizedRedisClient:
    """获取全局Redis客户端"""
    global _redis_client
    if _redis_client is None:
        config = RedisConfig(
            host="localhost",
            port=6379,
            max_connections=50,
            health_check_interval=30
        )
        _redis_client = OptimizedRedisClient(config)
        await _redis_client.initialize()
    
    return _redis_client

async def close_redis_client():
    """关闭全局Redis客户端"""
    global _redis_client
    if _redis_client:
        await _redis_client.close()
        _redis_client = None