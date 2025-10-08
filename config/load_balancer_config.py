"""
负载均衡和缓存优化配置
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import aioredis
import json

logger = logging.getLogger(__name__)

class LoadBalancingStrategy(str, Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    CONSISTENT_HASH = "consistent_hash"
    INTELLIGENT = "intelligent"

class CacheStrategy(str, Enum):
    """缓存策略"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

@dataclass
class ServerNode:
    """服务器节点"""
    id: str
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    response_time: float = 0.0
    success_rate: float = 1.0
    health_status: str = "healthy"  # healthy, unhealthy, unknown
    last_health_check: float = 0.0
    total_requests: int = 0
    failed_requests: int = 0
    
    @property
    def load_factor(self) -> float:
        """负载因子"""
        if self.max_connections == 0:
            return 1.0
        return self.current_connections / self.max_connections
    
    @property
    def score(self) -> float:
        """节点评分（越低越好）"""
        if self.health_status != "healthy":
            return float('inf')
        
        # 综合考虑负载、响应时间和成功率
        load_score = self.load_factor
        response_score = self.response_time / 1000  # 转换为秒
        success_score = 1.0 - self.success_rate
        
        return load_score * 0.4 + response_score * 0.3 + success_score * 0.3

@dataclass
class CacheConfig:
    """缓存配置"""
    strategy: CacheStrategy = CacheStrategy.ADAPTIVE
    max_size: int = 10000
    ttl: int = 3600  # 默认1小时
    compression: bool = True
    serialization: str = "json"  # json, pickle, msgpack
    
    # 自适应缓存参数
    hit_rate_threshold: float = 0.8
    size_adjustment_factor: float = 0.1
    ttl_adjustment_factor: float = 0.2

class LoadBalancer:
    """智能负载均衡器"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.INTELLIGENT):
        self.strategy = strategy
        self.nodes: Dict[str, ServerNode] = {}
        self.current_index = 0
        self.hash_ring: Dict[int, str] = {}  # 一致性哈希环
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time": 0.0
        }
    
    def add_node(self, node: ServerNode):
        """添加节点"""
        self.nodes[node.id] = node
        if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            self._rebuild_hash_ring()
    
    def remove_node(self, node_id: str):
        """移除节点"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            if self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                self._rebuild_hash_ring()
    
    def select_node(self, request_key: Optional[str] = None) -> Optional[ServerNode]:
        """选择节点"""
        healthy_nodes = [node for node in self.nodes.values() if node.health_status == "healthy"]
        
        if not healthy_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(healthy_nodes)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_select(request_key or "default")
        elif self.strategy == LoadBalancingStrategy.INTELLIGENT:
            return self._intelligent_select(healthy_nodes)
        else:
            return healthy_nodes[0]
    
    def _round_robin_select(self, nodes: List[ServerNode]) -> ServerNode:
        """轮询选择"""
        node = nodes[self.current_index % len(nodes)]
        self.current_index += 1
        return node
    
    def _weighted_round_robin_select(self, nodes: List[ServerNode]) -> ServerNode:
        """加权轮询选择"""
        total_weight = sum(node.weight for node in nodes)
        if total_weight == 0:
            return nodes[0]
        
        # 简化的加权轮询实现
        import random
        rand = random.uniform(0, total_weight)
        current_weight = 0
        
        for node in nodes:
            current_weight += node.weight
            if rand <= current_weight:
                return node
        
        return nodes[-1]
    
    def _least_connections_select(self, nodes: List[ServerNode]) -> ServerNode:
        """最少连接选择"""
        return min(nodes, key=lambda n: n.current_connections)
    
    def _least_response_time_select(self, nodes: List[ServerNode]) -> ServerNode:
        """最短响应时间选择"""
        return min(nodes, key=lambda n: n.response_time)
    
    def _consistent_hash_select(self, key: str) -> Optional[ServerNode]:
        """一致性哈希选择"""
        if not self.hash_ring:
            return None
        
        hash_value = hash(key) % (2**32)
        
        # 找到第一个大于等于hash_value的节点
        for ring_hash in sorted(self.hash_ring.keys()):
            if ring_hash >= hash_value:
                node_id = self.hash_ring[ring_hash]
                return self.nodes.get(node_id)
        
        # 如果没找到，返回第一个节点（环形）
        first_hash = min(self.hash_ring.keys())
        node_id = self.hash_ring[first_hash]
        return self.nodes.get(node_id)
    
    def _intelligent_select(self, nodes: List[ServerNode]) -> ServerNode:
        """智能选择（综合评分）"""
        return min(nodes, key=lambda n: n.score)
    
    def _rebuild_hash_ring(self):
        """重建哈希环"""
        self.hash_ring.clear()
        
        for node in self.nodes.values():
            # 为每个节点创建多个虚拟节点
            virtual_nodes = int(node.weight * 100)
            for i in range(virtual_nodes):
                virtual_key = f"{node.id}:{i}"
                hash_value = hash(virtual_key) % (2**32)
                self.hash_ring[hash_value] = node.id
    
    def update_node_stats(self, node_id: str, response_time: float, success: bool):
        """更新节点统计"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        node.total_requests += 1
        
        if success:
            # 更新响应时间（移动平均）
            alpha = 0.1  # 平滑因子
            node.response_time = alpha * response_time + (1 - alpha) * node.response_time
            self.stats["successful_requests"] += 1
        else:
            node.failed_requests += 1
            self.stats["failed_requests"] += 1
        
        # 更新成功率
        node.success_rate = (node.total_requests - node.failed_requests) / node.total_requests
        
        self.stats["total_requests"] += 1
        self.stats["avg_response_time"] = (
            self.stats["avg_response_time"] * (self.stats["total_requests"] - 1) + response_time
        ) / self.stats["total_requests"]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "strategy": self.strategy.value,
            "total_nodes": len(self.nodes),
            "healthy_nodes": len([n for n in self.nodes.values() if n.health_status == "healthy"]),
            "stats": self.stats,
            "nodes": [
                {
                    "id": node.id,
                    "host": node.host,
                    "port": node.port,
                    "weight": node.weight,
                    "current_connections": node.current_connections,
                    "response_time": node.response_time,
                    "success_rate": node.success_rate,
                    "health_status": node.health_status,
                    "load_factor": node.load_factor,
                    "score": node.score
                }
                for node in self.nodes.values()
            ]
        }

class AdaptiveCache:
    """自适应缓存"""
    
    def __init__(self, config: CacheConfig, redis_client=None):
        self.config = config
        self.redis_client = redis_client
        self.local_cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = {}
        self.access_time: Dict[str, float] = {}
        
        # 统计信息
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
            "size": 0
        }
        
        # 自适应参数
        self.current_ttl = config.ttl
        self.current_max_size = config.max_size
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        try:
            # 先检查本地缓存
            if key in self.local_cache:
                self._update_access_stats(key)
                self.stats["hits"] += 1
                return self.local_cache[key]
            
            # 检查Redis缓存
            if self.redis_client:
                value = await self.redis_client.get(key)
                if value:
                    # 反序列化
                    data = self._deserialize(value)
                    # 更新本地缓存
                    await self.set(key, data, local_only=True)
                    self.stats["hits"] += 1
                    return data
            
            self.stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"缓存获取失败: {e}")
            self.stats["misses"] += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, local_only: bool = False):
        """设置缓存"""
        try:
            ttl = ttl or self.current_ttl
            
            # 检查本地缓存大小
            if len(self.local_cache) >= self.current_max_size:
                await self._evict_local()
            
            # 设置本地缓存
            self.local_cache[key] = value
            self._update_access_stats(key)
            
            # 设置Redis缓存
            if self.redis_client and not local_only:
                serialized_value = self._serialize(value)
                await self.redis_client.setex(key, ttl, serialized_value)
            
            self.stats["sets"] += 1
            self.stats["size"] = len(self.local_cache)
            
        except Exception as e:
            logger.error(f"缓存设置失败: {e}")
    
    async def delete(self, key: str):
        """删除缓存"""
        try:
            # 删除本地缓存
            if key in self.local_cache:
                del self.local_cache[key]
            
            if key in self.access_count:
                del self.access_count[key]
            
            if key in self.access_time:
                del self.access_time[key]
            
            # 删除Redis缓存
            if self.redis_client:
                await self.redis_client.delete(key)
            
            self.stats["deletes"] += 1
            self.stats["size"] = len(self.local_cache)
            
        except Exception as e:
            logger.error(f"缓存删除失败: {e}")
    
    async def clear(self):
        """清空缓存"""
        self.local_cache.clear()
        self.access_count.clear()
        self.access_time.clear()
        
        if self.redis_client:
            await self.redis_client.flushdb()
        
        self.stats["size"] = 0
    
    def _update_access_stats(self, key: str):
        """更新访问统计"""
        self.access_count[key] = self.access_count.get(key, 0) + 1
        self.access_time[key] = time.time()
    
    async def _evict_local(self):
        """本地缓存淘汰"""
        if not self.local_cache:
            return
        
        if self.config.strategy == CacheStrategy.LRU:
            # 最近最少使用
            oldest_key = min(self.access_time.keys(), key=lambda k: self.access_time[k])
            del self.local_cache[oldest_key]
            del self.access_time[oldest_key]
            if oldest_key in self.access_count:
                del self.access_count[oldest_key]
        
        elif self.config.strategy == CacheStrategy.LFU:
            # 最少使用频率
            least_used_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.local_cache[least_used_key]
            del self.access_count[least_used_key]
            if least_used_key in self.access_time:
                del self.access_time[least_used_key]
        
        else:
            # 默认删除第一个
            key = next(iter(self.local_cache))
            del self.local_cache[key]
            if key in self.access_count:
                del self.access_count[key]
            if key in self.access_time:
                del self.access_time[key]
        
        self.stats["evictions"] += 1
    
    def _serialize(self, value: Any) -> str:
        """序列化"""
        if self.config.serialization == "json":
            return json.dumps(value, ensure_ascii=False)
        elif self.config.serialization == "pickle":
            import pickle
            import base64
            return base64.b64encode(pickle.dumps(value)).decode()
        else:
            return str(value)
    
    def _deserialize(self, value: str) -> Any:
        """反序列化"""
        if self.config.serialization == "json":
            return json.loads(value)
        elif self.config.serialization == "pickle":
            import pickle
            import base64
            return pickle.loads(base64.b64decode(value.encode()))
        else:
            return value
    
    def get_hit_rate(self) -> float:
        """获取命中率"""
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0
    
    async def optimize(self):
        """自适应优化"""
        if self.config.strategy != CacheStrategy.ADAPTIVE:
            return
        
        hit_rate = self.get_hit_rate()
        
        # 根据命中率调整缓存大小
        if hit_rate < self.config.hit_rate_threshold:
            # 命中率低，增加缓存大小
            self.current_max_size = int(
                self.current_max_size * (1 + self.config.size_adjustment_factor)
            )
        elif hit_rate > 0.95:
            # 命中率很高，可以减少缓存大小
            self.current_max_size = int(
                self.current_max_size * (1 - self.config.size_adjustment_factor)
            )
        
        # 根据访问模式调整TTL
        if self.access_time:
            avg_access_interval = (time.time() - min(self.access_time.values())) / len(self.access_time)
            optimal_ttl = int(avg_access_interval * 2)  # TTL设为平均访问间隔的2倍
            
            if abs(optimal_ttl - self.current_ttl) > self.current_ttl * self.config.ttl_adjustment_factor:
                self.current_ttl = optimal_ttl
        
        logger.info(f"缓存优化: 命中率={hit_rate:.2f}, 大小={self.current_max_size}, TTL={self.current_ttl}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "config": {
                "strategy": self.config.strategy.value,
                "max_size": self.config.max_size,
                "current_max_size": self.current_max_size,
                "ttl": self.config.ttl,
                "current_ttl": self.current_ttl
            },
            "stats": self.stats,
            "hit_rate": self.get_hit_rate(),
            "local_cache_size": len(self.local_cache),
            "access_patterns": {
                "most_accessed": sorted(
                    self.access_count.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10] if self.access_count else []
            }
        }

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
    
    async def start(self, load_balancer: LoadBalancer):
        """启动健康检查"""
        self.running = True
        while self.running:
            try:
                await self._check_all_nodes(load_balancer)
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
    
    async def stop(self):
        """停止健康检查"""
        self.running = False
    
    async def _check_all_nodes(self, load_balancer: LoadBalancer):
        """检查所有节点"""
        tasks = []
        for node in load_balancer.nodes.values():
            task = asyncio.create_task(self._check_node(node))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_node(self, node: ServerNode):
        """检查单个节点"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                async with session.get(
                    f"http://{node.host}:{node.port}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        node.health_status = "healthy"
                        # 更新响应时间
                        alpha = 0.1
                        node.response_time = alpha * response_time + (1 - alpha) * node.response_time
                    else:
                        node.health_status = "unhealthy"
            
            node.last_health_check = time.time()
            
        except Exception as e:
            logger.warning(f"节点 {node.id} 健康检查失败: {e}")
            node.health_status = "unhealthy"
            node.last_health_check = time.time()

# 全局配置
DEFAULT_LOAD_BALANCER_CONFIG = {
    "strategy": LoadBalancingStrategy.INTELLIGENT,
    "health_check_interval": 30,
    "max_retries": 3,
    "retry_delay": 1.0
}

DEFAULT_CACHE_CONFIG = CacheConfig(
    strategy=CacheStrategy.ADAPTIVE,
    max_size=10000,
    ttl=3600,
    compression=True,
    serialization="json"
)