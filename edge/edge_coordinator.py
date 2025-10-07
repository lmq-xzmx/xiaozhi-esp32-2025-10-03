#!/usr/bin/env python3
"""
Xiaozhi ESP32 Server - 边缘计算协调器
负责管理边缘节点的AI服务、负载均衡和与云端的通信
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import aiohttp
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import psutil
import GPUtil

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    VAD = "vad"
    ASR = "asr"
    TTS = "tts"
    LLM = "llm"

class NodeStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

@dataclass
class ServiceMetrics:
    """服务性能指标"""
    service_type: ServiceType
    requests_per_second: float
    average_latency: float
    error_rate: float
    queue_length: int
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    cache_hit_rate: float
    active_connections: int

@dataclass
class NodeInfo:
    """边缘节点信息"""
    node_id: str
    region: str
    status: NodeStatus
    capacity: Dict[str, Any]
    current_load: Dict[str, Any]
    services: List[ServiceMetrics]
    last_heartbeat: datetime
    uptime: float

class EdgeRequest(BaseModel):
    """边缘处理请求"""
    session_id: str
    service_type: str
    data: Dict[str, Any]
    priority: int = 1
    timeout: int = 30

class EdgeResponse(BaseModel):
    """边缘处理响应"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float
    node_id: str
    cached: bool = False

class IntelligentRouter:
    """智能路由器 - 负责请求分发和负载均衡"""
    
    def __init__(self):
        self.service_instances = {
            ServiceType.VAD: [],
            ServiceType.ASR: [],
            ServiceType.TTS: [],
            ServiceType.LLM: []
        }
        self.routing_stats = {}
        self.circuit_breakers = {}
        
    async def route_request(self, request: EdgeRequest) -> str:
        """智能路由请求到最优服务实例"""
        service_type = ServiceType(request.service_type)
        instances = self.service_instances[service_type]
        
        if not instances:
            raise HTTPException(status_code=503, detail=f"No {service_type.value} instances available")
        
        # 获取实例负载信息
        instance_loads = await self._get_instance_loads(instances)
        
        # 计算最优实例
        best_instance = self._select_best_instance(instances, instance_loads, request)
        
        # 更新路由统计
        await self._update_routing_stats(best_instance, request)
        
        return best_instance
    
    def _select_best_instance(self, instances: List[str], loads: Dict[str, Dict], request: EdgeRequest) -> str:
        """选择最优服务实例"""
        scores = {}
        
        for instance in instances:
            if instance in self.circuit_breakers and self.circuit_breakers[instance].is_open():
                continue
                
            load = loads.get(instance, {})
            score = self._calculate_instance_score(load, request)
            scores[instance] = score
        
        if not scores:
            # 所有实例都不可用，选择第一个实例并重置熔断器
            instance = instances[0]
            if instance in self.circuit_breakers:
                self.circuit_breakers[instance].reset()
            return instance
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _calculate_instance_score(self, load: Dict[str, Any], request: EdgeRequest) -> float:
        """计算实例评分"""
        # 多因子评分算法
        factors = {
            'cpu_usage': (100 - load.get('cpu', 50)) / 100 * 0.25,
            'memory_usage': (100 - load.get('memory', 50)) / 100 * 0.20,
            'gpu_usage': (100 - load.get('gpu', 50)) / 100 * 0.25,
            'queue_length': max(0, (50 - load.get('queue', 0)) / 50) * 0.20,
            'error_rate': max(0, (1 - load.get('error_rate', 0))) * 0.10
        }
        
        # 优先级加权
        priority_weight = min(request.priority / 5.0, 1.0)
        base_score = sum(factors.values())
        
        return base_score * (1 + priority_weight * 0.2)
    
    async def _get_instance_loads(self, instances: List[str]) -> Dict[str, Dict]:
        """获取实例负载信息"""
        loads = {}
        for instance in instances:
            try:
                # 这里应该从监控系统获取实际负载数据
                # 暂时使用模拟数据
                loads[instance] = {
                    'cpu': psutil.cpu_percent(),
                    'memory': psutil.virtual_memory().percent,
                    'gpu': self._get_gpu_usage(),
                    'queue': 0,  # 从实际队列获取
                    'error_rate': 0.01
                }
            except Exception as e:
                logger.error(f"Failed to get load for {instance}: {e}")
                loads[instance] = {'cpu': 100, 'memory': 100, 'gpu': 100, 'queue': 100, 'error_rate': 1.0}
        
        return loads
    
    def _get_gpu_usage(self) -> float:
        """获取GPU使用率"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return sum(gpu.load * 100 for gpu in gpus) / len(gpus)
        except:
            pass
        return 0.0
    
    async def _update_routing_stats(self, instance: str, request: EdgeRequest):
        """更新路由统计"""
        key = f"{request.service_type}:{instance}"
        if key not in self.routing_stats:
            self.routing_stats[key] = {
                'total_requests': 0,
                'total_latency': 0.0,
                'error_count': 0,
                'last_request': None
            }
        
        self.routing_stats[key]['total_requests'] += 1
        self.routing_stats[key]['last_request'] = datetime.now()

class CircuitBreaker:
    """熔断器 - 防止级联故障"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def is_open(self) -> bool:
        """检查熔断器是否开启"""
        if self.state == "open":
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).seconds > self.timeout:
                self.state = "half-open"
                return False
            return True
        return False
    
    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def reset(self):
        """重置熔断器"""
        self.failure_count = 0
        self.state = "closed"
        self.last_failure_time = None

class CacheManager:
    """多级缓存管理器"""
    
    def __init__(self):
        self.l1_cache = {}  # 内存缓存
        self.l1_max_size = 1000
        self.l1_access_times = {}
        
        self.redis_client = None
        self.cache_stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'total_requests': 0
        }
    
    async def initialize(self, redis_url: str):
        """初始化Redis连接"""
        try:
            self.redis_client = await aioredis.from_url(redis_url)
            logger.info("Redis cache initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        self.cache_stats['total_requests'] += 1
        
        # L1缓存查找
        if key in self.l1_cache:
            self.l1_access_times[key] = time.time()
            self.cache_stats['l1_hits'] += 1
            return self.l1_cache[key]
        
        # L2缓存查找 (Redis)
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    # 反序列化并提升到L1
                    value = json.loads(data)
                    await self.set_l1(key, value)
                    self.cache_stats['l2_hits'] += 1
                    return value
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """设置缓存数据"""
        # 设置到L1和L2
        await self.set_l1(key, value)
        
        if self.redis_client:
            try:
                await self.redis_client.setex(key, ttl, json.dumps(value))
            except Exception as e:
                logger.error(f"Redis set error: {e}")
    
    async def set_l1(self, key: str, value: Any):
        """设置L1缓存"""
        # LRU淘汰策略
        if len(self.l1_cache) >= self.l1_max_size:
            lru_key = min(self.l1_access_times.keys(), 
                         key=lambda k: self.l1_access_times[k])
            del self.l1_cache[lru_key]
            del self.l1_access_times[lru_key]
        
        self.l1_cache[key] = value
        self.l1_access_times[key] = time.time()
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_stats['total_requests']
        if total == 0:
            return 0.0
        
        hits = self.cache_stats['l1_hits'] + self.cache_stats['l2_hits']
        return hits / total

class EdgeCoordinator:
    """边缘计算协调器主类"""
    
    def __init__(self, node_id: str, cloud_endpoint: str):
        self.node_id = node_id
        self.cloud_endpoint = cloud_endpoint
        self.router = IntelligentRouter()
        self.cache_manager = CacheManager()
        
        self.node_info = NodeInfo(
            node_id=node_id,
            region="default",
            status=NodeStatus.HEALTHY,
            capacity={
                'max_devices': 150,
                'cpu_cores': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total // (1024**3),
                'gpu_count': len(GPUtil.getGPUs()) if GPUtil.getGPUs() else 0
            },
            current_load={},
            services=[],
            last_heartbeat=datetime.now(),
            uptime=0.0
        )
        
        self.start_time = time.time()
        self.metrics_history = []
        
    async def initialize(self):
        """初始化协调器"""
        # 初始化缓存
        await self.cache_manager.initialize("redis://localhost:6379")
        
        # 注册服务实例
        await self._discover_services()
        
        # 启动后台任务
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Edge coordinator {self.node_id} initialized")
    
    async def process_request(self, request: EdgeRequest) -> EdgeResponse:
        """处理边缘请求"""
        start_time = time.time()
        
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(request)
            cached_result = await self.cache_manager.get(cache_key)
            
            if cached_result:
                return EdgeResponse(
                    success=True,
                    data=cached_result,
                    processing_time=time.time() - start_time,
                    node_id=self.node_id,
                    cached=True
                )
            
            # 路由到最优服务实例
            instance = await self.router.route_request(request)
            
            # 处理请求
            result = await self._process_with_instance(instance, request)
            
            # 缓存结果
            if result['success']:
                await self.cache_manager.set(cache_key, result['data'])
            
            return EdgeResponse(
                success=result['success'],
                data=result.get('data'),
                error=result.get('error'),
                processing_time=time.time() - start_time,
                node_id=self.node_id,
                cached=False
            )
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            return EdgeResponse(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time,
                node_id=self.node_id
            )
    
    async def _process_with_instance(self, instance: str, request: EdgeRequest) -> Dict[str, Any]:
        """使用指定实例处理请求"""
        try:
            # 这里应该调用实际的服务实例
            # 暂时返回模拟结果
            await asyncio.sleep(0.1)  # 模拟处理时间
            
            return {
                'success': True,
                'data': {
                    'result': f"Processed by {instance}",
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_cache_key(self, request: EdgeRequest) -> str:
        """生成缓存键"""
        # 基于请求内容生成唯一键
        content_hash = hash(json.dumps(request.data, sort_keys=True))
        return f"{request.service_type}:{request.session_id}:{content_hash}"
    
    async def _discover_services(self):
        """发现可用服务实例"""
        # 这里应该从Kubernetes API或服务注册中心获取服务实例
        # 暂时使用静态配置
        self.router.service_instances = {
            ServiceType.VAD: ["vad-service-1", "vad-service-2", "vad-service-3"],
            ServiceType.ASR: ["asr-service-1", "asr-service-2", "asr-service-3", "asr-service-4"],
            ServiceType.TTS: ["tts-service-1", "tts-service-2", "tts-service-3"],
            ServiceType.LLM: ["llm-service-1", "llm-service-2"]
        }
        
        logger.info("Service discovery completed")
    
    async def _heartbeat_loop(self):
        """心跳循环 - 向云端报告状态"""
        while True:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(30)  # 30秒心跳间隔
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(10)
    
    async def _send_heartbeat(self):
        """发送心跳到云端"""
        self.node_info.last_heartbeat = datetime.now()
        self.node_info.uptime = time.time() - self.start_time
        self.node_info.current_load = await self._collect_current_load()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.cloud_endpoint}/api/edge/heartbeat",
                    json=asdict(self.node_info),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        logger.debug("Heartbeat sent successfully")
                    else:
                        logger.warning(f"Heartbeat failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    async def _collect_current_load(self) -> Dict[str, Any]:
        """收集当前负载信息"""
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'gpu_usage': self.router._get_gpu_usage(),
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': dict(psutil.net_io_counters()._asdict()),
            'active_connections': len(self.router.routing_stats),
            'cache_hit_rate': self.cache_manager.get_hit_rate()
        }
    
    async def _metrics_collection_loop(self):
        """指标收集循环"""
        while True:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 保留最近1小时的指标
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m['timestamp'] > cutoff_time
                ]
                
                await asyncio.sleep(60)  # 每分钟收集一次
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """收集详细指标"""
        return {
            'timestamp': datetime.now(),
            'node_id': self.node_id,
            'system': await self._collect_current_load(),
            'services': self._collect_service_metrics(),
            'cache': {
                'hit_rate': self.cache_manager.get_hit_rate(),
                'stats': self.cache_manager.cache_stats.copy()
            },
            'routing': self.router.routing_stats.copy()
        }
    
    def _collect_service_metrics(self) -> List[Dict[str, Any]]:
        """收集服务指标"""
        # 这里应该从实际服务收集指标
        # 暂时返回模拟数据
        return [
            {
                'service_type': 'vad',
                'requests_per_second': 10.5,
                'average_latency': 45.2,
                'error_rate': 0.001,
                'queue_length': 2
            },
            {
                'service_type': 'asr',
                'requests_per_second': 8.3,
                'average_latency': 320.1,
                'error_rate': 0.002,
                'queue_length': 5
            }
        ]
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # 30秒检查一次
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        # 检查系统资源
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # 更新节点状态
        if cpu_usage > 90 or memory_usage > 95:
            self.node_info.status = NodeStatus.UNHEALTHY
        elif cpu_usage > 80 or memory_usage > 85:
            self.node_info.status = NodeStatus.DEGRADED
        else:
            self.node_info.status = NodeStatus.HEALTHY
        
        # 检查服务实例健康状态
        # 这里应该实际检查各个服务的健康状态
        logger.debug(f"Node status: {self.node_info.status}")

# FastAPI应用
app = FastAPI(title="Xiaozhi Edge Coordinator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局协调器实例
coordinator = None

@app.on_event("startup")
async def startup_event():
    global coordinator
    import os
    
    node_id = os.getenv("NODE_ID", "edge-node-001")
    cloud_endpoint = os.getenv("CLOUD_ENDPOINT", "http://localhost:8080")
    
    coordinator = EdgeCoordinator(node_id, cloud_endpoint)
    await coordinator.initialize()

@app.post("/api/process", response_model=EdgeResponse)
async def process_request(request: EdgeRequest):
    """处理边缘请求"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    return await coordinator.process_request(request)

@app.get("/api/status")
async def get_status():
    """获取节点状态"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    return {
        "node_info": asdict(coordinator.node_info),
        "cache_stats": coordinator.cache_manager.cache_stats,
        "routing_stats": coordinator.router.routing_stats
    }

@app.get("/api/metrics")
async def get_metrics():
    """获取节点指标"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not initialized")
    
    return {
        "current": await coordinator._collect_metrics(),
        "history": coordinator.metrics_history[-60:]  # 最近60个数据点
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    if not coordinator:
        return {"status": "unhealthy", "reason": "coordinator not initialized"}
    
    return {
        "status": coordinator.node_info.status.value,
        "uptime": coordinator.node_info.uptime,
        "last_heartbeat": coordinator.node_info.last_heartbeat.isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "edge_coordinator:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1
    )