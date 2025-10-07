#!/usr/bin/env python3
"""
Xiaozhi ESP32 Server - 智能负载均衡器
实现多维度负载均衡、流量调度和自适应路由
"""

import asyncio
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from collections import defaultdict, deque
import statistics

import aioredis
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceType(Enum):
    VAD = "vad"
    ASR = "asr"
    LLM = "llm"
    TTS = "tts"
    EDGE_NODE = "edge_node"

class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    INTELLIGENT = "intelligent"

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class ServiceInstance:
    """服务实例"""
    id: str
    service_type: ServiceType
    host: str
    port: int
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    queue_length: int = 0
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    created_at: datetime = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def load_score(self) -> float:
        """计算负载评分（越低越好）"""
        # 综合考虑连接数、响应时间、资源使用率
        connection_score = self.current_connections / max(self.max_connections, 1)
        response_time_score = min(self.avg_response_time / 1000, 1.0)  # 归一化到1秒
        cpu_score = self.cpu_usage / 100
        memory_score = self.memory_usage / 100
        gpu_score = self.gpu_usage / 100 if self.gpu_usage > 0 else 0
        queue_score = min(self.queue_length / 100, 1.0)  # 归一化到100个请求
        
        # 加权计算
        weights = {
            'connection': 0.2,
            'response_time': 0.25,
            'cpu': 0.2,
            'memory': 0.15,
            'gpu': 0.1,
            'queue': 0.1
        }
        
        total_score = (
            connection_score * weights['connection'] +
            response_time_score * weights['response_time'] +
            cpu_score * weights['cpu'] +
            memory_score * weights['memory'] +
            gpu_score * weights['gpu'] +
            queue_score * weights['queue']
        )
        
        return total_score

@dataclass
class RequestMetrics:
    """请求指标"""
    timestamp: datetime
    service_type: ServiceType
    instance_id: str
    response_time: float
    success: bool
    error_type: Optional[str] = None
    request_size: int = 0
    response_size: int = 0

@dataclass
class TrafficPattern:
    """流量模式"""
    pattern_id: str
    service_type: ServiceType
    time_window: timedelta
    request_count: int
    avg_response_time: float
    peak_qps: float
    error_rate: float
    resource_usage: Dict[str, float]
    detected_at: datetime

class CircuitBreaker:
    """熔断器"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, request_threshold: int = 10):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.request_threshold = request_threshold
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.request_count = 0
        self.success_count = 0
    
    def can_execute(self) -> bool:
        """检查是否可以执行请求"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and \
               (datetime.now() - self.last_failure_time).seconds >= self.recovery_timeout:
                self.state = "HALF_OPEN"
                self.request_count = 0
                self.success_count = 0
                return True
            return False
        elif self.state == "HALF_OPEN":
            return self.request_count < self.request_threshold
        
        return False
    
    def record_success(self):
        """记录成功"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            self.request_count += 1
            
            if self.success_count >= self.request_threshold // 2:
                self.state = "CLOSED"
                self.failure_count = 0
        elif self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
        elif self.state == "HALF_OPEN":
            self.state = "OPEN"
            self.request_count += 1

class AdaptiveRouter:
    """自适应路由器"""
    
    def __init__(self):
        self.routing_rules = {}
        self.traffic_patterns = {}
        self.performance_history = defaultdict(list)
        self.learning_rate = 0.1
    
    def add_routing_rule(self, rule_id: str, condition: Dict[str, Any], action: Dict[str, Any]):
        """添加路由规则"""
        self.routing_rules[rule_id] = {
            'condition': condition,
            'action': action,
            'created_at': datetime.now(),
            'hit_count': 0,
            'success_rate': 0.0
        }
    
    def evaluate_request(self, request_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """评估请求并返回路由决策"""
        for rule_id, rule in self.routing_rules.items():
            if self._match_condition(request_info, rule['condition']):
                rule['hit_count'] += 1
                return rule['action']
        
        return None
    
    def _match_condition(self, request_info: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """匹配条件"""
        for key, expected_value in condition.items():
            if key not in request_info:
                return False
            
            actual_value = request_info[key]
            
            if isinstance(expected_value, dict):
                # 支持范围条件
                if 'min' in expected_value and actual_value < expected_value['min']:
                    return False
                if 'max' in expected_value and actual_value > expected_value['max']:
                    return False
                if 'in' in expected_value and actual_value not in expected_value['in']:
                    return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True
    
    def update_performance(self, rule_id: str, success: bool, response_time: float):
        """更新性能指标"""
        if rule_id in self.routing_rules:
            rule = self.routing_rules[rule_id]
            
            # 更新成功率
            if 'success_count' not in rule:
                rule['success_count'] = 0
                rule['total_count'] = 0
            
            rule['total_count'] += 1
            if success:
                rule['success_count'] += 1
            
            rule['success_rate'] = rule['success_count'] / rule['total_count']
            
            # 记录性能历史
            self.performance_history[rule_id].append({
                'timestamp': datetime.now(),
                'success': success,
                'response_time': response_time
            })
            
            # 保持最近1000条记录
            if len(self.performance_history[rule_id]) > 1000:
                self.performance_history[rule_id] = self.performance_history[rule_id][-1000:]

class TrafficAnalyzer:
    """流量分析器"""
    
    def __init__(self, window_size: int = 300):  # 5分钟窗口
        self.window_size = window_size
        self.request_history = defaultdict(deque)
        self.patterns = {}
        self.anomalies = []
    
    def record_request(self, service_type: ServiceType, request_info: Dict[str, Any]):
        """记录请求"""
        timestamp = datetime.now()
        
        self.request_history[service_type].append({
            'timestamp': timestamp,
            'info': request_info
        })
        
        # 清理过期数据
        cutoff_time = timestamp - timedelta(seconds=self.window_size)
        while (self.request_history[service_type] and 
               self.request_history[service_type][0]['timestamp'] < cutoff_time):
            self.request_history[service_type].popleft()
    
    def analyze_patterns(self) -> Dict[ServiceType, TrafficPattern]:
        """分析流量模式"""
        patterns = {}
        
        for service_type, history in self.request_history.items():
            if len(history) < 10:  # 数据不足
                continue
            
            # 计算QPS
            time_span = (history[-1]['timestamp'] - history[0]['timestamp']).total_seconds()
            if time_span > 0:
                qps = len(history) / time_span
            else:
                qps = 0
            
            # 分析时间分布
            timestamps = [req['timestamp'] for req in history]
            time_intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                time_intervals.append(interval)
            
            # 检测异常
            if time_intervals:
                avg_interval = statistics.mean(time_intervals)
                std_interval = statistics.stdev(time_intervals) if len(time_intervals) > 1 else 0
                
                # 检测突发流量
                recent_intervals = time_intervals[-10:]  # 最近10个间隔
                if recent_intervals:
                    recent_avg = statistics.mean(recent_intervals)
                    if recent_avg < avg_interval - 2 * std_interval:  # 间隔显著减少，表示流量突增
                        self._record_anomaly(service_type, "traffic_burst", {
                            'current_qps': qps,
                            'avg_interval': avg_interval,
                            'recent_avg': recent_avg
                        })
            
            pattern = TrafficPattern(
                pattern_id=f"{service_type.value}_{int(time.time())}",
                service_type=service_type,
                time_window=timedelta(seconds=self.window_size),
                request_count=len(history),
                avg_response_time=0.0,  # 需要从其他地方获取
                peak_qps=qps,
                error_rate=0.0,  # 需要从其他地方获取
                resource_usage={},
                detected_at=datetime.now()
            )
            
            patterns[service_type] = pattern
        
        return patterns
    
    def _record_anomaly(self, service_type: ServiceType, anomaly_type: str, details: Dict[str, Any]):
        """记录异常"""
        anomaly = {
            'timestamp': datetime.now(),
            'service_type': service_type,
            'type': anomaly_type,
            'details': details
        }
        
        self.anomalies.append(anomaly)
        
        # 保持最近100个异常记录
        if len(self.anomalies) > 100:
            self.anomalies = self.anomalies[-100:]
        
        logger.warning(f"Traffic anomaly detected: {anomaly}")

class IntelligentLoadBalancer:
    """智能负载均衡器"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # 服务实例管理
        self.service_instances = defaultdict(list)  # service_type -> [ServiceInstance]
        self.instance_circuit_breakers = {}  # instance_id -> CircuitBreaker
        self.instance_locks = defaultdict(threading.Lock)
        
        # 负载均衡策略
        self.strategies = {
            ServiceType.VAD: LoadBalanceStrategy.LEAST_RESPONSE_TIME,
            ServiceType.ASR: LoadBalanceStrategy.RESOURCE_BASED,
            ServiceType.LLM: LoadBalanceStrategy.INTELLIGENT,
            ServiceType.TTS: LoadBalanceStrategy.LEAST_CONNECTIONS
        }
        
        # 路由和分析组件
        self.adaptive_router = AdaptiveRouter()
        self.traffic_analyzer = TrafficAnalyzer()
        
        # 指标收集
        self.request_metrics = deque(maxlen=10000)
        self.performance_stats = defaultdict(lambda: defaultdict(list))
        
        # 自动扩缩容
        self.scaling_rules = {}
        self.scaling_cooldown = {}
        
        # 健康检查
        self.health_check_interval = 30  # 秒
        self.health_check_timeout = 5    # 秒
        
        # 启动后台任务
        self.background_tasks = []
        self.is_running = False
    
    async def initialize(self):
        """初始化负载均衡器"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            logger.info("Load balancer Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
        
        # 启动后台任务
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._auto_scaling_loop()),
            asyncio.create_task(self._pattern_analysis_loop())
        ]
        
        logger.info("Intelligent load balancer initialized")
    
    async def shutdown(self):
        """关闭负载均衡器"""
        self.is_running = False
        
        # 取消后台任务
        for task in self.background_tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Load balancer shutdown completed")
    
    def register_service(self, instance: ServiceInstance):
        """注册服务实例"""
        self.service_instances[instance.service_type].append(instance)
        self.instance_circuit_breakers[instance.id] = CircuitBreaker()
        
        logger.info(f"Registered service instance: {instance.id} ({instance.service_type.value})")
    
    def unregister_service(self, instance_id: str, service_type: ServiceType):
        """注销服务实例"""
        instances = self.service_instances[service_type]
        self.service_instances[service_type] = [
            inst for inst in instances if inst.id != instance_id
        ]
        
        if instance_id in self.instance_circuit_breakers:
            del self.instance_circuit_breakers[instance_id]
        
        logger.info(f"Unregistered service instance: {instance_id}")
    
    async def route_request(self, service_type: ServiceType, request_info: Dict[str, Any]) -> Optional[ServiceInstance]:
        """路由请求到最佳服务实例"""
        # 记录流量
        self.traffic_analyzer.record_request(service_type, request_info)
        
        # 获取可用实例
        available_instances = await self._get_available_instances(service_type)
        if not available_instances:
            logger.warning(f"No available instances for service type: {service_type}")
            return None
        
        # 应用自适应路由规则
        routing_decision = self.adaptive_router.evaluate_request(request_info)
        if routing_decision:
            # 根据路由决策过滤实例
            filtered_instances = self._apply_routing_decision(available_instances, routing_decision)
            if filtered_instances:
                available_instances = filtered_instances
        
        # 选择负载均衡策略
        strategy = self.strategies.get(service_type, LoadBalanceStrategy.ROUND_ROBIN)
        
        # 执行负载均衡
        selected_instance = await self._select_instance(available_instances, strategy, request_info)
        
        if selected_instance:
            # 更新连接计数
            with self.instance_locks[selected_instance.id]:
                selected_instance.current_connections += 1
        
        return selected_instance
    
    async def record_request_result(self, instance: ServiceInstance, success: bool, 
                                  response_time: float, error_type: Optional[str] = None):
        """记录请求结果"""
        # 更新实例统计
        with self.instance_locks[instance.id]:
            instance.current_connections = max(0, instance.current_connections - 1)
            instance.total_requests += 1
            
            if success:
                instance.successful_requests += 1
                self.instance_circuit_breakers[instance.id].record_success()
            else:
                instance.failed_requests += 1
                self.instance_circuit_breakers[instance.id].record_failure()
            
            # 更新平均响应时间
            if instance.total_requests == 1:
                instance.avg_response_time = response_time
            else:
                # 指数移动平均
                alpha = 0.1
                instance.avg_response_time = (
                    alpha * response_time + (1 - alpha) * instance.avg_response_time
                )
        
        # 记录指标
        metrics = RequestMetrics(
            timestamp=datetime.now(),
            service_type=instance.service_type,
            instance_id=instance.id,
            response_time=response_time,
            success=success,
            error_type=error_type
        )
        self.request_metrics.append(metrics)
        
        # 更新性能统计
        self.performance_stats[instance.service_type][instance.id].append({
            'timestamp': datetime.now(),
            'response_time': response_time,
            'success': success
        })
    
    async def _get_available_instances(self, service_type: ServiceType) -> List[ServiceInstance]:
        """获取可用的服务实例"""
        instances = self.service_instances[service_type]
        available = []
        
        for instance in instances:
            # 检查健康状态
            if instance.health_status == HealthStatus.UNHEALTHY:
                continue
            
            # 检查熔断器状态
            circuit_breaker = self.instance_circuit_breakers.get(instance.id)
            if circuit_breaker and not circuit_breaker.can_execute():
                continue
            
            # 检查连接数限制
            if instance.current_connections >= instance.max_connections:
                continue
            
            available.append(instance)
        
        return available
    
    def _apply_routing_decision(self, instances: List[ServiceInstance], 
                              decision: Dict[str, Any]) -> List[ServiceInstance]:
        """应用路由决策"""
        if 'target_instances' in decision:
            target_ids = set(decision['target_instances'])
            return [inst for inst in instances if inst.id in target_ids]
        
        if 'exclude_instances' in decision:
            exclude_ids = set(decision['exclude_instances'])
            return [inst for inst in instances if inst.id not in exclude_ids]
        
        if 'resource_requirements' in decision:
            requirements = decision['resource_requirements']
            filtered = []
            
            for instance in instances:
                if ('min_cpu' in requirements and 
                    instance.cpu_usage > requirements['min_cpu']):
                    continue
                if ('max_cpu' in requirements and 
                    instance.cpu_usage > requirements['max_cpu']):
                    continue
                if ('min_memory' in requirements and 
                    instance.memory_usage < requirements['min_memory']):
                    continue
                if ('max_memory' in requirements and 
                    instance.memory_usage > requirements['max_memory']):
                    continue
                
                filtered.append(instance)
            
            return filtered
        
        return instances
    
    async def _select_instance(self, instances: List[ServiceInstance], 
                             strategy: LoadBalanceStrategy, 
                             request_info: Dict[str, Any]) -> Optional[ServiceInstance]:
        """根据策略选择实例"""
        if not instances:
            return None
        
        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(instances)
        
        elif strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(instances)
        
        elif strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(instances, key=lambda x: x.current_connections)
        
        elif strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
            return min(instances, key=lambda x: x.avg_response_time)
        
        elif strategy == LoadBalanceStrategy.RESOURCE_BASED:
            return min(instances, key=lambda x: x.load_score)
        
        elif strategy == LoadBalanceStrategy.INTELLIGENT:
            return await self._intelligent_select(instances, request_info)
        
        else:
            return instances[0]  # 默认选择第一个
    
    def _round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """轮询选择"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = {}
        
        service_type = instances[0].service_type
        if service_type not in self._round_robin_index:
            self._round_robin_index[service_type] = 0
        
        index = self._round_robin_index[service_type]
        selected = instances[index % len(instances)]
        self._round_robin_index[service_type] = (index + 1) % len(instances)
        
        return selected
    
    def _weighted_round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权轮询选择"""
        if not hasattr(self, '_weighted_counters'):
            self._weighted_counters = {}
        
        service_type = instances[0].service_type
        if service_type not in self._weighted_counters:
            self._weighted_counters[service_type] = {inst.id: 0 for inst in instances}
        
        # 更新计数器
        counters = self._weighted_counters[service_type]
        for instance in instances:
            if instance.id not in counters:
                counters[instance.id] = 0
            counters[instance.id] += instance.weight
        
        # 选择计数器最大的实例
        selected = max(instances, key=lambda x: counters.get(x.id, 0))
        
        # 重置选中实例的计数器
        counters[selected.id] = 0
        
        return selected
    
    async def _intelligent_select(self, instances: List[ServiceInstance], 
                                request_info: Dict[str, Any]) -> ServiceInstance:
        """智能选择"""
        # 综合考虑多个因素
        scores = {}
        
        for instance in instances:
            # 基础负载评分
            load_score = instance.load_score
            
            # 成功率评分
            success_score = 1.0 - instance.success_rate
            
            # 响应时间评分
            response_time_score = min(instance.avg_response_time / 1000, 1.0)
            
            # 历史性能评分
            history_score = await self._calculate_history_score(instance)
            
            # 请求匹配度评分
            match_score = self._calculate_request_match_score(instance, request_info)
            
            # 综合评分
            total_score = (
                load_score * 0.3 +
                success_score * 0.2 +
                response_time_score * 0.2 +
                history_score * 0.15 +
                match_score * 0.15
            )
            
            scores[instance.id] = total_score
        
        # 选择评分最低的实例（评分越低越好）
        best_instance = min(instances, key=lambda x: scores[x.id])
        return best_instance
    
    async def _calculate_history_score(self, instance: ServiceInstance) -> float:
        """计算历史性能评分"""
        history = self.performance_stats[instance.service_type].get(instance.id, [])
        
        if not history:
            return 0.5  # 中性评分
        
        # 只考虑最近的记录
        recent_history = [h for h in history if 
                         (datetime.now() - h['timestamp']).seconds < 300]  # 5分钟内
        
        if not recent_history:
            return 0.5
        
        # 计算平均响应时间和成功率
        avg_response_time = statistics.mean([h['response_time'] for h in recent_history])
        success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
        
        # 归一化评分
        response_time_score = min(avg_response_time / 1000, 1.0)
        success_score = 1.0 - success_rate
        
        return (response_time_score + success_score) / 2
    
    def _calculate_request_match_score(self, instance: ServiceInstance, 
                                     request_info: Dict[str, Any]) -> float:
        """计算请求匹配度评分"""
        # 这里可以根据请求特征和实例特征计算匹配度
        # 例如：GPU密集型请求匹配GPU实例
        
        score = 0.0
        
        # 检查GPU需求
        if request_info.get('requires_gpu', False):
            if instance.gpu_usage >= 0:  # 有GPU
                score += 0.5
            else:
                score += 1.0  # 惩罚没有GPU的实例
        
        # 检查内存需求
        memory_requirement = request_info.get('memory_mb', 0)
        if memory_requirement > 0:
            # 根据实例内存使用率评分
            memory_available = (100 - instance.memory_usage) / 100
            if memory_available > 0.5:
                score += 0.3
            elif memory_available > 0.2:
                score += 0.5
            else:
                score += 1.0  # 内存不足
        
        # 检查优先级
        priority = request_info.get('priority', 'normal')
        if priority == 'high' and instance.queue_length > 10:
            score += 0.5  # 高优先级请求避免排队长的实例
        
        return score
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self.is_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5)
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        tasks = []
        
        for service_type, instances in self.service_instances.items():
            for instance in instances:
                task = asyncio.create_task(self._check_instance_health(instance))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_instance_health(self, instance: ServiceInstance):
        """检查单个实例健康状态"""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)) as session:
                health_url = f"{instance.url}/health"
                
                async with session.get(health_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # 更新实例状态
                        instance.health_status = HealthStatus.HEALTHY
                        instance.last_health_check = datetime.now()
                        
                        # 更新资源使用情况
                        if 'cpu_usage' in data:
                            instance.cpu_usage = data['cpu_usage']
                        if 'memory_usage' in data:
                            instance.memory_usage = data['memory_usage']
                        if 'gpu_usage' in data:
                            instance.gpu_usage = data['gpu_usage']
                        if 'queue_length' in data:
                            instance.queue_length = data['queue_length']
                        
                    else:
                        instance.health_status = HealthStatus.DEGRADED
                        
        except Exception as e:
            logger.warning(f"Health check failed for {instance.id}: {e}")
            instance.health_status = HealthStatus.UNHEALTHY
            instance.last_health_check = datetime.now()
    
    async def _metrics_collection_loop(self):
        """指标收集循环"""
        while self.is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(60)  # 每分钟收集一次
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(10)
    
    async def _collect_metrics(self):
        """收集指标"""
        if not self.redis_client:
            return
        
        # 收集全局指标
        total_requests = len(self.request_metrics)
        successful_requests = sum(1 for m in self.request_metrics if m.success)
        
        global_metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'success_rate': successful_requests / max(total_requests, 1),
            'avg_response_time': statistics.mean([m.response_time for m in self.request_metrics]) if self.request_metrics else 0,
            'active_instances': sum(len(instances) for instances in self.service_instances.values()),
            'healthy_instances': sum(
                1 for instances in self.service_instances.values()
                for instance in instances
                if instance.health_status == HealthStatus.HEALTHY
            )
        }
        
        # 存储到Redis
        await self.redis_client.lpush('lb_global_metrics', json.dumps(global_metrics))
        await self.redis_client.ltrim('lb_global_metrics', 0, 1000)  # 保持最近1000条记录
        
        # 收集服务级别指标
        for service_type, instances in self.service_instances.items():
            service_metrics = {
                'timestamp': datetime.now().isoformat(),
                'service_type': service_type.value,
                'instance_count': len(instances),
                'healthy_count': sum(1 for inst in instances if inst.health_status == HealthStatus.HEALTHY),
                'total_connections': sum(inst.current_connections for inst in instances),
                'avg_load_score': statistics.mean([inst.load_score for inst in instances]) if instances else 0,
                'instances': [
                    {
                        'id': inst.id,
                        'health_status': inst.health_status.value,
                        'current_connections': inst.current_connections,
                        'success_rate': inst.success_rate,
                        'avg_response_time': inst.avg_response_time,
                        'load_score': inst.load_score,
                        'cpu_usage': inst.cpu_usage,
                        'memory_usage': inst.memory_usage,
                        'gpu_usage': inst.gpu_usage,
                        'queue_length': inst.queue_length
                    }
                    for inst in instances
                ]
            }
            
            key = f'lb_service_metrics:{service_type.value}'
            await self.redis_client.lpush(key, json.dumps(service_metrics))
            await self.redis_client.ltrim(key, 0, 1000)
    
    async def _auto_scaling_loop(self):
        """自动扩缩容循环"""
        while self.is_running:
            try:
                await self._check_scaling_conditions()
                await asyncio.sleep(120)  # 每2分钟检查一次
            except Exception as e:
                logger.error(f"Auto scaling loop error: {e}")
                await asyncio.sleep(30)
    
    async def _check_scaling_conditions(self):
        """检查扩缩容条件"""
        for service_type, instances in self.service_instances.items():
            if not instances:
                continue
            
            # 计算平均负载
            avg_load = statistics.mean([inst.load_score for inst in instances])
            healthy_count = sum(1 for inst in instances if inst.health_status == HealthStatus.HEALTHY)
            
            # 扩容条件
            if avg_load > 0.8 and healthy_count > 0:  # 平均负载超过80%
                await self._trigger_scale_out(service_type, instances)
            
            # 缩容条件
            elif avg_load < 0.3 and healthy_count > 2:  # 平均负载低于30%且实例数大于2
                await self._trigger_scale_in(service_type, instances)
    
    async def _trigger_scale_out(self, service_type: ServiceType, instances: List[ServiceInstance]):
        """触发扩容"""
        # 检查冷却时间
        cooldown_key = f"scale_out_{service_type.value}"
        if cooldown_key in self.scaling_cooldown:
            last_scaling = self.scaling_cooldown[cooldown_key]
            if (datetime.now() - last_scaling).seconds < 300:  # 5分钟冷却
                return
        
        logger.info(f"Triggering scale out for {service_type.value}")
        
        # 这里应该调用Kubernetes API或其他编排系统进行扩容
        # 暂时只记录日志
        
        self.scaling_cooldown[cooldown_key] = datetime.now()
        
        # 发送扩容事件到Redis
        if self.redis_client:
            event = {
                'timestamp': datetime.now().isoformat(),
                'action': 'scale_out',
                'service_type': service_type.value,
                'current_instances': len(instances),
                'reason': 'high_load'
            }
            await self.redis_client.lpush('lb_scaling_events', json.dumps(event))
    
    async def _trigger_scale_in(self, service_type: ServiceType, instances: List[ServiceInstance]):
        """触发缩容"""
        # 检查冷却时间
        cooldown_key = f"scale_in_{service_type.value}"
        if cooldown_key in self.scaling_cooldown:
            last_scaling = self.scaling_cooldown[cooldown_key]
            if (datetime.now() - last_scaling).seconds < 600:  # 10分钟冷却
                return
        
        logger.info(f"Triggering scale in for {service_type.value}")
        
        # 选择要移除的实例（负载最低的）
        candidate = min(instances, key=lambda x: x.load_score)
        
        # 这里应该调用Kubernetes API或其他编排系统进行缩容
        # 暂时只记录日志
        
        self.scaling_cooldown[cooldown_key] = datetime.now()
        
        # 发送缩容事件到Redis
        if self.redis_client:
            event = {
                'timestamp': datetime.now().isoformat(),
                'action': 'scale_in',
                'service_type': service_type.value,
                'current_instances': len(instances),
                'target_instance': candidate.id,
                'reason': 'low_load'
            }
            await self.redis_client.lpush('lb_scaling_events', json.dumps(event))
    
    async def _pattern_analysis_loop(self):
        """模式分析循环"""
        while self.is_running:
            try:
                patterns = self.traffic_analyzer.analyze_patterns()
                
                # 根据模式调整路由规则
                await self._adapt_routing_rules(patterns)
                
                await asyncio.sleep(300)  # 每5分钟分析一次
            except Exception as e:
                logger.error(f"Pattern analysis loop error: {e}")
                await asyncio.sleep(60)
    
    async def _adapt_routing_rules(self, patterns: Dict[ServiceType, TrafficPattern]):
        """根据流量模式调整路由规则"""
        for service_type, pattern in patterns.items():
            # 如果检测到高QPS，添加负载分散规则
            if pattern.peak_qps > 100:  # QPS阈值
                rule_id = f"high_qps_{service_type.value}"
                self.adaptive_router.add_routing_rule(
                    rule_id,
                    condition={'service_type': service_type.value, 'qps': {'min': 50}},
                    action={'strategy': 'distribute_load', 'weight_factor': 0.8}
                )
            
            # 如果检测到错误率高，添加故障转移规则
            if pattern.error_rate > 0.1:  # 10%错误率阈值
                rule_id = f"high_error_{service_type.value}"
                self.adaptive_router.add_routing_rule(
                    rule_id,
                    condition={'service_type': service_type.value, 'error_rate': {'min': 0.05}},
                    action={'strategy': 'failover', 'exclude_unhealthy': True}
                )
    
    def get_status(self) -> Dict[str, Any]:
        """获取负载均衡器状态"""
        return {
            'is_running': self.is_running,
            'total_services': sum(len(instances) for instances in self.service_instances.values()),
            'healthy_services': sum(
                1 for instances in self.service_instances.values()
                for instance in instances
                if instance.health_status == HealthStatus.HEALTHY
            ),
            'service_breakdown': {
                service_type.value: {
                    'total': len(instances),
                    'healthy': sum(1 for inst in instances if inst.health_status == HealthStatus.HEALTHY),
                    'avg_load': statistics.mean([inst.load_score for inst in instances]) if instances else 0
                }
                for service_type, instances in self.service_instances.items()
            },
            'recent_requests': len(self.request_metrics),
            'routing_rules': len(self.adaptive_router.routing_rules),
            'traffic_anomalies': len(self.traffic_analyzer.anomalies)
        }

# FastAPI应用
app = FastAPI(title="Intelligent Load Balancer", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局负载均衡器实例
load_balancer = IntelligentLoadBalancer()

@app.on_event("startup")
async def startup_event():
    await load_balancer.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await load_balancer.shutdown()

@app.post("/register")
async def register_service(instance_data: Dict[str, Any]):
    """注册服务实例"""
    try:
        instance = ServiceInstance(
            id=instance_data['id'],
            service_type=ServiceType(instance_data['service_type']),
            host=instance_data['host'],
            port=instance_data['port'],
            weight=instance_data.get('weight', 1.0),
            max_connections=instance_data.get('max_connections', 100),
            metadata=instance_data.get('metadata', {})
        )
        
        load_balancer.register_service(instance)
        
        return {"status": "success", "message": f"Service {instance.id} registered"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/unregister/{service_type}/{instance_id}")
async def unregister_service(service_type: str, instance_id: str):
    """注销服务实例"""
    try:
        service_type_enum = ServiceType(service_type)
        load_balancer.unregister_service(instance_id, service_type_enum)
        
        return {"status": "success", "message": f"Service {instance_id} unregistered"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/route/{service_type}")
async def route_request(service_type: str, request_info: Dict[str, Any]):
    """路由请求"""
    try:
        service_type_enum = ServiceType(service_type)
        instance = await load_balancer.route_request(service_type_enum, request_info)
        
        if instance:
            return {
                "status": "success",
                "instance": {
                    "id": instance.id,
                    "url": instance.url,
                    "load_score": instance.load_score
                }
            }
        else:
            raise HTTPException(status_code=503, detail="No available instances")
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid service type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/record_result")
async def record_result(result_data: Dict[str, Any]):
    """记录请求结果"""
    try:
        # 查找实例
        instance_id = result_data['instance_id']
        service_type = ServiceType(result_data['service_type'])
        
        instance = None
        for inst in load_balancer.service_instances[service_type]:
            if inst.id == instance_id:
                instance = inst
                break
        
        if not instance:
            raise HTTPException(status_code=404, detail="Instance not found")
        
        await load_balancer.record_request_result(
            instance=instance,
            success=result_data['success'],
            response_time=result_data['response_time'],
            error_type=result_data.get('error_type')
        )
        
        return {"status": "success", "message": "Result recorded"}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/status")
async def get_status():
    """获取负载均衡器状态"""
    return load_balancer.get_status()

@app.get("/metrics")
async def get_metrics():
    """获取指标"""
    if not load_balancer.redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        # 获取全局指标
        global_metrics = await load_balancer.redis_client.lrange('lb_global_metrics', 0, 9)
        global_metrics = [json.loads(m) for m in global_metrics]
        
        # 获取服务指标
        service_metrics = {}
        for service_type in ServiceType:
            key = f'lb_service_metrics:{service_type.value}'
            metrics = await load_balancer.redis_client.lrange(key, 0, 9)
            service_metrics[service_type.value] = [json.loads(m) for m in metrics]
        
        return {
            "global_metrics": global_metrics,
            "service_metrics": service_metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "load_balancer_running": load_balancer.is_running
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)