#!/usr/bin/env python3
"""
智能负载均衡器 - 4核8GB服务器优化版本
支持动态负载分配、健康检查和性能监控
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psutil
import os

logger = logging.getLogger(__name__)

class ServiceType(Enum):
    VAD = "vad"
    ASR = "asr"
    LLM = "llm"
    TTS = "tts"

@dataclass
class ServiceInstance:
    """服务实例信息"""
    service_type: ServiceType
    instance_id: str
    endpoint: str
    weight: float = 1.0
    current_load: float = 0.0
    health_score: float = 1.0
    last_health_check: float = field(default_factory=time.time)
    response_times: List[float] = field(default_factory=list)
    error_count: int = 0
    total_requests: int = 0

    @property
    def avg_response_time(self) -> float:
        """平均响应时间"""
        if not self.response_times:
            return 0.0
        return sum(self.response_times[-10:]) / len(self.response_times[-10:])  # 最近10次的平均值

    @property
    def error_rate(self) -> float:
        """错误率"""
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests

    @property
    def effective_weight(self) -> float:
        """有效权重（考虑健康状态和负载）"""
        load_factor = max(0.1, 1.0 - self.current_load)  # 负载越高，权重越低
        health_factor = self.health_score
        return self.weight * load_factor * health_factor

class IntelligentLoadBalancer:
    """智能负载均衡器"""
    
    def __init__(self):
        self.services: Dict[ServiceType, List[ServiceInstance]] = {
            ServiceType.VAD: [],
            ServiceType.ASR: [],
            ServiceType.LLM: [],
            ServiceType.TTS: []
        }
        
        # 系统资源监控
        self.system_stats = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'load_average': 0.0,
            'network_io': 0.0
        }
        
        # 负载均衡策略配置
        self.config = {
            'health_check_interval': 30,  # 健康检查间隔（秒）
            'max_response_time': 2.0,     # 最大响应时间（秒）
            'max_error_rate': 0.1,        # 最大错误率
            'cpu_threshold': 0.85,        # CPU使用率阈值
            'memory_threshold': 0.80,     # 内存使用率阈值
            'load_balancing_algorithm': 'weighted_round_robin'  # 负载均衡算法
        }
        
        # 启动监控任务
        asyncio.create_task(self.system_monitor())
        asyncio.create_task(self.health_checker())
        
        logger.info("智能负载均衡器初始化完成")

    def register_service(self, service_type: ServiceType, instance_id: str, 
                        endpoint: str, weight: float = 1.0):
        """注册服务实例"""
        instance = ServiceInstance(
            service_type=service_type,
            instance_id=instance_id,
            endpoint=endpoint,
            weight=weight
        )
        
        self.services[service_type].append(instance)
        logger.info(f"注册服务实例: {service_type.value}:{instance_id} -> {endpoint}")

    async def get_best_instance(self, service_type: ServiceType) -> Optional[ServiceInstance]:
        """获取最佳服务实例"""
        instances = self.services.get(service_type, [])
        if not instances:
            return None
        
        # 过滤健康的实例
        healthy_instances = [
            inst for inst in instances 
            if inst.health_score > 0.5 and inst.error_rate < self.config['max_error_rate']
        ]
        
        if not healthy_instances:
            logger.warning(f"没有健康的{service_type.value}实例可用")
            return None
        
        # 根据配置的算法选择实例
        if self.config['load_balancing_algorithm'] == 'weighted_round_robin':
            return self._weighted_round_robin(healthy_instances)
        elif self.config['load_balancing_algorithm'] == 'least_connections':
            return self._least_connections(healthy_instances)
        else:
            return self._least_response_time(healthy_instances)

    def _weighted_round_robin(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权轮询算法"""
        total_weight = sum(inst.effective_weight for inst in instances)
        if total_weight == 0:
            return instances[0]
        
        # 简化的加权选择
        best_instance = max(instances, key=lambda x: x.effective_weight)
        return best_instance

    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少连接算法"""
        return min(instances, key=lambda x: x.current_load)

    def _least_response_time(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最短响应时间算法"""
        return min(instances, key=lambda x: x.avg_response_time or float('inf'))

    async def record_request(self, service_type: ServiceType, instance_id: str, 
                           response_time: float, success: bool = True):
        """记录请求结果"""
        for instance in self.services.get(service_type, []):
            if instance.instance_id == instance_id:
                instance.total_requests += 1
                instance.response_times.append(response_time)
                
                # 保持最近50次记录
                if len(instance.response_times) > 50:
                    instance.response_times = instance.response_times[-50:]
                
                if not success:
                    instance.error_count += 1
                
                # 更新健康分数
                self._update_health_score(instance)
                break

    def _update_health_score(self, instance: ServiceInstance):
        """更新健康分数"""
        # 基于响应时间和错误率计算健康分数
        response_time_score = max(0, 1.0 - (instance.avg_response_time / self.config['max_response_time']))
        error_rate_score = max(0, 1.0 - (instance.error_rate / self.config['max_error_rate']))
        
        # 综合健康分数
        instance.health_score = (response_time_score + error_rate_score) / 2

    async def system_monitor(self):
        """系统资源监控"""
        while True:
            try:
                # 获取系统资源使用情况
                self.system_stats['cpu_usage'] = psutil.cpu_percent(interval=1)
                self.system_stats['memory_usage'] = psutil.virtual_memory().percent / 100.0
                self.system_stats['load_average'] = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0
                
                # 网络IO统计
                net_io = psutil.net_io_counters()
                self.system_stats['network_io'] = (net_io.bytes_sent + net_io.bytes_recv) / 1024 / 1024  # MB
                
                # 根据系统负载调整服务权重
                await self._adjust_weights_by_system_load()
                
                await asyncio.sleep(10)  # 每10秒监控一次
                
            except Exception as e:
                logger.error(f"系统监控错误: {e}")
                await asyncio.sleep(30)

    async def _adjust_weights_by_system_load(self):
        """根据系统负载调整服务权重"""
        cpu_usage = self.system_stats['cpu_usage']
        memory_usage = self.system_stats['memory_usage']
        
        # 如果系统负载过高，降低所有服务的权重
        if cpu_usage > self.config['cpu_threshold'] or memory_usage > self.config['memory_threshold']:
            adjustment_factor = 0.7  # 降低30%的权重
            
            for service_type in self.services:
                for instance in self.services[service_type]:
                    instance.weight = max(0.1, instance.weight * adjustment_factor)
            
            logger.warning(f"系统负载过高 (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%), 调整服务权重")

    async def health_checker(self):
        """健康检查任务"""
        while True:
            try:
                for service_type in self.services:
                    for instance in self.services[service_type]:
                        await self._check_instance_health(instance)
                
                await asyncio.sleep(self.config['health_check_interval'])
                
            except Exception as e:
                logger.error(f"健康检查错误: {e}")
                await asyncio.sleep(60)

    async def _check_instance_health(self, instance: ServiceInstance):
        """检查单个实例健康状态"""
        try:
            # 简化的健康检查（实际应该发送HTTP请求）
            current_time = time.time()
            
            # 如果最近没有请求，保持当前健康分数
            if current_time - instance.last_health_check > 300:  # 5分钟无请求
                instance.health_score = max(0.5, instance.health_score * 0.9)
            
            instance.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"健康检查失败 {instance.instance_id}: {e}")
            instance.health_score = max(0.1, instance.health_score * 0.5)

    def get_stats(self) -> Dict:
        """获取负载均衡器统计信息"""
        stats = {
            'system_stats': self.system_stats,
            'services': {}
        }
        
        for service_type in self.services:
            service_stats = []
            for instance in self.services[service_type]:
                service_stats.append({
                    'instance_id': instance.instance_id,
                    'endpoint': instance.endpoint,
                    'weight': instance.weight,
                    'effective_weight': instance.effective_weight,
                    'current_load': instance.current_load,
                    'health_score': instance.health_score,
                    'avg_response_time': instance.avg_response_time,
                    'error_rate': instance.error_rate,
                    'total_requests': instance.total_requests
                })
            
            stats['services'][service_type.value] = service_stats
        
        return stats

# 全局负载均衡器实例
load_balancer = IntelligentLoadBalancer()

# 注册默认服务实例
load_balancer.register_service(ServiceType.VAD, "vad-1", "http://localhost:8000/vad", weight=1.0)
load_balancer.register_service(ServiceType.ASR, "asr-1", "http://localhost:8001/asr", weight=1.0)
load_balancer.register_service(ServiceType.LLM, "llm-1", "http://localhost:8000/llm", weight=1.0)
load_balancer.register_service(ServiceType.TTS, "tts-1", "http://localhost:8000/tts", weight=1.0)

logger.info("智能负载均衡器配置完成")