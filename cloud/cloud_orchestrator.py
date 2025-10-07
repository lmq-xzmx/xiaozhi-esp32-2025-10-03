#!/usr/bin/env python3
"""
Xiaozhi ESP32 Server - 云端集群编排器
负责管理边缘节点、LLM服务集群和全局资源调度
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib

import aiohttp
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import kubernetes
from kubernetes import client, config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"

class LLMProvider(Enum):
    QWEN = "qwen"
    BAICHUAN = "baichuan"
    LOCAL = "local"

@dataclass
class EdgeNodeInfo:
    """边缘节点信息"""
    node_id: str
    region: str
    status: str
    capacity: Dict[str, Any]
    current_load: Dict[str, Any]
    last_heartbeat: datetime
    uptime: float
    active_devices: int
    services_health: Dict[str, str]

@dataclass
class LLMInstance:
    """LLM实例信息"""
    instance_id: str
    provider: LLMProvider
    model_name: str
    status: str
    capacity: Dict[str, Any]
    current_load: Dict[str, Any]
    endpoint: str
    priority: int

class ChatRequest(BaseModel):
    """聊天请求"""
    session_id: str
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False

class ChatResponse(BaseModel):
    """聊天响应"""
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    processing_time: float
    tokens_used: int = 0

class ComplexityAnalyzer:
    """复杂度分析器 - 判断请求应该使用边缘模型还是云端模型"""
    
    def __init__(self):
        self.complexity_keywords = {
            'high': ['分析', '推理', '计算', '解释', '详细', '复杂', '深入', '专业'],
            'medium': ['比较', '总结', '建议', '方案', '步骤'],
            'low': ['你好', '谢谢', '再见', '是的', '不是', '好的']
        }
        
        self.context_weights = {
            'conversation_length': 0.3,
            'technical_terms': 0.4,
            'question_complexity': 0.3
        }
    
    def analyze_complexity(self, messages: List[Dict[str, str]], session_context: Dict = None) -> float:
        """分析请求复杂度 (0-1, 越高越复杂)"""
        if not messages:
            return 0.0
        
        latest_message = messages[-1].get('content', '')
        
        # 文本长度因子
        length_factor = min(len(latest_message) / 200, 1.0)
        
        # 关键词复杂度
        keyword_factor = self._analyze_keywords(latest_message)
        
        # 对话历史因子
        context_factor = min(len(messages) / 10, 1.0)
        
        # 技术术语因子
        technical_factor = self._analyze_technical_terms(latest_message)
        
        # 综合评分
        complexity = (
            length_factor * 0.2 +
            keyword_factor * 0.3 +
            context_factor * 0.2 +
            technical_factor * 0.3
        )
        
        return min(complexity, 1.0)
    
    def _analyze_keywords(self, text: str) -> float:
        """分析关键词复杂度"""
        text_lower = text.lower()
        
        high_count = sum(1 for keyword in self.complexity_keywords['high'] if keyword in text_lower)
        medium_count = sum(1 for keyword in self.complexity_keywords['medium'] if keyword in text_lower)
        low_count = sum(1 for keyword in self.complexity_keywords['low'] if keyword in text_lower)
        
        if high_count > 0:
            return 0.8 + min(high_count * 0.1, 0.2)
        elif medium_count > 0:
            return 0.5 + min(medium_count * 0.1, 0.3)
        elif low_count > 0:
            return max(0.1, 0.3 - low_count * 0.1)
        else:
            return 0.5  # 默认中等复杂度
    
    def _analyze_technical_terms(self, text: str) -> float:
        """分析技术术语密度"""
        technical_patterns = [
            'API', 'HTTP', 'JSON', 'SQL', 'Python', 'JavaScript',
            '算法', '数据结构', '机器学习', '深度学习', '神经网络',
            '服务器', '数据库', '缓存', '负载均衡', '微服务'
        ]
        
        text_lower = text.lower()
        technical_count = sum(1 for term in technical_patterns if term.lower() in text_lower)
        
        return min(technical_count / 5, 1.0)

class LLMLoadBalancer:
    """LLM负载均衡器"""
    
    def __init__(self):
        self.instances = {
            LLMProvider.QWEN: [],
            LLMProvider.BAICHUAN: [],
            LLMProvider.LOCAL: []
        }
        self.circuit_breakers = {}
        self.request_history = {}
        self.complexity_analyzer = ComplexityAnalyzer()
    
    async def route_request(self, request: ChatRequest, session_context: Dict = None) -> LLMInstance:
        """路由LLM请求"""
        # 分析复杂度
        complexity = self.complexity_analyzer.analyze_complexity(request.messages, session_context)
        
        # 选择合适的提供商
        provider = self._select_provider(complexity, request)
        
        # 选择最优实例
        instance = await self._select_best_instance(provider, request)
        
        if not instance:
            # 降级到其他提供商
            for fallback_provider in [LLMProvider.QWEN, LLMProvider.BAICHUAN, LLMProvider.LOCAL]:
                if fallback_provider != provider:
                    instance = await self._select_best_instance(fallback_provider, request)
                    if instance:
                        break
        
        if not instance:
            raise HTTPException(status_code=503, detail="No LLM instances available")
        
        return instance
    
    def _select_provider(self, complexity: float, request: ChatRequest) -> LLMProvider:
        """根据复杂度选择提供商"""
        # 指定模型的情况
        if request.model:
            if 'qwen' in request.model.lower():
                return LLMProvider.QWEN
            elif 'baichuan' in request.model.lower():
                return LLMProvider.BAICHUAN
            elif 'local' in request.model.lower():
                return LLMProvider.LOCAL
        
        # 根据复杂度自动选择
        if complexity > 0.7:
            # 高复杂度：优先使用大模型
            return LLMProvider.QWEN
        elif complexity > 0.4:
            # 中等复杂度：使用中等模型
            return LLMProvider.BAICHUAN
        else:
            # 低复杂度：使用本地模型
            return LLMProvider.LOCAL
    
    async def _select_best_instance(self, provider: LLMProvider, request: ChatRequest) -> Optional[LLMInstance]:
        """选择最优实例"""
        instances = self.instances.get(provider, [])
        if not instances:
            return None
        
        # 过滤健康的实例
        healthy_instances = [
            inst for inst in instances 
            if inst.status == 'healthy' and not self._is_circuit_open(inst.instance_id)
        ]
        
        if not healthy_instances:
            return None
        
        # 计算实例评分
        scores = {}
        for instance in healthy_instances:
            score = self._calculate_instance_score(instance, request)
            scores[instance.instance_id] = score
        
        # 选择最高评分的实例
        best_instance_id = max(scores.keys(), key=lambda k: scores[k])
        return next(inst for inst in healthy_instances if inst.instance_id == best_instance_id)
    
    def _calculate_instance_score(self, instance: LLMInstance, request: ChatRequest) -> float:
        """计算实例评分"""
        load = instance.current_load
        
        # 基础评分因子
        factors = {
            'cpu_usage': (100 - load.get('cpu', 50)) / 100 * 0.3,
            'memory_usage': (100 - load.get('memory', 50)) / 100 * 0.2,
            'gpu_usage': (100 - load.get('gpu', 50)) / 100 * 0.3,
            'queue_length': max(0, (20 - load.get('queue', 0)) / 20) * 0.2
        }
        
        base_score = sum(factors.values())
        
        # 优先级加权
        priority_weight = instance.priority / 10.0
        
        return base_score * (1 + priority_weight)
    
    def _is_circuit_open(self, instance_id: str) -> bool:
        """检查熔断器状态"""
        if instance_id not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[instance_id]
        return breaker.get('open', False)
    
    async def update_instances(self, provider: LLMProvider, instances: List[LLMInstance]):
        """更新实例列表"""
        self.instances[provider] = instances
        logger.info(f"Updated {provider.value} instances: {len(instances)}")

class GlobalResourceManager:
    """全局资源管理器"""
    
    def __init__(self):
        self.edge_nodes = {}
        self.total_capacity = {
            'devices': 0,
            'cpu_cores': 0,
            'memory_gb': 0,
            'gpu_count': 0
        }
        self.current_usage = {
            'active_devices': 0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'gpu_usage': 0.0
        }
        
        self.scaling_thresholds = {
            'scale_up_cpu': 80,
            'scale_up_memory': 85,
            'scale_up_devices': 90,
            'scale_down_cpu': 30,
            'scale_down_memory': 40,
            'scale_down_devices': 50
        }
    
    async def register_edge_node(self, node_info: EdgeNodeInfo):
        """注册边缘节点"""
        self.edge_nodes[node_info.node_id] = node_info
        await self._update_total_capacity()
        logger.info(f"Registered edge node: {node_info.node_id}")
    
    async def update_node_status(self, node_id: str, status_data: Dict[str, Any]):
        """更新节点状态"""
        if node_id in self.edge_nodes:
            node = self.edge_nodes[node_id]
            node.current_load = status_data.get('current_load', {})
            node.last_heartbeat = datetime.now()
            node.active_devices = status_data.get('active_devices', 0)
            node.services_health = status_data.get('services_health', {})
            
            await self._update_current_usage()
    
    async def _update_total_capacity(self):
        """更新总容量"""
        self.total_capacity = {
            'devices': sum(node.capacity.get('max_devices', 0) for node in self.edge_nodes.values()),
            'cpu_cores': sum(node.capacity.get('cpu_cores', 0) for node in self.edge_nodes.values()),
            'memory_gb': sum(node.capacity.get('memory_gb', 0) for node in self.edge_nodes.values()),
            'gpu_count': sum(node.capacity.get('gpu_count', 0) for node in self.edge_nodes.values())
        }
    
    async def _update_current_usage(self):
        """更新当前使用情况"""
        if not self.edge_nodes:
            return
        
        total_devices = sum(node.active_devices for node in self.edge_nodes.values())
        avg_cpu = sum(node.current_load.get('cpu_usage', 0) for node in self.edge_nodes.values()) / len(self.edge_nodes)
        avg_memory = sum(node.current_load.get('memory_usage', 0) for node in self.edge_nodes.values()) / len(self.edge_nodes)
        avg_gpu = sum(node.current_load.get('gpu_usage', 0) for node in self.edge_nodes.values()) / len(self.edge_nodes)
        
        self.current_usage = {
            'active_devices': total_devices,
            'cpu_usage': avg_cpu,
            'memory_usage': avg_memory,
            'gpu_usage': avg_gpu
        }
    
    async def check_scaling_needs(self) -> Dict[str, Any]:
        """检查是否需要扩缩容"""
        recommendations = {
            'scale_up': [],
            'scale_down': [],
            'alerts': []
        }
        
        # 检查设备容量
        device_usage_percent = (self.current_usage['active_devices'] / max(self.total_capacity['devices'], 1)) * 100
        
        if device_usage_percent > self.scaling_thresholds['scale_up_devices']:
            recommendations['scale_up'].append({
                'resource': 'devices',
                'current_usage': device_usage_percent,
                'threshold': self.scaling_thresholds['scale_up_devices'],
                'action': 'add_edge_node'
            })
        
        # 检查CPU使用率
        if self.current_usage['cpu_usage'] > self.scaling_thresholds['scale_up_cpu']:
            recommendations['scale_up'].append({
                'resource': 'cpu',
                'current_usage': self.current_usage['cpu_usage'],
                'threshold': self.scaling_thresholds['scale_up_cpu'],
                'action': 'scale_up_services'
            })
        
        # 检查内存使用率
        if self.current_usage['memory_usage'] > self.scaling_thresholds['scale_up_memory']:
            recommendations['scale_up'].append({
                'resource': 'memory',
                'current_usage': self.current_usage['memory_usage'],
                'threshold': self.scaling_thresholds['scale_up_memory'],
                'action': 'scale_up_services'
            })
        
        return recommendations
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """获取集群健康状态"""
        healthy_nodes = sum(1 for node in self.edge_nodes.values() if node.status == 'healthy')
        total_nodes = len(self.edge_nodes)
        
        if total_nodes == 0:
            cluster_status = ClusterStatus.CRITICAL
        elif healthy_nodes / total_nodes >= 0.8:
            cluster_status = ClusterStatus.HEALTHY
        elif healthy_nodes / total_nodes >= 0.5:
            cluster_status = ClusterStatus.DEGRADED
        else:
            cluster_status = ClusterStatus.CRITICAL
        
        return {
            'status': cluster_status.value,
            'total_nodes': total_nodes,
            'healthy_nodes': healthy_nodes,
            'total_capacity': self.total_capacity,
            'current_usage': self.current_usage,
            'usage_percentage': {
                'devices': (self.current_usage['active_devices'] / max(self.total_capacity['devices'], 1)) * 100,
                'cpu': self.current_usage['cpu_usage'],
                'memory': self.current_usage['memory_usage'],
                'gpu': self.current_usage['gpu_usage']
            }
        }

class CloudOrchestrator:
    """云端编排器主类"""
    
    def __init__(self):
        self.resource_manager = GlobalResourceManager()
        self.llm_balancer = LLMLoadBalancer()
        self.redis_client = None
        self.k8s_client = None
        
        self.session_contexts = {}  # 会话上下文缓存
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_latency': 0.0
        }
        
    async def initialize(self):
        """初始化编排器"""
        # 初始化Redis
        try:
            self.redis_client = await aioredis.from_url("redis://localhost:6379")
            logger.info("Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
        
        # 初始化Kubernetes客户端
        try:
            config.load_incluster_config()  # 在集群内运行
            self.k8s_client = client.AppsV1Api()
            logger.info("Kubernetes client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Kubernetes client: {e}")
        
        # 启动后台任务
        asyncio.create_task(self._resource_monitoring_loop())
        asyncio.create_task(self._auto_scaling_loop())
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._cleanup_loop())
        
        # 初始化LLM实例
        await self._discover_llm_instances()
        
        logger.info("Cloud orchestrator initialized")
    
    async def process_chat_request(self, request: ChatRequest) -> ChatResponse:
        """处理聊天请求"""
        start_time = time.time()
        self.request_stats['total_requests'] += 1
        
        try:
            # 获取会话上下文
            session_context = await self._get_session_context(request.session_id)
            
            # 路由到最优LLM实例
            llm_instance = await self.llm_balancer.route_request(request, session_context)
            
            # 处理请求
            response = await self._process_with_llm(llm_instance, request)
            
            # 更新会话上下文
            await self._update_session_context(request.session_id, request.messages, response.response)
            
            # 更新统计
            processing_time = time.time() - start_time
            self.request_stats['successful_requests'] += 1
            self._update_latency_stats(processing_time)
            
            response.processing_time = processing_time
            return response
            
        except Exception as e:
            logger.error(f"Chat request processing error: {e}")
            self.request_stats['failed_requests'] += 1
            
            return ChatResponse(
                success=False,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    async def _process_with_llm(self, instance: LLMInstance, request: ChatRequest) -> ChatResponse:
        """使用指定LLM实例处理请求"""
        try:
            # 构建请求数据
            llm_request = {
                'messages': request.messages,
                'temperature': request.temperature,
                'max_tokens': request.max_tokens,
                'stream': request.stream
            }
            
            # 发送请求到LLM实例
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{instance.endpoint}/chat/completions",
                    json=llm_request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return ChatResponse(
                            success=True,
                            response=result.get('response', ''),
                            model_used=instance.model_name,
                            processing_time=0.0,  # 将在外层设置
                            tokens_used=result.get('tokens_used', 0)
                        )
                    else:
                        error_text = await response.text()
                        return ChatResponse(
                            success=False,
                            error=f"LLM request failed: {response.status} - {error_text}",
                            processing_time=0.0
                        )
                        
        except Exception as e:
            return ChatResponse(
                success=False,
                error=f"LLM processing error: {str(e)}",
                processing_time=0.0
            )
    
    async def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """获取会话上下文"""
        if session_id in self.session_contexts:
            return self.session_contexts[session_id]
        
        # 从Redis获取
        if self.redis_client:
            try:
                context_data = await self.redis_client.get(f"session:{session_id}")
                if context_data:
                    context = json.loads(context_data)
                    self.session_contexts[session_id] = context
                    return context
            except Exception as e:
                logger.error(f"Failed to get session context from Redis: {e}")
        
        # 返回默认上下文
        return {
            'conversation_count': 0,
            'topics': [],
            'complexity_history': [],
            'last_activity': datetime.now().isoformat()
        }
    
    async def _update_session_context(self, session_id: str, messages: List[Dict], response: str):
        """更新会话上下文"""
        context = await self._get_session_context(session_id)
        
        # 更新上下文
        context['conversation_count'] += 1
        context['last_activity'] = datetime.now().isoformat()
        
        # 分析话题
        if messages:
            latest_message = messages[-1].get('content', '')
            topics = self._extract_topics(latest_message)
            context['topics'].extend(topics)
            context['topics'] = list(set(context['topics'][-10:]))  # 保留最近10个话题
        
        # 保存到内存和Redis
        self.session_contexts[session_id] = context
        
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"session:{session_id}",
                    3600,  # 1小时过期
                    json.dumps(context)
                )
            except Exception as e:
                logger.error(f"Failed to save session context to Redis: {e}")
    
    def _extract_topics(self, text: str) -> List[str]:
        """提取话题关键词"""
        # 简单的关键词提取，实际应用中可以使用更复杂的NLP技术
        keywords = []
        common_topics = [
            '技术', '编程', '人工智能', '机器学习', '数据分析',
            '产品', '设计', '管理', '营销', '财务',
            '健康', '教育', '娱乐', '旅游', '美食'
        ]
        
        for topic in common_topics:
            if topic in text:
                keywords.append(topic)
        
        return keywords
    
    def _update_latency_stats(self, latency: float):
        """更新延迟统计"""
        current_avg = self.request_stats['average_latency']
        total_requests = self.request_stats['successful_requests']
        
        if total_requests == 1:
            self.request_stats['average_latency'] = latency
        else:
            # 计算移动平均
            self.request_stats['average_latency'] = (current_avg * (total_requests - 1) + latency) / total_requests
    
    async def _discover_llm_instances(self):
        """发现LLM实例"""
        # 这里应该从Kubernetes API或服务注册中心获取实例
        # 暂时使用静态配置
        
        qwen_instances = [
            LLMInstance(
                instance_id="qwen-72b-1",
                provider=LLMProvider.QWEN,
                model_name="Qwen-72B-Chat",
                status="healthy",
                capacity={"max_concurrent": 32, "max_tokens": 8192},
                current_load={"cpu": 60, "memory": 70, "gpu": 65, "queue": 5},
                endpoint="http://qwen-service:8000",
                priority=9
            ),
            LLMInstance(
                instance_id="qwen-14b-1",
                provider=LLMProvider.QWEN,
                model_name="Qwen-14B-Chat",
                status="healthy",
                capacity={"max_concurrent": 64, "max_tokens": 4096},
                current_load={"cpu": 45, "memory": 55, "gpu": 50, "queue": 3},
                endpoint="http://qwen-service:8001",
                priority=7
            )
        ]
        
        baichuan_instances = [
            LLMInstance(
                instance_id="baichuan-13b-1",
                provider=LLMProvider.BAICHUAN,
                model_name="Baichuan2-13B-Chat",
                status="healthy",
                capacity={"max_concurrent": 48, "max_tokens": 4096},
                current_load={"cpu": 50, "memory": 60, "gpu": 55, "queue": 2},
                endpoint="http://baichuan-service:8000",
                priority=6
            )
        ]
        
        local_instances = [
            LLMInstance(
                instance_id="local-7b-1",
                provider=LLMProvider.LOCAL,
                model_name="Qwen-7B-Chat",
                status="healthy",
                capacity={"max_concurrent": 16, "max_tokens": 2048},
                current_load={"cpu": 30, "memory": 40, "gpu": 35, "queue": 1},
                endpoint="http://local-llm-service:8000",
                priority=4
            )
        ]
        
        await self.llm_balancer.update_instances(LLMProvider.QWEN, qwen_instances)
        await self.llm_balancer.update_instances(LLMProvider.BAICHUAN, baichuan_instances)
        await self.llm_balancer.update_instances(LLMProvider.LOCAL, local_instances)
        
        logger.info("LLM instances discovered and updated")
    
    async def _resource_monitoring_loop(self):
        """资源监控循环"""
        while True:
            try:
                # 收集资源使用情况
                await self._collect_resource_metrics()
                
                # 检查扩缩容需求
                scaling_recommendations = await self.resource_manager.check_scaling_needs()
                
                if scaling_recommendations['scale_up']:
                    logger.info(f"Scale up recommendations: {scaling_recommendations['scale_up']}")
                    # 这里可以触发自动扩容
                
                await asyncio.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _collect_resource_metrics(self):
        """收集资源指标"""
        # 这里应该从实际的监控系统收集指标
        # 暂时使用模拟数据
        pass
    
    async def _auto_scaling_loop(self):
        """自动扩缩容循环"""
        while True:
            try:
                if self.k8s_client:
                    await self._perform_auto_scaling()
                await asyncio.sleep(300)  # 每5分钟检查一次
            except Exception as e:
                logger.error(f"Auto scaling error: {e}")
                await asyncio.sleep(60)
    
    async def _perform_auto_scaling(self):
        """执行自动扩缩容"""
        # 获取扩缩容建议
        recommendations = await self.resource_manager.check_scaling_needs()
        
        for recommendation in recommendations['scale_up']:
            if recommendation['action'] == 'scale_up_services':
                await self._scale_up_services(recommendation['resource'])
    
    async def _scale_up_services(self, resource_type: str):
        """扩容服务"""
        try:
            # 扩容ASR服务（通常是瓶颈）
            deployment = self.k8s_client.read_namespaced_deployment(
                name="asr-service",
                namespace="xiaozhi-system"
            )
            
            current_replicas = deployment.spec.replicas
            new_replicas = min(current_replicas + 2, 12)  # 最多12个副本
            
            deployment.spec.replicas = new_replicas
            
            self.k8s_client.patch_namespaced_deployment(
                name="asr-service",
                namespace="xiaozhi-system",
                body=deployment
            )
            
            logger.info(f"Scaled up asr-service from {current_replicas} to {new_replicas} replicas")
            
        except Exception as e:
            logger.error(f"Failed to scale up services: {e}")
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                await self._check_edge_nodes_health()
                await self._check_llm_instances_health()
                await asyncio.sleep(30)  # 每30秒检查一次
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _check_edge_nodes_health(self):
        """检查边缘节点健康状态"""
        current_time = datetime.now()
        
        for node_id, node in self.resource_manager.edge_nodes.items():
            # 检查心跳超时
            if (current_time - node.last_heartbeat).seconds > 120:  # 2分钟超时
                node.status = 'offline'
                logger.warning(f"Edge node {node_id} is offline (last heartbeat: {node.last_heartbeat})")
    
    async def _check_llm_instances_health(self):
        """检查LLM实例健康状态"""
        for provider in LLMProvider:
            instances = self.llm_balancer.instances.get(provider, [])
            for instance in instances:
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            f"{instance.endpoint}/health",
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status == 200:
                                instance.status = 'healthy'
                            else:
                                instance.status = 'unhealthy'
                except Exception:
                    instance.status = 'unhealthy'
    
    async def _cleanup_loop(self):
        """清理循环"""
        while True:
            try:
                # 清理过期的会话上下文
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, context in self.session_contexts.items():
                    last_activity = datetime.fromisoformat(context['last_activity'])
                    if (current_time - last_activity).seconds > 3600:  # 1小时过期
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.session_contexts[session_id]
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                await asyncio.sleep(1800)  # 每30分钟清理一次
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)

# FastAPI应用
app = FastAPI(title="Xiaozhi Cloud Orchestrator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局编排器实例
orchestrator = None

@app.on_event("startup")
async def startup_event():
    global orchestrator
    orchestrator = CloudOrchestrator()
    await orchestrator.initialize()

@app.post("/api/edge/heartbeat")
async def edge_heartbeat(node_data: dict):
    """接收边缘节点心跳"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    node_info = EdgeNodeInfo(
        node_id=node_data['node_id'],
        region=node_data.get('region', 'default'),
        status=node_data.get('status', 'unknown'),
        capacity=node_data.get('capacity', {}),
        current_load=node_data.get('current_load', {}),
        last_heartbeat=datetime.now(),
        uptime=node_data.get('uptime', 0),
        active_devices=node_data.get('active_devices', 0),
        services_health=node_data.get('services_health', {})
    )
    
    await orchestrator.resource_manager.register_edge_node(node_info)
    return {"status": "ok"}

@app.post("/api/llm/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """LLM聊天完成"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return await orchestrator.process_chat_request(request)

@app.get("/api/cluster/status")
async def get_cluster_status():
    """获取集群状态"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return {
        "cluster_health": orchestrator.resource_manager.get_cluster_health(),
        "request_stats": orchestrator.request_stats,
        "edge_nodes": {
            node_id: asdict(node) 
            for node_id, node in orchestrator.resource_manager.edge_nodes.items()
        }
    }

@app.get("/api/cluster/metrics")
async def get_cluster_metrics():
    """获取集群指标"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    scaling_recommendations = await orchestrator.resource_manager.check_scaling_needs()
    
    return {
        "resource_usage": orchestrator.resource_manager.current_usage,
        "total_capacity": orchestrator.resource_manager.total_capacity,
        "scaling_recommendations": scaling_recommendations,
        "llm_instances": {
            provider.value: [asdict(inst) for inst in instances]
            for provider, instances in orchestrator.llm_balancer.instances.items()
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    if not orchestrator:
        return {"status": "unhealthy", "reason": "orchestrator not initialized"}
    
    cluster_health = orchestrator.resource_manager.get_cluster_health()
    
    return {
        "status": cluster_health['status'],
        "cluster_health": cluster_health,
        "uptime": time.time() - (orchestrator.start_time if hasattr(orchestrator, 'start_time') else time.time())
    }

if __name__ == "__main__":
    uvicorn.run(
        "cloud_orchestrator:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1
    )