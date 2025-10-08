"""
智能路由器
基于负载、成本、延迟、成功率等多维度指标进行API智能选择
"""

import asyncio
import time
import random
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from config.remote_api_config import APIEndpoint, APIProvider, remote_api_config

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """路由策略"""
    ROUND_ROBIN = "round_robin"           # 轮询
    WEIGHTED_ROUND_ROBIN = "weighted_rr"  # 加权轮询
    LEAST_CONNECTIONS = "least_conn"      # 最少连接
    FASTEST_RESPONSE = "fastest"          # 最快响应
    COST_OPTIMIZED = "cost_optimized"     # 成本优化
    INTELLIGENT = "intelligent"           # 智能路由

class RequestPriority(Enum):
    """请求优先级"""
    CRITICAL = 1    # 关键请求，优先选择最快最稳定的API
    HIGH = 2        # 高优先级，平衡速度和成本
    MEDIUM = 3      # 中优先级，平衡各项指标
    LOW = 4         # 低优先级，优先选择成本最低的API

@dataclass
class RoutingContext:
    """路由上下文"""
    request_id: str
    priority: RequestPriority
    text_length: int = 0
    expected_response_length: int = 0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timeout: float = 30.0
    cost_budget: float = 0.0  # 成本预算
    quality_requirement: float = 0.8  # 质量要求 (0-1)

class IntelligentRouter:
    """智能路由器"""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT):
        self.strategy = strategy
        self.round_robin_counters = {}  # 轮询计数器
        self.request_history = {}  # 请求历史
        self.load_stats = {}  # 负载统计
        
        # 路由权重配置
        self.routing_weights = {
            "latency": 0.3,      # 延迟权重
            "success_rate": 0.25, # 成功率权重
            "cost": 0.2,         # 成本权重
            "load": 0.15,        # 负载权重
            "quality": 0.1       # 质量权重
        }
    
    async def select_llm_endpoint(self, context: RoutingContext) -> Optional[APIEndpoint]:
        """选择LLM端点"""
        endpoints = remote_api_config.get_llm_endpoints(healthy_only=True)
        if not endpoints:
            logger.error("没有可用的LLM端点")
            return None
        
        return await self._select_endpoint(endpoints, context, "llm")
    
    async def select_tts_endpoint(self, context: RoutingContext) -> Optional[APIEndpoint]:
        """选择TTS端点"""
        endpoints = remote_api_config.get_tts_endpoints(healthy_only=True)
        if not endpoints:
            logger.error("没有可用的TTS端点")
            return None
        
        return await self._select_endpoint(endpoints, context, "tts")
    
    async def _select_endpoint(self, endpoints: List[APIEndpoint], context: RoutingContext, service_type: str) -> Optional[APIEndpoint]:
        """选择端点的核心逻辑"""
        if len(endpoints) == 1:
            return endpoints[0]
        
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(endpoints, service_type)
        elif self.strategy == RoutingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(endpoints, service_type)
        elif self.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(endpoints)
        elif self.strategy == RoutingStrategy.FASTEST_RESPONSE:
            return self._fastest_response_select(endpoints)
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_select(endpoints, context)
        elif self.strategy == RoutingStrategy.INTELLIGENT:
            return await self._intelligent_select(endpoints, context)
        else:
            return random.choice(endpoints)
    
    def _round_robin_select(self, endpoints: List[APIEndpoint], service_type: str) -> APIEndpoint:
        """轮询选择"""
        if service_type not in self.round_robin_counters:
            self.round_robin_counters[service_type] = 0
        
        index = self.round_robin_counters[service_type] % len(endpoints)
        self.round_robin_counters[service_type] += 1
        return endpoints[index]
    
    def _weighted_round_robin_select(self, endpoints: List[APIEndpoint], service_type: str) -> APIEndpoint:
        """加权轮询选择"""
        weights = [ep.weight * ep.success_rate for ep in endpoints]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(endpoints)
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return endpoints[i]
        
        return endpoints[-1]
    
    def _least_connections_select(self, endpoints: List[APIEndpoint]) -> APIEndpoint:
        """最少连接选择"""
        return min(endpoints, key=lambda ep: ep.current_load)
    
    def _fastest_response_select(self, endpoints: List[APIEndpoint]) -> APIEndpoint:
        """最快响应选择"""
        return min(endpoints, key=lambda ep: ep.avg_response_time or float('inf'))
    
    def _cost_optimized_select(self, endpoints: List[APIEndpoint], context: RoutingContext) -> APIEndpoint:
        """成本优化选择"""
        # 过滤掉超出预算的端点
        if context.cost_budget > 0:
            affordable_endpoints = [
                ep for ep in endpoints 
                if ep.cost_per_1k_tokens * (context.expected_response_length / 1000) <= context.cost_budget
            ]
            if affordable_endpoints:
                endpoints = affordable_endpoints
        
        # 在可负担的端点中选择成本最低的
        return min(endpoints, key=lambda ep: ep.cost_per_1k_tokens)
    
    async def _intelligent_select(self, endpoints: List[APIEndpoint], context: RoutingContext) -> APIEndpoint:
        """智能选择算法"""
        scores = []
        
        for endpoint in endpoints:
            score = await self._calculate_endpoint_score(endpoint, context)
            scores.append((endpoint, score))
        
        # 按分数排序，选择最高分的端点
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 添加一些随机性，避免所有请求都打到同一个端点
        if len(scores) > 1 and random.random() < 0.1:
            # 10%的概率选择第二高分的端点
            return scores[1][0] if len(scores) > 1 else scores[0][0]
        
        return scores[0][0]
    
    async def _calculate_endpoint_score(self, endpoint: APIEndpoint, context: RoutingContext) -> float:
        """计算端点评分"""
        # 基础分数
        score = 0.0
        
        # 1. 延迟评分 (响应时间越短分数越高)
        latency_score = endpoint.latency_score
        score += latency_score * self.routing_weights["latency"]
        
        # 2. 成功率评分
        success_score = endpoint.success_rate
        score += success_score * self.routing_weights["success_rate"]
        
        # 3. 成本评分 (成本越低分数越高)
        if endpoint.cost_per_1k_tokens > 0:
            # 归一化成本评分
            max_cost = 0.1  # 假设最高成本为0.1元/1k tokens
            cost_score = max(0, 1 - (endpoint.cost_per_1k_tokens / max_cost))
        else:
            cost_score = 1.0  # 免费服务得满分
        score += cost_score * self.routing_weights["cost"]
        
        # 4. 负载评分 (负载越低分数越高)
        if endpoint.max_concurrent > 0:
            load_ratio = endpoint.current_load / endpoint.max_concurrent
            load_score = max(0, 1 - load_ratio)
        else:
            load_score = 0.5
        score += load_score * self.routing_weights["load"]
        
        # 5. 质量评分 (基于历史表现)
        quality_score = endpoint.cost_efficiency
        score += quality_score * self.routing_weights["quality"]
        
        # 6. 根据请求优先级调整权重
        if context.priority == RequestPriority.CRITICAL:
            # 关键请求更重视成功率和延迟
            score = score * 0.7 + (success_score * 0.4 + latency_score * 0.3)
        elif context.priority == RequestPriority.LOW:
            # 低优先级请求更重视成本
            score = score * 0.7 + (cost_score * 0.3)
        
        # 7. 考虑当前负载情况
        if endpoint.current_load >= endpoint.max_concurrent * 0.9:
            score *= 0.5  # 接近满载时大幅降低分数
        elif endpoint.current_load >= endpoint.max_concurrent * 0.7:
            score *= 0.8  # 高负载时适度降低分数
        
        return score
    
    async def update_endpoint_load(self, endpoint: APIEndpoint, delta: int):
        """更新端点负载"""
        endpoint.current_load = max(0, endpoint.current_load + delta)
        
        # 记录负载统计
        endpoint_key = f"{endpoint.provider.value}:{endpoint.name}"
        if endpoint_key not in self.load_stats:
            self.load_stats[endpoint_key] = {
                "peak_load": 0,
                "avg_load": 0,
                "load_history": []
            }
        
        stats = self.load_stats[endpoint_key]
        stats["peak_load"] = max(stats["peak_load"], endpoint.current_load)
        stats["load_history"].append(endpoint.current_load)
        
        # 保持历史记录在合理范围内
        if len(stats["load_history"]) > 1000:
            stats["load_history"] = stats["load_history"][-500:]
        
        # 计算平均负载
        if stats["load_history"]:
            stats["avg_load"] = sum(stats["load_history"]) / len(stats["load_history"])
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        return {
            "strategy": self.strategy.value,
            "routing_weights": self.routing_weights,
            "round_robin_counters": self.round_robin_counters,
            "load_stats": self.load_stats,
            "total_requests": sum(
                len(stats.get("load_history", [])) 
                for stats in self.load_stats.values()
            )
        }
    
    async def optimize_routing_weights(self):
        """基于历史数据优化路由权重"""
        # 这里可以实现基于机器学习的权重优化
        # 暂时使用简单的启发式规则
        
        # 分析最近的性能数据
        recent_performance = {}
        for endpoint in remote_api_config.llm_endpoints + remote_api_config.tts_endpoints:
            key = f"{endpoint.provider.value}:{endpoint.name}"
            recent_performance[key] = {
                "success_rate": endpoint.success_rate,
                "avg_response_time": endpoint.avg_response_time,
                "cost_efficiency": endpoint.cost_efficiency
            }
        
        # 根据整体性能调整权重
        avg_success_rate = np.mean([p["success_rate"] for p in recent_performance.values()])
        avg_response_time = np.mean([p["avg_response_time"] for p in recent_performance.values() if p["avg_response_time"] > 0])
        
        if avg_success_rate < 0.9:
            # 成功率偏低，增加成功率权重
            self.routing_weights["success_rate"] = min(0.4, self.routing_weights["success_rate"] * 1.1)
            self.routing_weights["cost"] = max(0.1, self.routing_weights["cost"] * 0.9)
        
        if avg_response_time > 5.0:
            # 响应时间偏高，增加延迟权重
            self.routing_weights["latency"] = min(0.4, self.routing_weights["latency"] * 1.1)
            self.routing_weights["cost"] = max(0.1, self.routing_weights["cost"] * 0.9)
        
        # 确保权重总和为1
        total_weight = sum(self.routing_weights.values())
        for key in self.routing_weights:
            self.routing_weights[key] /= total_weight

# 全局路由器实例
intelligent_router = IntelligentRouter()

async def get_intelligent_router() -> IntelligentRouter:
    """获取智能路由器实例"""
    return intelligent_router