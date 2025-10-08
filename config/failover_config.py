"""
故障转移和监控配置
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class FailoverStrategy(str, Enum):
    """故障转移策略"""
    IMMEDIATE = "immediate"  # 立即转移
    GRADUAL = "gradual"      # 逐步转移
    CIRCUIT_BREAKER = "circuit_breaker"  # 熔断器
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # 重试退避

class MonitoringLevel(str, Enum):
    """监控级别"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

@dataclass
class FailoverRule:
    """故障转移规则"""
    name: str
    condition: str  # 触发条件
    strategy: FailoverStrategy
    threshold: float  # 阈值
    window_size: int = 60  # 时间窗口（秒）
    cooldown: int = 300  # 冷却时间（秒）
    enabled: bool = True
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    max_retry_delay: float = 60.0

@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5  # 失败阈值
    success_threshold: int = 3  # 成功阈值（半开状态）
    timeout: int = 60  # 超时时间（秒）
    
    # 状态
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_success_time: float = 0.0

@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric: str  # 监控指标
    operator: str  # 操作符: >, <, >=, <=, ==, !=
    threshold: float  # 阈值
    duration: int = 60  # 持续时间（秒）
    severity: str = "warning"  # critical, warning, info
    enabled: bool = True
    
    # 告警渠道
    channels: List[str] = field(default_factory=list)  # email, slack, webhook
    
    # 抑制配置
    suppress_duration: int = 300  # 抑制时间（秒）
    last_alert_time: float = 0.0

class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self):
        self.rules: Dict[str, FailoverRule] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerConfig] = {}
        self.metrics_history: Dict[str, List[Tuple[float, float]]] = {}  # (timestamp, value)
        self.active_failures: Dict[str, float] = {}  # service_id -> failure_time
        
    def add_rule(self, rule: FailoverRule):
        """添加故障转移规则"""
        self.rules[rule.name] = rule
        logger.info(f"添加故障转移规则: {rule.name}")
    
    def add_circuit_breaker(self, service_id: str, config: CircuitBreakerConfig):
        """添加熔断器"""
        self.circuit_breakers[service_id] = config
        logger.info(f"添加熔断器: {service_id}")
    
    async def check_failover_conditions(self, service_id: str, metrics: Dict[str, float]) -> bool:
        """检查故障转移条件"""
        current_time = time.time()
        
        # 更新指标历史
        for metric_name, value in metrics.items():
            key = f"{service_id}:{metric_name}"
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            
            self.metrics_history[key].append((current_time, value))
            
            # 清理过期数据
            self.metrics_history[key] = [
                (ts, val) for ts, val in self.metrics_history[key]
                if current_time - ts <= 3600  # 保留1小时数据
            ]
        
        # 检查每个规则
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if await self._evaluate_rule(rule, service_id, metrics, current_time):
                logger.warning(f"触发故障转移规则: {rule.name} for service: {service_id}")
                await self._execute_failover(rule, service_id)
                return True
        
        return False
    
    async def _evaluate_rule(self, rule: FailoverRule, service_id: str, 
                           metrics: Dict[str, float], current_time: float) -> bool:
        """评估规则条件"""
        try:
            # 检查冷却时间
            if service_id in self.active_failures:
                if current_time - self.active_failures[service_id] < rule.cooldown:
                    return False
            
            # 解析条件
            if rule.condition == "error_rate":
                error_rate = metrics.get("error_rate", 0.0)
                return error_rate > rule.threshold
            
            elif rule.condition == "response_time":
                response_time = metrics.get("avg_response_time", 0.0)
                return response_time > rule.threshold
            
            elif rule.condition == "success_rate":
                success_rate = metrics.get("success_rate", 1.0)
                return success_rate < rule.threshold
            
            elif rule.condition == "connection_count":
                connections = metrics.get("current_connections", 0)
                return connections > rule.threshold
            
            elif rule.condition == "cpu_usage":
                cpu_usage = metrics.get("cpu_usage", 0.0)
                return cpu_usage > rule.threshold
            
            elif rule.condition == "memory_usage":
                memory_usage = metrics.get("memory_usage", 0.0)
                return memory_usage > rule.threshold
            
            else:
                logger.warning(f"未知的故障转移条件: {rule.condition}")
                return False
                
        except Exception as e:
            logger.error(f"评估故障转移规则失败: {e}")
            return False
    
    async def _execute_failover(self, rule: FailoverRule, service_id: str):
        """执行故障转移"""
        current_time = time.time()
        self.active_failures[service_id] = current_time
        
        if rule.strategy == FailoverStrategy.IMMEDIATE:
            await self._immediate_failover(service_id)
        elif rule.strategy == FailoverStrategy.GRADUAL:
            await self._gradual_failover(service_id)
        elif rule.strategy == FailoverStrategy.CIRCUIT_BREAKER:
            await self._circuit_breaker_failover(service_id)
        elif rule.strategy == FailoverStrategy.RETRY_WITH_BACKOFF:
            await self._retry_with_backoff(service_id, rule)
    
    async def _immediate_failover(self, service_id: str):
        """立即故障转移"""
        logger.info(f"执行立即故障转移: {service_id}")
        # 这里实现具体的故障转移逻辑
        # 例如：切换到备用服务、重启服务等
    
    async def _gradual_failover(self, service_id: str):
        """逐步故障转移"""
        logger.info(f"执行逐步故障转移: {service_id}")
        # 这里实现逐步转移逻辑
        # 例如：逐步减少流量、分批迁移等
    
    async def _circuit_breaker_failover(self, service_id: str):
        """熔断器故障转移"""
        if service_id not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[service_id]
        breaker.state = "OPEN"
        breaker.last_failure_time = time.time()
        
        logger.info(f"熔断器开启: {service_id}")
    
    async def _retry_with_backoff(self, service_id: str, rule: FailoverRule):
        """重试退避故障转移"""
        logger.info(f"执行重试退避: {service_id}")
        
        for attempt in range(rule.max_retries):
            delay = min(
                rule.retry_delay * (rule.backoff_multiplier ** attempt),
                rule.max_retry_delay
            )
            
            logger.info(f"重试 {attempt + 1}/{rule.max_retries}, 延迟 {delay}s")
            await asyncio.sleep(delay)
            
            # 这里实现重试逻辑
            # 如果成功，则跳出循环
            # if await self._retry_service(service_id):
            #     break
    
    def check_circuit_breaker(self, service_id: str) -> bool:
        """检查熔断器状态"""
        if service_id not in self.circuit_breakers:
            return True  # 没有熔断器，允许通过
        
        breaker = self.circuit_breakers[service_id]
        current_time = time.time()
        
        if breaker.state == "CLOSED":
            return True
        elif breaker.state == "OPEN":
            if current_time - breaker.last_failure_time > breaker.timeout:
                breaker.state = "HALF_OPEN"
                breaker.success_count = 0
                return True
            return False
        elif breaker.state == "HALF_OPEN":
            return True
        
        return False
    
    def record_success(self, service_id: str):
        """记录成功"""
        if service_id not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[service_id]
        
        if breaker.state == "HALF_OPEN":
            breaker.success_count += 1
            if breaker.success_count >= breaker.success_threshold:
                breaker.state = "CLOSED"
                breaker.failure_count = 0
                logger.info(f"熔断器关闭: {service_id}")
        elif breaker.state == "CLOSED":
            breaker.failure_count = 0
        
        breaker.last_success_time = time.time()
    
    def record_failure(self, service_id: str):
        """记录失败"""
        if service_id not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[service_id]
        breaker.failure_count += 1
        breaker.last_failure_time = time.time()
        
        if breaker.state == "CLOSED" and breaker.failure_count >= breaker.failure_threshold:
            breaker.state = "OPEN"
            logger.warning(f"熔断器开启: {service_id}")
        elif breaker.state == "HALF_OPEN":
            breaker.state = "OPEN"
            breaker.success_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "rules": {
                name: {
                    "name": rule.name,
                    "condition": rule.condition,
                    "strategy": rule.strategy.value,
                    "threshold": rule.threshold,
                    "enabled": rule.enabled
                }
                for name, rule in self.rules.items()
            },
            "circuit_breakers": {
                service_id: {
                    "state": breaker.state,
                    "failure_count": breaker.failure_count,
                    "success_count": breaker.success_count,
                    "last_failure_time": breaker.last_failure_time,
                    "last_success_time": breaker.last_success_time
                }
                for service_id, breaker in self.circuit_breakers.items()
            },
            "active_failures": self.active_failures,
            "metrics_history_size": {
                key: len(history) for key, history in self.metrics_history.items()
            }
        }

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.alert_handlers: Dict[str, Callable] = {}
        self.active_alerts: Dict[str, float] = {}  # alert_id -> start_time
        
    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules[rule.name] = rule
        logger.info(f"添加告警规则: {rule.name}")
    
    def add_handler(self, channel: str, handler: Callable):
        """添加告警处理器"""
        self.alert_handlers[channel] = handler
        logger.info(f"添加告警处理器: {channel}")
    
    async def check_alerts(self, service_id: str, metrics: Dict[str, float]):
        """检查告警条件"""
        current_time = time.time()
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            alert_id = f"{service_id}:{rule.name}"
            
            # 检查抑制时间
            if current_time - rule.last_alert_time < rule.suppress_duration:
                continue
            
            # 评估告警条件
            if await self._evaluate_alert_rule(rule, metrics):
                # 检查持续时间
                if alert_id not in self.active_alerts:
                    self.active_alerts[alert_id] = current_time
                elif current_time - self.active_alerts[alert_id] >= rule.duration:
                    # 触发告警
                    await self._trigger_alert(rule, service_id, metrics)
                    rule.last_alert_time = current_time
            else:
                # 条件不满足，清除活跃告警
                if alert_id in self.active_alerts:
                    del self.active_alerts[alert_id]
    
    async def _evaluate_alert_rule(self, rule: AlertRule, metrics: Dict[str, float]) -> bool:
        """评估告警规则"""
        try:
            metric_value = metrics.get(rule.metric, 0.0)
            
            if rule.operator == ">":
                return metric_value > rule.threshold
            elif rule.operator == "<":
                return metric_value < rule.threshold
            elif rule.operator == ">=":
                return metric_value >= rule.threshold
            elif rule.operator == "<=":
                return metric_value <= rule.threshold
            elif rule.operator == "==":
                return metric_value == rule.threshold
            elif rule.operator == "!=":
                return metric_value != rule.threshold
            else:
                logger.warning(f"未知的告警操作符: {rule.operator}")
                return False
                
        except Exception as e:
            logger.error(f"评估告警规则失败: {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, service_id: str, metrics: Dict[str, float]):
        """触发告警"""
        alert_data = {
            "rule_name": rule.name,
            "service_id": service_id,
            "metric": rule.metric,
            "threshold": rule.threshold,
            "current_value": metrics.get(rule.metric, 0.0),
            "severity": rule.severity,
            "timestamp": time.time(),
            "metrics": metrics
        }
        
        logger.warning(f"触发告警: {rule.name} for service: {service_id}")
        
        # 发送到各个渠道
        for channel in rule.channels:
            if channel in self.alert_handlers:
                try:
                    await self.alert_handlers[channel](alert_data)
                except Exception as e:
                    logger.error(f"发送告警到 {channel} 失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "rules": {
                name: {
                    "name": rule.name,
                    "metric": rule.metric,
                    "operator": rule.operator,
                    "threshold": rule.threshold,
                    "severity": rule.severity,
                    "enabled": rule.enabled,
                    "last_alert_time": rule.last_alert_time
                }
                for name, rule in self.rules.items()
            },
            "active_alerts": len(self.active_alerts),
            "alert_handlers": list(self.alert_handlers.keys())
        }

class MonitoringCollector:
    """监控数据收集器"""
    
    def __init__(self, level: MonitoringLevel = MonitoringLevel.DETAILED):
        self.level = level
        self.metrics: Dict[str, List[Tuple[float, Dict[str, float]]]] = {}
        self.collection_interval = 30  # 30秒
        self.running = False
    
    async def start_collection(self, services: List[str]):
        """开始收集监控数据"""
        self.running = True
        
        while self.running:
            try:
                for service_id in services:
                    metrics = await self._collect_service_metrics(service_id)
                    self._store_metrics(service_id, metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"监控数据收集失败: {e}")
    
    async def stop_collection(self):
        """停止收集"""
        self.running = False
    
    async def _collect_service_metrics(self, service_id: str) -> Dict[str, float]:
        """收集服务指标"""
        metrics = {}
        
        try:
            # 基础指标
            if self.level in [MonitoringLevel.BASIC, MonitoringLevel.DETAILED, MonitoringLevel.COMPREHENSIVE]:
                metrics.update(await self._collect_basic_metrics(service_id))
            
            # 详细指标
            if self.level in [MonitoringLevel.DETAILED, MonitoringLevel.COMPREHENSIVE]:
                metrics.update(await self._collect_detailed_metrics(service_id))
            
            # 全面指标
            if self.level == MonitoringLevel.COMPREHENSIVE:
                metrics.update(await self._collect_comprehensive_metrics(service_id))
            
        except Exception as e:
            logger.error(f"收集服务 {service_id} 指标失败: {e}")
        
        return metrics
    
    async def _collect_basic_metrics(self, service_id: str) -> Dict[str, float]:
        """收集基础指标"""
        # 这里实现基础指标收集
        # 例如：CPU使用率、内存使用率、请求数等
        return {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "request_count": 0.0,
            "error_count": 0.0
        }
    
    async def _collect_detailed_metrics(self, service_id: str) -> Dict[str, float]:
        """收集详细指标"""
        # 这里实现详细指标收集
        # 例如：响应时间、吞吐量、连接数等
        return {
            "avg_response_time": 0.0,
            "throughput": 0.0,
            "current_connections": 0.0,
            "queue_size": 0.0
        }
    
    async def _collect_comprehensive_metrics(self, service_id: str) -> Dict[str, float]:
        """收集全面指标"""
        # 这里实现全面指标收集
        # 例如：GC时间、线程数、磁盘IO等
        return {
            "gc_time": 0.0,
            "thread_count": 0.0,
            "disk_io": 0.0,
            "network_io": 0.0
        }
    
    def _store_metrics(self, service_id: str, metrics: Dict[str, float]):
        """存储指标数据"""
        current_time = time.time()
        
        if service_id not in self.metrics:
            self.metrics[service_id] = []
        
        self.metrics[service_id].append((current_time, metrics))
        
        # 清理过期数据（保留24小时）
        cutoff_time = current_time - 86400
        self.metrics[service_id] = [
            (ts, data) for ts, data in self.metrics[service_id]
            if ts > cutoff_time
        ]
    
    def get_metrics(self, service_id: str, start_time: Optional[float] = None, 
                   end_time: Optional[float] = None) -> List[Tuple[float, Dict[str, float]]]:
        """获取指标数据"""
        if service_id not in self.metrics:
            return []
        
        data = self.metrics[service_id]
        
        if start_time:
            data = [(ts, metrics) for ts, metrics in data if ts >= start_time]
        
        if end_time:
            data = [(ts, metrics) for ts, metrics in data if ts <= end_time]
        
        return data
    
    def get_latest_metrics(self, service_id: str) -> Optional[Dict[str, float]]:
        """获取最新指标"""
        if service_id not in self.metrics or not self.metrics[service_id]:
            return None
        
        return self.metrics[service_id][-1][1]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "level": self.level.value,
            "collection_interval": self.collection_interval,
            "running": self.running,
            "services": list(self.metrics.keys()),
            "total_data_points": sum(len(data) for data in self.metrics.values())
        }

# 默认配置
DEFAULT_FAILOVER_RULES = [
    FailoverRule(
        name="high_error_rate",
        condition="error_rate",
        strategy=FailoverStrategy.CIRCUIT_BREAKER,
        threshold=0.1,  # 10%错误率
        window_size=60,
        cooldown=300
    ),
    FailoverRule(
        name="slow_response",
        condition="response_time",
        strategy=FailoverStrategy.RETRY_WITH_BACKOFF,
        threshold=5000,  # 5秒
        window_size=60,
        cooldown=180
    ),
    FailoverRule(
        name="low_success_rate",
        condition="success_rate",
        strategy=FailoverStrategy.GRADUAL,
        threshold=0.8,  # 80%成功率
        window_size=120,
        cooldown=600
    )
]

DEFAULT_ALERT_RULES = [
    AlertRule(
        name="high_cpu_usage",
        metric="cpu_usage",
        operator=">",
        threshold=80.0,
        duration=300,
        severity="warning",
        channels=["email", "slack"]
    ),
    AlertRule(
        name="high_memory_usage",
        metric="memory_usage",
        operator=">",
        threshold=85.0,
        duration=300,
        severity="warning",
        channels=["email", "slack"]
    ),
    AlertRule(
        name="service_down",
        metric="success_rate",
        operator="<",
        threshold=0.5,
        duration=60,
        severity="critical",
        channels=["email", "slack", "webhook"]
    )
]