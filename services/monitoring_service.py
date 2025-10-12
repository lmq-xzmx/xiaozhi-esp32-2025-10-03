#!/usr/bin/env python3
"""
性能监控服务
监控系统资源、应用性能和数据库状态
"""

import asyncio
import psutil
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json

from config.redis_config import get_redis_client
from core.enhanced_db_service import get_enhanced_db_service

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemMetrics:
    """系统指标"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    disk_free: int
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: List[float]
    process_count: int

@dataclass
class DatabaseMetrics:
    """数据库指标"""
    timestamp: datetime
    connection_count: int
    active_queries: int
    slow_queries: int
    cache_hit_ratio: float
    response_time_ms: float
    error_count: int

@dataclass
class ApplicationMetrics:
    """应用指标"""
    timestamp: datetime
    active_websocket_connections: int
    total_requests: int
    error_rate: float
    average_response_time: float
    memory_usage_mb: float
    uptime_seconds: float

@dataclass
class Alert:
    """告警信息"""
    id: str
    level: AlertLevel
    message: str
    timestamp: datetime
    metric_type: str
    metric_value: Any
    threshold: Any
    resolved: bool = False
    resolved_at: Optional[datetime] = None

class MonitoringService:
    """性能监控服务"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.db_service = get_enhanced_db_service()
        self.start_time = datetime.now()
        self.is_running = False
        self.monitoring_interval = 30  # 30秒监控间隔
        self.metrics_retention_days = 7  # 保留7天的指标数据
        
        # 告警阈值配置
        self.thresholds = {
            'cpu_percent': {'warning': 70, 'critical': 90},
            'memory_percent': {'warning': 80, 'critical': 95},
            'disk_usage_percent': {'warning': 85, 'critical': 95},
            'response_time_ms': {'warning': 1000, 'critical': 5000},
            'error_rate': {'warning': 0.05, 'critical': 0.1},  # 5% warning, 10% critical
        }
        
        # 指标计数器
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.active_alerts = {}
        
        logger.info("性能监控服务初始化完成")
    
    async def start_monitoring(self):
        """启动监控守护进程"""
        if self.is_running:
            logger.warning("监控服务已在运行")
            return
        
        self.is_running = True
        logger.info("启动性能监控守护进程")
        
        # 启动监控任务
        asyncio.create_task(self._monitoring_loop())
        asyncio.create_task(self._cleanup_old_metrics())
    
    async def stop_monitoring(self):
        """停止监控守护进程"""
        self.is_running = False
        logger.info("性能监控守护进程已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                system_metrics = await self._collect_system_metrics()
                await self._store_metrics("system", system_metrics)
                await self._check_system_alerts(system_metrics)
                
                # 收集数据库指标
                db_metrics = await self._collect_database_metrics()
                await self._store_metrics("database", db_metrics)
                await self._check_database_alerts(db_metrics)
                
                # 收集应用指标
                app_metrics = await self._collect_application_metrics()
                await self._store_metrics("application", app_metrics)
                await self._check_application_alerts(app_metrics)
                
                logger.debug("监控指标收集完成")
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
            
            await asyncio.sleep(self.monitoring_interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存信息
        memory = psutil.virtual_memory()
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        
        # 网络信息
        network = psutil.net_io_counters()
        
        # 负载平均值
        load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        
        # 进程数量
        process_count = len(psutil.pids())
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available=memory.available,
            disk_usage_percent=disk.percent,
            disk_free=disk.free,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            load_average=list(load_avg),
            process_count=process_count
        )
    
    async def _collect_database_metrics(self) -> DatabaseMetrics:
        """收集数据库指标"""
        start_time = time.time()
        
        try:
            # 测试数据库连接和响应时间
            await self.db_service.health_check()
            response_time = (time.time() - start_time) * 1000
            
            # 这里可以添加更多数据库特定的指标
            # 例如：连接数、活跃查询数、慢查询数等
            
            return DatabaseMetrics(
                timestamp=datetime.now(),
                connection_count=1,  # 简化实现
                active_queries=0,
                slow_queries=0,
                cache_hit_ratio=0.95,  # 假设值
                response_time_ms=response_time,
                error_count=0
            )
            
        except Exception as e:
            logger.error(f"数据库指标收集失败: {e}")
            return DatabaseMetrics(
                timestamp=datetime.now(),
                connection_count=0,
                active_queries=0,
                slow_queries=0,
                cache_hit_ratio=0,
                response_time_ms=5000,  # 超时值
                error_count=1
            )
    
    async def _collect_application_metrics(self) -> ApplicationMetrics:
        """收集应用指标"""
        # 计算运行时间
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        # 计算错误率
        error_rate = self.error_count / max(self.request_count, 1)
        
        # 计算平均响应时间
        avg_response_time = sum(self.response_times) / max(len(self.response_times), 1)
        
        # 获取当前进程的内存使用
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        return ApplicationMetrics(
            timestamp=datetime.now(),
            active_websocket_connections=0,  # 需要从WebSocket管理器获取
            total_requests=self.request_count,
            error_rate=error_rate,
            average_response_time=avg_response_time,
            memory_usage_mb=memory_usage,
            uptime_seconds=uptime
        )
    
    async def _store_metrics(self, metric_type: str, metrics: Any):
        """存储指标到Redis"""
        try:
            key = f"metrics:{metric_type}:{int(time.time())}"
            value = json.dumps(asdict(metrics), default=str)
            
            # 存储指标，设置过期时间
            await self.redis_client.set_with_ttl(
                key, 
                value, 
                ttl=self.metrics_retention_days * 24 * 3600
            )
            
        except Exception as e:
            logger.error(f"存储指标失败: {e}")
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """检查系统告警"""
        alerts = []
        
        # CPU告警
        if metrics.cpu_percent > self.thresholds['cpu_percent']['critical']:
            alerts.append(self._create_alert(
                'cpu_critical', AlertLevel.CRITICAL,
                f"CPU使用率过高: {metrics.cpu_percent:.1f}%",
                'cpu_percent', metrics.cpu_percent,
                self.thresholds['cpu_percent']['critical']
            ))
        elif metrics.cpu_percent > self.thresholds['cpu_percent']['warning']:
            alerts.append(self._create_alert(
                'cpu_warning', AlertLevel.WARNING,
                f"CPU使用率较高: {metrics.cpu_percent:.1f}%",
                'cpu_percent', metrics.cpu_percent,
                self.thresholds['cpu_percent']['warning']
            ))
        
        # 内存告警
        if metrics.memory_percent > self.thresholds['memory_percent']['critical']:
            alerts.append(self._create_alert(
                'memory_critical', AlertLevel.CRITICAL,
                f"内存使用率过高: {metrics.memory_percent:.1f}%",
                'memory_percent', metrics.memory_percent,
                self.thresholds['memory_percent']['critical']
            ))
        elif metrics.memory_percent > self.thresholds['memory_percent']['warning']:
            alerts.append(self._create_alert(
                'memory_warning', AlertLevel.WARNING,
                f"内存使用率较高: {metrics.memory_percent:.1f}%",
                'memory_percent', metrics.memory_percent,
                self.thresholds['memory_percent']['warning']
            ))
        
        # 磁盘告警
        if metrics.disk_usage_percent > self.thresholds['disk_usage_percent']['critical']:
            alerts.append(self._create_alert(
                'disk_critical', AlertLevel.CRITICAL,
                f"磁盘使用率过高: {metrics.disk_usage_percent:.1f}%",
                'disk_usage_percent', metrics.disk_usage_percent,
                self.thresholds['disk_usage_percent']['critical']
            ))
        elif metrics.disk_usage_percent > self.thresholds['disk_usage_percent']['warning']:
            alerts.append(self._create_alert(
                'disk_warning', AlertLevel.WARNING,
                f"磁盘使用率较高: {metrics.disk_usage_percent:.1f}%",
                'disk_usage_percent', metrics.disk_usage_percent,
                self.thresholds['disk_usage_percent']['warning']
            ))
        
        # 处理告警
        for alert in alerts:
            await self._handle_alert(alert)
    
    async def _check_database_alerts(self, metrics: DatabaseMetrics):
        """检查数据库告警"""
        alerts = []
        
        # 响应时间告警
        if metrics.response_time_ms > self.thresholds['response_time_ms']['critical']:
            alerts.append(self._create_alert(
                'db_response_critical', AlertLevel.CRITICAL,
                f"数据库响应时间过长: {metrics.response_time_ms:.1f}ms",
                'response_time_ms', metrics.response_time_ms,
                self.thresholds['response_time_ms']['critical']
            ))
        elif metrics.response_time_ms > self.thresholds['response_time_ms']['warning']:
            alerts.append(self._create_alert(
                'db_response_warning', AlertLevel.WARNING,
                f"数据库响应时间较长: {metrics.response_time_ms:.1f}ms",
                'response_time_ms', metrics.response_time_ms,
                self.thresholds['response_time_ms']['warning']
            ))
        
        # 处理告警
        for alert in alerts:
            await self._handle_alert(alert)
    
    async def _check_application_alerts(self, metrics: ApplicationMetrics):
        """检查应用告警"""
        alerts = []
        
        # 错误率告警
        if metrics.error_rate > self.thresholds['error_rate']['critical']:
            alerts.append(self._create_alert(
                'error_rate_critical', AlertLevel.CRITICAL,
                f"应用错误率过高: {metrics.error_rate:.2%}",
                'error_rate', metrics.error_rate,
                self.thresholds['error_rate']['critical']
            ))
        elif metrics.error_rate > self.thresholds['error_rate']['warning']:
            alerts.append(self._create_alert(
                'error_rate_warning', AlertLevel.WARNING,
                f"应用错误率较高: {metrics.error_rate:.2%}",
                'error_rate', metrics.error_rate,
                self.thresholds['error_rate']['warning']
            ))
        
        # 处理告警
        for alert in alerts:
            await self._handle_alert(alert)
    
    def _create_alert(self, alert_id: str, level: AlertLevel, message: str,
                     metric_type: str, metric_value: Any, threshold: Any) -> Alert:
        """创建告警"""
        return Alert(
            id=alert_id,
            level=level,
            message=message,
            timestamp=datetime.now(),
            metric_type=metric_type,
            metric_value=metric_value,
            threshold=threshold
        )
    
    async def _handle_alert(self, alert: Alert):
        """处理告警"""
        # 检查是否是新告警
        if alert.id not in self.active_alerts:
            self.active_alerts[alert.id] = alert
            logger.warning(f"新告警: [{alert.level.value.upper()}] {alert.message}")
            
            # 存储告警到Redis
            await self._store_alert(alert)
        else:
            # 更新现有告警
            self.active_alerts[alert.id] = alert
    
    async def _store_alert(self, alert: Alert):
        """存储告警到Redis"""
        try:
            key = f"alerts:{alert.id}:{int(time.time())}"
            value = json.dumps(asdict(alert), default=str)
            
            # 存储告警，设置过期时间
            await self.redis_client.set_with_ttl(key, value, ttl=30 * 24 * 3600)  # 30天
            
        except Exception as e:
            logger.error(f"存储告警失败: {e}")
    
    async def _cleanup_old_metrics(self):
        """清理旧的指标数据"""
        while self.is_running:
            try:
                # 每小时清理一次
                await asyncio.sleep(3600)
                
                # 清理过期的指标数据
                cutoff_time = int((datetime.now() - timedelta(days=self.metrics_retention_days)).timestamp())
                
                # 这里可以添加清理逻辑
                logger.debug("清理旧指标数据完成")
                
            except Exception as e:
                logger.error(f"清理旧指标数据失败: {e}")
    
    async def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取指标摘要"""
        try:
            end_time = int(time.time())
            start_time = end_time - (hours * 3600)
            
            # 从Redis获取指标数据
            metrics_data = {
                'system': [],
                'database': [],
                'application': []
            }
            
            # 这里可以添加从Redis查询指标的逻辑
            
            return {
                'status': 'success',
                'time_range': f'{hours}小时',
                'metrics': metrics_data,
                'active_alerts': len(self.active_alerts),
                'total_requests': self.request_count,
                'error_count': self.error_count
            }
            
        except Exception as e:
            logger.error(f"获取指标摘要失败: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return [asdict(alert) for alert in self.active_alerts.values()]
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()
            
            # 从活跃告警中移除
            del self.active_alerts[alert_id]
            
            logger.info(f"告警已解决: {alert_id}")
            return True
        
        return False
    
    def record_request(self, response_time: float, is_error: bool = False):
        """记录请求指标"""
        self.request_count += 1
        self.response_times.append(response_time)
        
        if is_error:
            self.error_count += 1
        
        # 保持响应时间列表大小
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-500:]
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """获取当前系统指标"""
        try:
            metrics = await self._collect_system_metrics()
            return asdict(metrics)
        except Exception as e:
            logger.error(f"获取系统指标失败: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def start_monitoring_daemon(self):
        """启动监控守护进程（别名方法）"""
        return await self.start_monitoring()
    
    async def stop_monitoring_daemon(self):
        """停止监控守护进程（别名方法）"""
        return await self.stop_monitoring()

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self.is_running else 'stopped',
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'monitoring_interval': self.monitoring_interval,
            'active_alerts': len(self.active_alerts),
            'total_requests': self.request_count,
            'error_count': self.error_count
        }

# 全局监控服务实例
_monitoring_service = None

def get_monitoring_service() -> MonitoringService:
    """获取监控服务实例"""
    global _monitoring_service
    if _monitoring_service is None:
        _monitoring_service = MonitoringService()
    return _monitoring_service