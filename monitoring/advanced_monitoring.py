#!/usr/bin/env python3
"""
Xiaozhi ESP32 Server - 高级监控告警系统
实现全方位监控、智能告警、性能分析和自动化运维
"""

import asyncio
import json
import time
import logging
import smtplib
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
from collections import defaultdict, deque
import statistics
import numpy as np

import aioredis
import aiohttp
import psutil
import GPUtil
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"

@dataclass
class MetricPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}

@dataclass
class Alert:
    """告警"""
    id: str
    level: AlertLevel
    title: str
    description: str
    service: str
    metric: str
    threshold: float
    current_value: float
    created_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ServiceHealth:
    """服务健康状态"""
    service_name: str
    status: ServiceStatus
    last_check: datetime
    response_time: float
    error_rate: float
    uptime: float
    version: str = ""
    dependencies: List[str] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metrics is None:
            self.metrics = {}

class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_history = defaultdict(deque)
        self.baselines = {}
        self.anomalies = []
    
    def add_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """添加指标数据"""
        if timestamp is None:
            timestamp = datetime.now()
        
        point = MetricPoint(timestamp=timestamp, value=value)
        self.metric_history[metric_name].append(point)
        
        # 保持窗口大小
        if len(self.metric_history[metric_name]) > self.window_size:
            self.metric_history[metric_name].popleft()
        
        # 更新基线
        self._update_baseline(metric_name)
    
    def _update_baseline(self, metric_name: str):
        """更新基线"""
        history = self.metric_history[metric_name]
        if len(history) < 10:  # 数据不足
            return
        
        values = [point.value for point in history]
        
        # 计算统计指标
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        median = statistics.median(values)
        
        # 计算分位数
        sorted_values = sorted(values)
        q25 = sorted_values[len(sorted_values) // 4]
        q75 = sorted_values[3 * len(sorted_values) // 4]
        iqr = q75 - q25
        
        self.baselines[metric_name] = {
            'mean': mean,
            'std': std,
            'median': median,
            'q25': q25,
            'q75': q75,
            'iqr': iqr,
            'min': min(values),
            'max': max(values),
            'updated_at': datetime.now()
        }
    
    def detect_anomaly(self, metric_name: str, value: float) -> Optional[Dict[str, Any]]:
        """检测异常"""
        if metric_name not in self.baselines:
            return None
        
        baseline = self.baselines[metric_name]
        
        # Z-score异常检测
        z_score = abs(value - baseline['mean']) / max(baseline['std'], 0.001)
        
        # IQR异常检测
        iqr_lower = baseline['q25'] - 1.5 * baseline['iqr']
        iqr_upper = baseline['q75'] + 1.5 * baseline['iqr']
        
        # 判断异常
        is_anomaly = False
        anomaly_type = None
        confidence = 0.0
        
        if z_score > self.sensitivity:
            is_anomaly = True
            anomaly_type = "z_score"
            confidence = min(z_score / self.sensitivity, 1.0)
        
        if value < iqr_lower or value > iqr_upper:
            is_anomaly = True
            if anomaly_type is None:
                anomaly_type = "iqr"
                confidence = 0.8
        
        if is_anomaly:
            anomaly = {
                'metric_name': metric_name,
                'value': value,
                'baseline': baseline,
                'anomaly_type': anomaly_type,
                'confidence': confidence,
                'z_score': z_score,
                'timestamp': datetime.now()
            }
            
            self.anomalies.append(anomaly)
            
            # 保持最近1000个异常记录
            if len(self.anomalies) > 1000:
                self.anomalies = self.anomalies[-1000:]
            
            return anomaly
        
        return None

class AlertManager:
    """告警管理器"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_rules = {}    # rule_id -> rule_config
        self.notification_channels = {}  # channel_id -> channel_config
        
        # 告警抑制和聚合
        self.suppression_rules = {}
        self.alert_groups = defaultdict(list)
        
        # 告警历史
        self.alert_history = deque(maxlen=10000)
        
        # 通知限流
        self.notification_limits = defaultdict(lambda: {'count': 0, 'last_sent': None})
    
    async def initialize(self):
        """初始化告警管理器"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            logger.info("Alert manager Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
        
        # 加载告警规则
        await self._load_alert_rules()
        
        # 加载通知渠道
        await self._load_notification_channels()
    
    async def _load_alert_rules(self):
        """加载告警规则"""
        # 默认告警规则
        default_rules = {
            'high_cpu_usage': {
                'metric': 'cpu_usage',
                'condition': 'greater_than',
                'threshold': 80.0,
                'duration': 300,  # 5分钟
                'level': AlertLevel.WARNING,
                'title': 'High CPU Usage',
                'description': 'CPU usage is above {threshold}% for {duration} seconds'
            },
            'high_memory_usage': {
                'metric': 'memory_usage',
                'condition': 'greater_than',
                'threshold': 85.0,
                'duration': 300,
                'level': AlertLevel.WARNING,
                'title': 'High Memory Usage',
                'description': 'Memory usage is above {threshold}% for {duration} seconds'
            },
            'high_response_time': {
                'metric': 'response_time',
                'condition': 'greater_than',
                'threshold': 2000.0,  # 2秒
                'duration': 180,      # 3分钟
                'level': AlertLevel.WARNING,
                'title': 'High Response Time',
                'description': 'Average response time is above {threshold}ms for {duration} seconds'
            },
            'high_error_rate': {
                'metric': 'error_rate',
                'condition': 'greater_than',
                'threshold': 5.0,  # 5%
                'duration': 120,   # 2分钟
                'level': AlertLevel.CRITICAL,
                'title': 'High Error Rate',
                'description': 'Error rate is above {threshold}% for {duration} seconds'
            },
            'service_down': {
                'metric': 'service_availability',
                'condition': 'less_than',
                'threshold': 1.0,
                'duration': 60,  # 1分钟
                'level': AlertLevel.CRITICAL,
                'title': 'Service Down',
                'description': 'Service is not responding for {duration} seconds'
            },
            'disk_space_low': {
                'metric': 'disk_usage',
                'condition': 'greater_than',
                'threshold': 90.0,
                'duration': 600,  # 10分钟
                'level': AlertLevel.WARNING,
                'title': 'Low Disk Space',
                'description': 'Disk usage is above {threshold}% for {duration} seconds'
            },
            'queue_backlog': {
                'metric': 'queue_length',
                'condition': 'greater_than',
                'threshold': 1000,
                'duration': 300,
                'level': AlertLevel.WARNING,
                'title': 'Queue Backlog',
                'description': 'Queue length is above {threshold} for {duration} seconds'
            }
        }
        
        self.alert_rules.update(default_rules)
    
    async def _load_notification_channels(self):
        """加载通知渠道"""
        # 默认通知渠道配置
        default_channels = {
            'email': {
                'type': 'email',
                'enabled': False,  # 需要配置后启用
                'config': {
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'recipients': []
                }
            },
            'webhook': {
                'type': 'webhook',
                'enabled': False,
                'config': {
                    'url': '',
                    'method': 'POST',
                    'headers': {},
                    'timeout': 10
                }
            },
            'slack': {
                'type': 'slack',
                'enabled': False,
                'config': {
                    'webhook_url': '',
                    'channel': '#alerts',
                    'username': 'Xiaozhi Monitor'
                }
            }
        }
        
        self.notification_channels.update(default_channels)
    
    def add_alert_rule(self, rule_id: str, rule_config: Dict[str, Any]):
        """添加告警规则"""
        self.alert_rules[rule_id] = rule_config
        logger.info(f"Added alert rule: {rule_id}")
    
    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    async def evaluate_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """评估指标并触发告警"""
        if labels is None:
            labels = {}
        
        for rule_id, rule in self.alert_rules.items():
            if rule['metric'] == metric_name:
                await self._evaluate_rule(rule_id, rule, value, labels)
    
    async def _evaluate_rule(self, rule_id: str, rule: Dict[str, Any], value: float, labels: Dict[str, str]):
        """评估单个规则"""
        condition = rule['condition']
        threshold = rule['threshold']
        
        # 检查条件
        triggered = False
        if condition == 'greater_than' and value > threshold:
            triggered = True
        elif condition == 'less_than' and value < threshold:
            triggered = True
        elif condition == 'equals' and value == threshold:
            triggered = True
        elif condition == 'not_equals' and value != threshold:
            triggered = True
        
        alert_id = f"{rule_id}_{hash(str(labels))}"
        
        if triggered:
            # 检查是否已经存在活跃告警
            if alert_id in self.active_alerts:
                # 更新现有告警
                alert = self.active_alerts[alert_id]
                alert.current_value = value
            else:
                # 创建新告警
                alert = Alert(
                    id=alert_id,
                    level=AlertLevel(rule['level']),
                    title=rule['title'],
                    description=rule['description'].format(
                        threshold=threshold,
                        duration=rule.get('duration', 0),
                        value=value
                    ),
                    service=labels.get('service', 'unknown'),
                    metric=rule['metric'],
                    threshold=threshold,
                    current_value=value,
                    created_at=datetime.now(),
                    tags=[rule_id] + list(labels.keys()),
                    metadata={'rule_id': rule_id, 'labels': labels}
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # 发送通知
                await self._send_alert_notification(alert)
                
                logger.warning(f"Alert triggered: {alert.title} (value: {value}, threshold: {threshold})")
        
        else:
            # 检查是否需要解决告警
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved_at = datetime.now()
                
                # 发送解决通知
                await self._send_resolution_notification(alert)
                
                # 移除活跃告警
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert.title}")
    
    async def _send_alert_notification(self, alert: Alert):
        """发送告警通知"""
        # 检查通知限流
        limit_key = f"{alert.service}_{alert.metric}"
        limit_info = self.notification_limits[limit_key]
        
        now = datetime.now()
        if limit_info['last_sent']:
            time_since_last = (now - limit_info['last_sent']).seconds
            if time_since_last < 300:  # 5分钟内限制通知
                limit_info['count'] += 1
                if limit_info['count'] > 3:  # 最多3次通知
                    return
            else:
                limit_info['count'] = 1
        else:
            limit_info['count'] = 1
        
        limit_info['last_sent'] = now
        
        # 发送到各个通知渠道
        for channel_id, channel in self.notification_channels.items():
            if channel['enabled']:
                try:
                    await self._send_to_channel(channel, alert, 'alert')
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel_id}: {e}")
    
    async def _send_resolution_notification(self, alert: Alert):
        """发送解决通知"""
        for channel_id, channel in self.notification_channels.items():
            if channel['enabled']:
                try:
                    await self._send_to_channel(channel, alert, 'resolution')
                except Exception as e:
                    logger.error(f"Failed to send resolution to {channel_id}: {e}")
    
    async def _send_to_channel(self, channel: Dict[str, Any], alert: Alert, notification_type: str):
        """发送到指定渠道"""
        channel_type = channel['type']
        config = channel['config']
        
        if channel_type == 'email':
            await self._send_email(config, alert, notification_type)
        elif channel_type == 'webhook':
            await self._send_webhook(config, alert, notification_type)
        elif channel_type == 'slack':
            await self._send_slack(config, alert, notification_type)
    
    async def _send_email(self, config: Dict[str, Any], alert: Alert, notification_type: str):
        """发送邮件通知"""
        if not config.get('username') or not config.get('recipients'):
            return
        
        subject = f"[{alert.level.value.upper()}] {alert.title}"
        if notification_type == 'resolution':
            subject = f"[RESOLVED] {alert.title}"
        
        body = f"""
        Alert Details:
        - Service: {alert.service}
        - Metric: {alert.metric}
        - Current Value: {alert.current_value}
        - Threshold: {alert.threshold}
        - Level: {alert.level.value}
        - Created: {alert.created_at}
        
        Description:
        {alert.description}
        
        Tags: {', '.join(alert.tags)}
        """
        
        if notification_type == 'resolution':
            body += f"\n\nResolved at: {alert.resolved_at}"
        
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = config['username']
        msg['To'] = ', '.join(config['recipients'])
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        # 发送邮件
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls(context=context)
                server.login(config['username'], config['password'])
                server.send_message(msg)
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    async def _send_webhook(self, config: Dict[str, Any], alert: Alert, notification_type: str):
        """发送Webhook通知"""
        if not config.get('url'):
            return
        
        payload = {
            'type': notification_type,
            'alert': {
                'id': alert.id,
                'level': alert.level.value,
                'title': alert.title,
                'description': alert.description,
                'service': alert.service,
                'metric': alert.metric,
                'threshold': alert.threshold,
                'current_value': alert.current_value,
                'created_at': alert.created_at.isoformat(),
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
                'tags': alert.tags,
                'metadata': alert.metadata
            }
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    config.get('method', 'POST'),
                    config['url'],
                    json=payload,
                    headers=config.get('headers', {}),
                    timeout=aiohttp.ClientTimeout(total=config.get('timeout', 10))
                ) as response:
                    if response.status >= 400:
                        logger.error(f"Webhook returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
    
    async def _send_slack(self, config: Dict[str, Any], alert: Alert, notification_type: str):
        """发送Slack通知"""
        if not config.get('webhook_url'):
            return
        
        color = {
            AlertLevel.INFO: 'good',
            AlertLevel.WARNING: 'warning',
            AlertLevel.CRITICAL: 'danger',
            AlertLevel.EMERGENCY: 'danger'
        }.get(alert.level, 'warning')
        
        if notification_type == 'resolution':
            color = 'good'
        
        payload = {
            'channel': config.get('channel', '#alerts'),
            'username': config.get('username', 'Xiaozhi Monitor'),
            'attachments': [{
                'color': color,
                'title': alert.title,
                'text': alert.description,
                'fields': [
                    {'title': 'Service', 'value': alert.service, 'short': True},
                    {'title': 'Metric', 'value': alert.metric, 'short': True},
                    {'title': 'Current Value', 'value': str(alert.current_value), 'short': True},
                    {'title': 'Threshold', 'value': str(alert.threshold), 'short': True},
                    {'title': 'Level', 'value': alert.level.value.upper(), 'short': True},
                    {'title': 'Status', 'value': notification_type.upper(), 'short': True}
                ],
                'timestamp': int(alert.created_at.timestamp())
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(config['webhook_url'], json=payload) as response:
                    if response.status >= 400:
                        logger.error(f"Slack webhook returned status {response.status}")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """确认告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = acknowledged_by
            logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return list(self.alert_history)[-limit:]

class SystemMonitor:
    """系统监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.last_collection = None
        
        # Prometheus指标
        self.cpu_usage_gauge = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.memory_usage_gauge = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.disk_usage_gauge = Gauge('system_disk_usage_percent', 'Disk usage percentage', ['device'])
        self.network_bytes_counter = Counter('system_network_bytes_total', 'Network bytes', ['direction', 'interface'])
        self.gpu_usage_gauge = Gauge('system_gpu_usage_percent', 'GPU usage percentage', ['gpu_id'])
        self.gpu_memory_gauge = Gauge('system_gpu_memory_usage_percent', 'GPU memory usage percentage', ['gpu_id'])
    
    async def collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        metrics = {}
        
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics['cpu_usage'] = cpu_percent
            self.cpu_usage_gauge.set(cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            metrics['memory_usage'] = memory_percent
            self.memory_usage_gauge.set(memory_percent)
            
            # 磁盘使用率
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            metrics['disk_usage'] = disk_percent
            self.disk_usage_gauge.labels(device='/').set(disk_percent)
            
            # 网络流量
            network = psutil.net_io_counters()
            metrics['network_bytes_sent'] = network.bytes_sent
            metrics['network_bytes_recv'] = network.bytes_recv
            
            # 更新网络计数器（需要计算差值）
            if self.last_collection:
                time_diff = time.time() - self.last_collection['timestamp']
                if time_diff > 0:
                    sent_rate = (network.bytes_sent - self.last_collection.get('network_bytes_sent', 0)) / time_diff
                    recv_rate = (network.bytes_recv - self.last_collection.get('network_bytes_recv', 0)) / time_diff
                    metrics['network_send_rate'] = sent_rate
                    metrics['network_recv_rate'] = recv_rate
            
            # GPU使用率（如果有GPU）
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    metrics[f'gpu_{i}_usage'] = gpu.load * 100
                    metrics[f'gpu_{i}_memory'] = gpu.memoryUtil * 100
                    metrics[f'gpu_{i}_temperature'] = gpu.temperature
                    
                    self.gpu_usage_gauge.labels(gpu_id=str(i)).set(gpu.load * 100)
                    self.gpu_memory_gauge.labels(gpu_id=str(i)).set(gpu.memoryUtil * 100)
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
            
            # 进程信息
            process_count = len(psutil.pids())
            metrics['process_count'] = process_count
            
            # 负载平均值
            load_avg = psutil.getloadavg()
            metrics['load_avg_1m'] = load_avg[0]
            metrics['load_avg_5m'] = load_avg[1]
            metrics['load_avg_15m'] = load_avg[2]
            
            # 更新最后收集时间
            self.last_collection = {
                'timestamp': time.time(),
                **metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics

class ServiceMonitor:
    """服务监控器"""
    
    def __init__(self):
        self.services = {}
        self.health_checks = {}
        
        # Prometheus指标
        self.service_up_gauge = Gauge('service_up', 'Service availability', ['service'])
        self.service_response_time_histogram = Histogram('service_response_time_seconds', 'Service response time', ['service'])
        self.service_requests_counter = Counter('service_requests_total', 'Service requests', ['service', 'status'])
    
    def register_service(self, service_name: str, health_check_url: str, 
                        check_interval: int = 30, timeout: int = 5):
        """注册服务"""
        self.services[service_name] = {
            'health_check_url': health_check_url,
            'check_interval': check_interval,
            'timeout': timeout,
            'last_check': None,
            'status': ServiceStatus.UNKNOWN,
            'response_time': 0.0,
            'error_count': 0,
            'total_checks': 0
        }
        
        logger.info(f"Registered service: {service_name}")
    
    async def check_service_health(self, service_name: str) -> ServiceHealth:
        """检查服务健康状态"""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not registered")
        
        service_config = self.services[service_name]
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=service_config['timeout'])
            ) as session:
                async with session.get(service_config['health_check_url']) as response:
                    response_time = (time.time() - start_time) * 1000  # 毫秒
                    
                    if response.status == 200:
                        status = ServiceStatus.HEALTHY
                        data = await response.json()
                    else:
                        status = ServiceStatus.DEGRADED
                        data = {}
                    
                    # 更新服务状态
                    service_config['last_check'] = datetime.now()
                    service_config['status'] = status
                    service_config['response_time'] = response_time
                    service_config['total_checks'] += 1
                    
                    # 更新Prometheus指标
                    self.service_up_gauge.labels(service=service_name).set(1 if status == ServiceStatus.HEALTHY else 0)
                    self.service_response_time_histogram.labels(service=service_name).observe(response_time / 1000)
                    self.service_requests_counter.labels(service=service_name, status='success').inc()
                    
                    # 计算错误率
                    error_rate = (service_config['error_count'] / service_config['total_checks']) * 100
                    
                    return ServiceHealth(
                        service_name=service_name,
                        status=status,
                        last_check=service_config['last_check'],
                        response_time=response_time,
                        error_rate=error_rate,
                        uptime=self._calculate_uptime(service_name),
                        version=data.get('version', ''),
                        dependencies=data.get('dependencies', []),
                        metrics=data.get('metrics', {})
                    )
                    
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            
            # 更新错误状态
            service_config['last_check'] = datetime.now()
            service_config['status'] = ServiceStatus.UNHEALTHY
            service_config['error_count'] += 1
            service_config['total_checks'] += 1
            
            # 更新Prometheus指标
            self.service_up_gauge.labels(service=service_name).set(0)
            self.service_requests_counter.labels(service=service_name, status='error').inc()
            
            error_rate = (service_config['error_count'] / service_config['total_checks']) * 100
            
            return ServiceHealth(
                service_name=service_name,
                status=ServiceStatus.UNHEALTHY,
                last_check=service_config['last_check'],
                response_time=0.0,
                error_rate=error_rate,
                uptime=self._calculate_uptime(service_name)
            )
    
    def _calculate_uptime(self, service_name: str) -> float:
        """计算服务正常运行时间百分比"""
        service_config = self.services[service_name]
        
        if service_config['total_checks'] == 0:
            return 0.0
        
        success_checks = service_config['total_checks'] - service_config['error_count']
        return (success_checks / service_config['total_checks']) * 100
    
    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """检查所有服务健康状态"""
        results = {}
        
        tasks = []
        for service_name in self.services:
            task = asyncio.create_task(self.check_service_health(service_name))
            tasks.append((service_name, task))
        
        for service_name, task in tasks:
            try:
                health = await task
                results[service_name] = health
            except Exception as e:
                logger.error(f"Failed to check {service_name}: {e}")
        
        return results

class AdvancedMonitoringSystem:
    """高级监控系统"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        
        # 组件
        self.system_monitor = SystemMonitor()
        self.service_monitor = ServiceMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.alert_manager = AlertManager(redis_url)
        
        # 监控状态
        self.is_running = False
        self.background_tasks = []
        
        # 数据存储
        self.metric_history = defaultdict(deque)
        self.performance_baselines = {}
        
        # 自动化运维
        self.automation_rules = {}
        self.maintenance_windows = []
    
    async def initialize(self):
        """初始化监控系统"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            logger.info("Monitoring system Redis client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
        
        # 初始化告警管理器
        await self.alert_manager.initialize()
        
        # 注册默认服务
        await self._register_default_services()
        
        # 启动后台任务
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._system_monitoring_loop()),
            asyncio.create_task(self._service_monitoring_loop()),
            asyncio.create_task(self._anomaly_detection_loop()),
            asyncio.create_task(self._data_retention_loop()),
            asyncio.create_task(self._automation_loop())
        ]
        
        logger.info("Advanced monitoring system initialized")
    
    async def shutdown(self):
        """关闭监控系统"""
        self.is_running = False
        
        # 取消后台任务
        for task in self.background_tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Monitoring system shutdown completed")
    
    async def _register_default_services(self):
        """注册默认服务"""
        default_services = [
            ('vad_service', 'http://localhost:8001/health'),
            ('asr_service', 'http://localhost:8002/health'),
            ('llm_service', 'http://localhost:8003/health'),
            ('tts_service', 'http://localhost:8004/health'),
            ('load_balancer', 'http://localhost:8080/health')
        ]
        
        for service_name, health_url in default_services:
            self.service_monitor.register_service(service_name, health_url)
    
    async def _system_monitoring_loop(self):
        """系统监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                metrics = await self.system_monitor.collect_system_metrics()
                
                # 存储指标
                await self._store_metrics('system', metrics)
                
                # 检查告警
                for metric_name, value in metrics.items():
                    await self.alert_manager.evaluate_metric(
                        metric_name, value, {'service': 'system'}
                    )
                
                await asyncio.sleep(30)  # 每30秒收集一次
                
            except Exception as e:
                logger.error(f"System monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _service_monitoring_loop(self):
        """服务监控循环"""
        while self.is_running:
            try:
                # 检查所有服务健康状态
                health_results = await self.service_monitor.check_all_services()
                
                # 处理健康检查结果
                for service_name, health in health_results.items():
                    # 存储服务指标
                    service_metrics = {
                        'service_availability': 1.0 if health.status == ServiceStatus.HEALTHY else 0.0,
                        'response_time': health.response_time,
                        'error_rate': health.error_rate,
                        'uptime': health.uptime
                    }
                    
                    await self._store_metrics(f'service_{service_name}', service_metrics)
                    
                    # 检查告警
                    for metric_name, value in service_metrics.items():
                        await self.alert_manager.evaluate_metric(
                            metric_name, value, {'service': service_name}
                        )
                
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"Service monitoring loop error: {e}")
                await asyncio.sleep(30)
    
    async def _anomaly_detection_loop(self):
        """异常检测循环"""
        while self.is_running:
            try:
                # 获取最近的指标数据
                recent_metrics = await self._get_recent_metrics()
                
                # 执行异常检测
                for metric_name, values in recent_metrics.items():
                    if values:
                        latest_value = values[-1]['value']
                        anomaly = self.anomaly_detector.detect_anomaly(metric_name, latest_value)
                        
                        if anomaly:
                            # 创建异常告警
                            await self._create_anomaly_alert(anomaly)
                
                await asyncio.sleep(120)  # 每2分钟检测一次
                
            except Exception as e:
                logger.error(f"Anomaly detection loop error: {e}")
                await asyncio.sleep(60)
    
    async def _data_retention_loop(self):
        """数据保留循环"""
        while self.is_running:
            try:
                # 清理过期数据
                await self._cleanup_old_data()
                
                # 压缩历史数据
                await self._compress_historical_data()
                
                await asyncio.sleep(3600)  # 每小时执行一次
                
            except Exception as e:
                logger.error(f"Data retention loop error: {e}")
                await asyncio.sleep(1800)
    
    async def _automation_loop(self):
        """自动化运维循环"""
        while self.is_running:
            try:
                # 执行自动化规则
                await self._execute_automation_rules()
                
                # 检查维护窗口
                await self._check_maintenance_windows()
                
                await asyncio.sleep(300)  # 每5分钟执行一次
                
            except Exception as e:
                logger.error(f"Automation loop error: {e}")
                await asyncio.sleep(180)
    
    async def _store_metrics(self, category: str, metrics: Dict[str, float]):
        """存储指标数据"""
        if not self.redis_client:
            return
        
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            # 添加到异常检测器
            self.anomaly_detector.add_metric(metric_name, value, timestamp)
            
            # 存储到Redis
            key = f"metrics:{category}:{metric_name}"
            data = {
                'timestamp': timestamp.isoformat(),
                'value': value
            }
            
            await self.redis_client.lpush(key, json.dumps(data))
            await self.redis_client.ltrim(key, 0, 1000)  # 保持最近1000个数据点
    
    async def _get_recent_metrics(self, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """获取最近的指标数据"""
        if not self.redis_client:
            return {}
        
        metrics = {}
        
        # 获取所有指标键
        keys = await self.redis_client.keys("metrics:*")
        
        for key in keys:
            try:
                data = await self.redis_client.lrange(key, 0, limit - 1)
                metric_name = key.decode().split(':')[-1]
                
                metrics[metric_name] = []
                for item in data:
                    point = json.loads(item)
                    metrics[metric_name].append(point)
                
            except Exception as e:
                logger.error(f"Failed to get metrics for {key}: {e}")
        
        return metrics
    
    async def _create_anomaly_alert(self, anomaly: Dict[str, Any]):
        """创建异常告警"""
        alert_id = f"anomaly_{anomaly['metric_name']}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            level=AlertLevel.WARNING,
            title=f"Anomaly Detected: {anomaly['metric_name']}",
            description=f"Anomalous value {anomaly['value']} detected for {anomaly['metric_name']} "
                       f"(confidence: {anomaly['confidence']:.2f}, type: {anomaly['anomaly_type']})",
            service='system',
            metric=anomaly['metric_name'],
            threshold=0.0,
            current_value=anomaly['value'],
            created_at=datetime.now(),
            tags=['anomaly', anomaly['anomaly_type']],
            metadata=anomaly
        )
        
        # 添加到活跃告警
        self.alert_manager.active_alerts[alert_id] = alert
        self.alert_manager.alert_history.append(alert)
        
        # 发送通知
        await self.alert_manager._send_alert_notification(alert)
        
        logger.warning(f"Anomaly alert created: {alert.title}")
    
    async def _cleanup_old_data(self):
        """清理过期数据"""
        if not self.redis_client:
            return
        
        # 清理超过7天的指标数据
        cutoff_time = datetime.now() - timedelta(days=7)
        
        keys = await self.redis_client.keys("metrics:*")
        
        for key in keys:
            try:
                # 获取所有数据
                data = await self.redis_client.lrange(key, 0, -1)
                
                # 过滤过期数据
                valid_data = []
                for item in data:
                    point = json.loads(item)
                    point_time = datetime.fromisoformat(point['timestamp'])
                    
                    if point_time > cutoff_time:
                        valid_data.append(item)
                
                # 重新设置数据
                if valid_data:
                    await self.redis_client.delete(key)
                    await self.redis_client.lpush(key, *valid_data)
                else:
                    await self.redis_client.delete(key)
                
            except Exception as e:
                logger.error(f"Failed to cleanup data for {key}: {e}")
    
    async def _compress_historical_data(self):
        """压缩历史数据"""
        # 这里可以实现数据压缩逻辑
        # 例如：将小时级数据聚合为天级数据
        pass
    
    async def _execute_automation_rules(self):
        """执行自动化规则"""
        # 这里可以实现自动化运维规则
        # 例如：自动重启失败的服务、自动扩容等
        pass
    
    async def _check_maintenance_windows(self):
        """检查维护窗口"""
        # 这里可以实现维护窗口检查逻辑
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'monitoring_active': self.is_running,
            'active_alerts': len(self.alert_manager.active_alerts),
            'total_services': len(self.service_monitor.services),
            'anomaly_count': len(self.anomaly_detector.anomalies),
            'last_system_check': self.system_monitor.last_collection['timestamp'] if self.system_monitor.last_collection else None
        }

# FastAPI应用
app = FastAPI(title="Advanced Monitoring System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局监控系统实例
monitoring_system = AdvancedMonitoringSystem()

@app.on_event("startup")
async def startup_event():
    await monitoring_system.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await monitoring_system.shutdown()

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "monitoring_active": monitoring_system.is_running
    }

@app.get("/status")
async def get_status():
    """获取系统状态"""
    return monitoring_system.get_system_status()

@app.get("/alerts")
async def get_alerts():
    """获取告警信息"""
    return {
        "active_alerts": [asdict(alert) for alert in monitoring_system.alert_manager.get_active_alerts()],
        "alert_history": [asdict(alert) for alert in monitoring_system.alert_manager.get_alert_history(50)]
    }

@app.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(alert_id: str, data: Dict[str, str]):
    """确认告警"""
    acknowledged_by = data.get('acknowledged_by', 'unknown')
    monitoring_system.alert_manager.acknowledge_alert(alert_id, acknowledged_by)
    return {"status": "success", "message": f"Alert {alert_id} acknowledged"}

@app.get("/metrics")
async def get_metrics():
    """获取指标数据"""
    recent_metrics = await monitoring_system._get_recent_metrics(100)
    return recent_metrics

@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """获取Prometheus格式的指标"""
    return generate_latest()

@app.post("/services/register")
async def register_service(service_data: Dict[str, Any]):
    """注册服务"""
    monitoring_system.service_monitor.register_service(
        service_data['name'],
        service_data['health_check_url'],
        service_data.get('check_interval', 30),
        service_data.get('timeout', 5)
    )
    return {"status": "success", "message": f"Service {service_data['name']} registered"}

@app.get("/services/health")
async def get_services_health():
    """获取所有服务健康状态"""
    health_results = await monitoring_system.service_monitor.check_all_services()
    return {
        service_name: asdict(health)
        for service_name, health in health_results.items()
    }

@app.post("/alerts/rules")
async def add_alert_rule(rule_data: Dict[str, Any]):
    """添加告警规则"""
    rule_id = rule_data['rule_id']
    rule_config = {k: v for k, v in rule_data.items() if k != 'rule_id'}
    
    monitoring_system.alert_manager.add_alert_rule(rule_id, rule_config)
    return {"status": "success", "message": f"Alert rule {rule_id} added"}

@app.delete("/alerts/rules/{rule_id}")
async def remove_alert_rule(rule_id: str):
    """移除告警规则"""
    monitoring_system.alert_manager.remove_alert_rule(rule_id)
    return {"status": "success", "message": f"Alert rule {rule_id} removed"}

@app.get("/anomalies")
async def get_anomalies():
    """获取异常检测结果"""
    return {
        "anomalies": monitoring_system.anomaly_detector.anomalies[-100:],  # 最近100个异常
        "baselines": monitoring_system.anomaly_detector.baselines
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)