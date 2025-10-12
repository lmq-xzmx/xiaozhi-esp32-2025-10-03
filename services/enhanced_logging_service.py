#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强日志记录服务
提供结构化日志、日志轮转、日志分析和日志聚合功能
"""

import asyncio
import json
import logging
import logging.handlers
import os
import sys
import gzip
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import re
from collections import defaultdict, Counter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.redis_config import get_redis_client
from core.enhanced_db_service import get_enhanced_db_service

class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogCategory(Enum):
    """日志分类"""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER_ACTION = "user_action"
    API = "api"
    WEBSOCKET = "websocket"
    CACHE = "cache"

@dataclass
class LogContext:
    """日志上下文"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class StructuredLogRecord:
    """结构化日志记录"""
    timestamp: datetime
    level: LogLevel
    category: LogCategory
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    context: LogContext
    tags: List[str]
    correlation_id: Optional[str] = None

class CustomJSONFormatter(logging.Formatter):
    """自定义JSON格式化器"""
    
    def format(self, record):
        """格式化日志记录"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'process': record.process
        }
        
        # 添加自定义字段
        if hasattr(record, 'category'):
            log_data['category'] = record.category
        if hasattr(record, 'context'):
            log_data['context'] = record.context
        if hasattr(record, 'tags'):
            log_data['tags'] = record.tags
        if hasattr(record, 'correlation_id'):
            log_data['correlation_id'] = record.correlation_id
        
        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)

class CompressedTimedRotatingFileHandler(logging.handlers.TimedRotatingFileHandler):
    """支持压缩的定时轮转文件处理器"""
    
    def __init__(self, filename, when='h', interval=1, backupCount=0, 
                 encoding=None, delay=False, utc=False, atTime=None, 
                 compress=True):
        super().__init__(filename, when, interval, backupCount, 
                        encoding, delay, utc, atTime)
        self.compress = compress
    
    def doRollover(self):
        """执行日志轮转"""
        super().doRollover()
        
        if self.compress and self.backupCount > 0:
            # 压缩最新的备份文件
            for i in range(1, self.backupCount + 1):
                sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                if os.path.exists(sfn) and not sfn.endswith('.gz'):
                    self._compress_file(sfn)
    
    def _compress_file(self, filename):
        """压缩文件"""
        try:
            with open(filename, 'rb') as f_in:
                with gzip.open(f"{filename}.gz", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(filename)
        except Exception as e:
            print(f"压缩日志文件失败: {e}")

class EnhancedLoggingService:
    """增强日志记录服务"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.redis_client = None
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_cache_ttl = 86400 * 3  # 3天
        self.max_log_size = 100 * 1024 * 1024  # 100MB
        self.backup_count = 30  # 保留30个备份文件
        
        # 日志分析配置
        self.analysis_patterns = {
            'error_patterns': [
                r'error|exception|failed|failure',
                r'timeout|connection.*refused',
                r'permission.*denied|access.*denied',
                r'not.*found|missing|invalid'
            ],
            'performance_patterns': [
                r'slow.*query|query.*timeout',
                r'high.*memory|memory.*leak',
                r'cpu.*usage|high.*load'
            ],
            'security_patterns': [
                r'unauthorized|forbidden|authentication.*failed',
                r'sql.*injection|xss|csrf',
                r'brute.*force|suspicious.*activity'
            ]
        }
    
    async def initialize(self):
        """初始化日志服务"""
        try:
            self.redis_client = get_redis_client()
            await self.redis_client.ping()
            
            # 设置默认日志记录器
            self._setup_loggers()
            
            # 启动日志分析任务
            asyncio.create_task(self._log_analysis_daemon())
            
            # 启动日志清理任务
            asyncio.create_task(self._log_cleanup_daemon())
            
            self.get_logger("system").info("增强日志服务初始化成功")
            return True
        except Exception as e:
            print(f"增强日志服务初始化失败: {e}")
            return False
    
    def _setup_loggers(self):
        """设置日志记录器"""
        # 为每个分类创建专门的日志记录器
        for category in LogCategory:
            logger_name = f"xiaozhi.{category.value}"
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            
            # 清除现有处理器
            logger.handlers.clear()
            
            # 文件处理器（JSON格式）
            json_file = self.log_dir / f"{category.value}.json.log"
            json_handler = CompressedTimedRotatingFileHandler(
                str(json_file),
                when='midnight',
                interval=1,
                backupCount=self.backup_count,
                encoding='utf-8',
                compress=True
            )
            json_handler.setFormatter(CustomJSONFormatter())
            logger.addHandler(json_handler)
            
            # 文本文件处理器（人类可读格式）
            text_file = self.log_dir / f"{category.value}.log"
            text_handler = CompressedTimedRotatingFileHandler(
                str(text_file),
                when='midnight',
                interval=1,
                backupCount=self.backup_count,
                encoding='utf-8',
                compress=True
            )
            text_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            text_handler.setFormatter(text_formatter)
            logger.addHandler(text_handler)
            
            # 控制台处理器（仅ERROR及以上级别）
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            self.loggers[category.value] = logger
    
    def get_logger(self, category: Union[str, LogCategory]) -> logging.Logger:
        """获取指定分类的日志记录器"""
        if isinstance(category, LogCategory):
            category = category.value
        
        return self.loggers.get(category, logging.getLogger(f"xiaozhi.{category}"))
    
    async def log_structured(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        context: Optional[LogContext] = None,
        tags: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        **kwargs
    ):
        """记录结构化日志"""
        try:
            logger = self.get_logger(category)
            
            # 创建日志记录
            record = StructuredLogRecord(
                timestamp=datetime.now(),
                level=level,
                category=category,
                message=message,
                logger_name=logger.name,
                module=kwargs.get('module', ''),
                function=kwargs.get('function', ''),
                line_number=kwargs.get('line_number', 0),
                context=context or LogContext(),
                tags=tags or [],
                correlation_id=correlation_id
            )
            
            # 添加自定义字段到日志记录
            log_record = logger.makeRecord(
                logger.name, getattr(logging, level.value), 
                '', 0, message, (), None
            )
            
            log_record.category = category.value
            log_record.context = asdict(record.context)
            log_record.tags = record.tags
            log_record.correlation_id = record.correlation_id
            
            # 记录日志
            logger.handle(log_record)
            
            # 存储到Redis（用于实时分析）
            await self._store_to_redis(record)
            
        except Exception as e:
            # 使用标准日志记录错误，避免递归
            logging.getLogger("xiaozhi.system").error(f"记录结构化日志失败: {e}")
    
    async def _store_to_redis(self, record: StructuredLogRecord):
        """存储日志到Redis"""
        try:
            if not self.redis_client:
                return
            
            # 转换为字典
            log_dict = asdict(record)
            log_dict['timestamp'] = record.timestamp.isoformat()
            log_dict['level'] = record.level.value
            log_dict['category'] = record.category.value
            
            # 存储到Redis列表
            key = f"logs:{record.category.value}:{datetime.now().strftime('%Y%m%d')}"
            await self.redis_client.lpush(key, json.dumps(log_dict))
            await self.redis_client.expire(key, self.log_cache_ttl)
            
            # 限制列表长度
            await self.redis_client.ltrim(key, 0, 9999)  # 保留最新10000条
            
        except Exception as e:
            logging.getLogger("xiaozhi.system").error(f"存储日志到Redis失败: {e}")
    
    async def search_logs(
        self,
        category: Optional[LogCategory] = None,
        level: Optional[LogLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        keyword: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """搜索日志"""
        try:
            logs = []
            
            # 确定搜索范围
            if start_time is None:
                start_time = datetime.now() - timedelta(days=1)
            if end_time is None:
                end_time = datetime.now()
            
            # 生成日期范围
            current_date = start_time.date()
            end_date = end_time.date()
            
            while current_date <= end_date:
                date_str = current_date.strftime('%Y%m%d')
                
                # 确定要搜索的分类
                categories = [category.value] if category else [cat.value for cat in LogCategory]
                
                for cat in categories:
                    key = f"logs:{cat}:{date_str}"
                    log_entries = await self.redis_client.lrange(key, 0, -1)
                    
                    for entry in log_entries:
                        try:
                            log_data = json.loads(entry)
                            log_time = datetime.fromisoformat(log_data['timestamp'])
                            
                            # 时间过滤
                            if log_time < start_time or log_time > end_time:
                                continue
                            
                            # 级别过滤
                            if level and log_data.get('level') != level.value:
                                continue
                            
                            # 关键词过滤
                            if keyword:
                                message = log_data.get('message', '').lower()
                                if keyword.lower() not in message:
                                    continue
                            
                            logs.append(log_data)
                            
                        except Exception:
                            continue
                
                current_date += timedelta(days=1)
            
            # 按时间排序并限制数量
            logs.sort(key=lambda x: x['timestamp'], reverse=True)
            return logs[:limit]
            
        except Exception as e:
            logging.getLogger("xiaozhi.system").error(f"搜索日志失败: {e}")
            return []
    
    async def analyze_logs(
        self,
        category: Optional[LogCategory] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """分析日志"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # 获取日志
            logs = await self.search_logs(
                category=category,
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            
            # 基本统计
            total_logs = len(logs)
            level_counts = Counter(log.get('level', 'UNKNOWN') for log in logs)
            category_counts = Counter(log.get('category', 'unknown') for log in logs)
            
            # 错误分析
            error_logs = [log for log in logs if log.get('level') in ['ERROR', 'CRITICAL']]
            error_patterns = self._analyze_patterns(error_logs, 'error_patterns')
            
            # 性能分析
            performance_issues = self._analyze_patterns(logs, 'performance_patterns')
            
            # 安全分析
            security_issues = self._analyze_patterns(logs, 'security_patterns')
            
            # 时间趋势分析
            hourly_counts = self._analyze_time_trends(logs, hours)
            
            # 热点分析
            top_errors = self._get_top_errors(error_logs)
            top_endpoints = self._get_top_endpoints(logs)
            
            return {
                'analysis_period': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'hours': hours
                },
                'summary': {
                    'total_logs': total_logs,
                    'error_count': len(error_logs),
                    'error_rate': len(error_logs) / total_logs if total_logs > 0 else 0
                },
                'level_distribution': dict(level_counts),
                'category_distribution': dict(category_counts),
                'error_patterns': error_patterns,
                'performance_issues': performance_issues,
                'security_issues': security_issues,
                'time_trends': hourly_counts,
                'top_errors': top_errors,
                'top_endpoints': top_endpoints
            }
            
        except Exception as e:
            logging.getLogger("xiaozhi.system").error(f"分析日志失败: {e}")
            return {}
    
    def _analyze_patterns(self, logs: List[Dict], pattern_type: str) -> List[Dict[str, Any]]:
        """分析日志模式"""
        patterns = self.analysis_patterns.get(pattern_type, [])
        results = []
        
        for pattern in patterns:
            matches = []
            for log in logs:
                message = log.get('message', '').lower()
                if re.search(pattern, message, re.IGNORECASE):
                    matches.append(log)
            
            if matches:
                results.append({
                    'pattern': pattern,
                    'match_count': len(matches),
                    'recent_examples': matches[:5]  # 最近5个例子
                })
        
        return sorted(results, key=lambda x: x['match_count'], reverse=True)
    
    def _analyze_time_trends(self, logs: List[Dict], hours: int) -> Dict[str, int]:
        """分析时间趋势"""
        hourly_counts = defaultdict(int)
        
        for log in logs:
            try:
                timestamp = datetime.fromisoformat(log['timestamp'])
                hour_key = timestamp.strftime('%Y-%m-%d %H:00')
                hourly_counts[hour_key] += 1
            except Exception:
                continue
        
        return dict(hourly_counts)
    
    def _get_top_errors(self, error_logs: List[Dict]) -> List[Dict[str, Any]]:
        """获取最频繁的错误"""
        error_messages = [log.get('message', '') for log in error_logs]
        error_counts = Counter(error_messages)
        
        return [
            {'message': message, 'count': count}
            for message, count in error_counts.most_common(10)
        ]
    
    def _get_top_endpoints(self, logs: List[Dict]) -> List[Dict[str, Any]]:
        """获取最活跃的端点"""
        endpoints = []
        for log in logs:
            context = log.get('context', {})
            endpoint = context.get('endpoint')
            if endpoint:
                endpoints.append(endpoint)
        
        endpoint_counts = Counter(endpoints)
        return [
            {'endpoint': endpoint, 'count': count}
            for endpoint, count in endpoint_counts.most_common(10)
        ]
    
    async def get_log_statistics(self) -> Dict[str, Any]:
        """获取日志统计信息"""
        try:
            stats = {}
            
            # 获取各分类的日志数量
            for category in LogCategory:
                today = datetime.now().strftime('%Y%m%d')
                key = f"logs:{category.value}:{today}"
                count = await self.redis_client.llen(key)
                stats[f"{category.value}_count"] = count
            
            # 获取磁盘使用情况
            total_size = 0
            file_count = 0
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.is_file():
                    total_size += log_file.stat().st_size
                    file_count += 1
            
            stats.update({
                'disk_usage_bytes': total_size,
                'disk_usage_mb': round(total_size / (1024 * 1024), 2),
                'log_file_count': file_count,
                'log_directory': str(self.log_dir)
            })
            
            return stats
            
        except Exception as e:
            logging.getLogger("xiaozhi.system").error(f"获取日志统计失败: {e}")
            return {}
    
    async def _log_analysis_daemon(self):
        """日志分析守护进程"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时运行一次
                
                # 执行自动分析
                analysis = await self.analyze_logs(hours=1)
                
                # 检查是否有严重问题
                if analysis.get('summary', {}).get('error_rate', 0) > 0.1:  # 错误率超过10%
                    await self.log_structured(
                        LogLevel.WARNING,
                        LogCategory.SYSTEM,
                        f"检测到高错误率: {analysis['summary']['error_rate']:.2%}",
                        tags=['auto_analysis', 'high_error_rate']
                    )
                
            except Exception as e:
                logging.getLogger("xiaozhi.system").error(f"日志分析守护进程错误: {e}")
    
    async def _log_cleanup_daemon(self):
        """日志清理守护进程"""
        while True:
            try:
                await asyncio.sleep(86400)  # 每天运行一次
                await self._cleanup_old_logs()
            except Exception as e:
                logging.getLogger("xiaozhi.system").error(f"日志清理守护进程错误: {e}")
    
    async def _cleanup_old_logs(self):
        """清理旧日志"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.backup_count)
            cutoff_str = cutoff_date.strftime('%Y%m%d')
            
            # 清理Redis中的旧日志
            for category in LogCategory:
                pattern = f"logs:{category.value}:*"
                keys = await self.redis_client.keys(pattern)
                
                for key in keys:
                    date_part = key.split(':')[-1]
                    if date_part < cutoff_str:
                        await self.redis_client.delete(key)
            
            # 清理磁盘上的旧日志文件
            for log_file in self.log_dir.glob("*.log.*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
            
            await self.log_structured(
                LogLevel.INFO,
                LogCategory.SYSTEM,
                f"清理了 {cutoff_date.strftime('%Y-%m-%d')} 之前的旧日志",
                tags=['cleanup', 'maintenance']
            )
            
        except Exception as e:
            logging.getLogger("xiaozhi.system").error(f"清理旧日志失败: {e}")
    
    async def export_logs(
        self,
        category: Optional[LogCategory] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        format: str = 'json'
    ) -> str:
        """导出日志"""
        try:
            logs = await self.search_logs(category, None, start_time, end_time, None, 10000)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            category_str = category.value if category else 'all'
            filename = f"logs_export_{category_str}_{timestamp}.{format}"
            filepath = self.log_dir / filename
            
            if format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(logs, f, ensure_ascii=False, indent=2)
            elif format == 'csv':
                import csv
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    if logs:
                        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
                        writer.writeheader()
                        writer.writerows(logs)
            
            return str(filepath)
            
        except Exception as e:
            logging.getLogger("xiaozhi.system").error(f"导出日志失败: {e}")
            return ""
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查Redis连接
            redis_ok = await self.redis_client.ping() if self.redis_client else False
            
            # 检查日志目录
            log_dir_ok = self.log_dir.exists() and self.log_dir.is_dir()
            
            # 检查磁盘空间
            disk_usage = self.log_dir.stat().st_size if log_dir_ok else 0
            disk_ok = disk_usage < self.max_log_size
            
            # 获取统计信息
            stats = await self.get_log_statistics()
            
            status = 'healthy' if all([redis_ok, log_dir_ok, disk_ok]) else 'unhealthy'
            
            return {
                'status': status,
                'redis_connection': redis_ok,
                'log_directory': log_dir_ok,
                'disk_usage_ok': disk_ok,
                'statistics': stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# 全局日志服务实例
_logging_service = None

async def get_logging_service() -> EnhancedLoggingService:
    """获取日志服务实例"""
    global _logging_service
    if _logging_service is None:
        _logging_service = EnhancedLoggingService()
        await _logging_service.initialize()
    return _logging_service

# 便捷函数
async def log_info(category: LogCategory, message: str, context: Optional[LogContext] = None, **kwargs):
    """记录INFO级别日志"""
    service = await get_logging_service()
    await service.log_structured(LogLevel.INFO, category, message, context, **kwargs)

async def log_warning(category: LogCategory, message: str, context: Optional[LogContext] = None, **kwargs):
    """记录WARNING级别日志"""
    service = await get_logging_service()
    await service.log_structured(LogLevel.WARNING, category, message, context, **kwargs)

async def log_error(category: LogCategory, message: str, context: Optional[LogContext] = None, **kwargs):
    """记录ERROR级别日志"""
    service = await get_logging_service()
    await service.log_structured(LogLevel.ERROR, category, message, context, **kwargs)

async def log_critical(category: LogCategory, message: str, context: Optional[LogContext] = None, **kwargs):
    """记录CRITICAL级别日志"""
    service = await get_logging_service()
    await service.log_structured(LogLevel.CRITICAL, category, message, context, **kwargs)