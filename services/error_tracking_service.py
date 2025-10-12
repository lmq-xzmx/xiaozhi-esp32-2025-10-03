#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
错误追踪服务
提供错误捕获、记录、分析和报告功能
"""

import asyncio
import json
import traceback
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.redis_config import get_redis_client
from core.enhanced_db_service import get_enhanced_db_service

class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """错误分类"""
    SYSTEM = "system"
    DATABASE = "database"
    NETWORK = "network"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_API = "external_api"
    UNKNOWN = "unknown"

@dataclass
class ErrorContext:
    """错误上下文信息"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

@dataclass
class ErrorRecord:
    """错误记录"""
    id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    exception_type: str
    stack_trace: str
    context: ErrorContext
    resolved: bool = False
    resolution_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    occurrence_count: int = 1
    first_occurrence: datetime = None
    last_occurrence: datetime = None

    def __post_init__(self):
        if self.first_occurrence is None:
            self.first_occurrence = self.timestamp
        if self.last_occurrence is None:
            self.last_occurrence = self.timestamp

class ErrorTrackingService:
    """错误追踪服务"""
    
    def __init__(self):
        self.redis_client = get_redis_client()
        self.logger = logging.getLogger(__name__)
        self.error_handlers: Dict[ErrorCategory, List[Callable]] = {}
        self.error_cache_ttl = 86400 * 7  # 7天
        self.max_stack_trace_length = 5000
        self.max_errors_per_minute = 100
        self.error_count_window = {}
        
    async def initialize(self):
        """初始化错误追踪服务"""
        try:
            # 确保Redis连接正常
            await self.redis_client.ping()
            
            # 设置默认错误处理器
            self._setup_default_handlers()
            
            # 启动清理任务
            asyncio.create_task(self._cleanup_daemon())
            
            self.logger.info("错误追踪服务初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"错误追踪服务初始化失败: {e}")
            return False
    
    def _setup_default_handlers(self):
        """设置默认错误处理器"""
        # 数据库错误处理器
        self.register_error_handler(ErrorCategory.DATABASE, self._handle_database_error)
        
        # 网络错误处理器
        self.register_error_handler(ErrorCategory.NETWORK, self._handle_network_error)
        
        # 系统错误处理器
        self.register_error_handler(ErrorCategory.SYSTEM, self._handle_system_error)
    
    def register_error_handler(self, category: ErrorCategory, handler: Callable):
        """注册错误处理器"""
        if category not in self.error_handlers:
            self.error_handlers[category] = []
        self.error_handlers[category].append(handler)
    
    async def capture_error(
        self,
        exception: Exception,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[ErrorContext] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """捕获错误"""
        try:
            # 检查错误频率限制
            if not await self._check_rate_limit():
                return None
            
            # 生成错误ID
            error_id = str(uuid.uuid4())
            
            # 获取堆栈跟踪
            stack_trace = self._get_stack_trace(exception)
            
            # 创建错误记录
            error_record = ErrorRecord(
                id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                message=str(exception),
                exception_type=type(exception).__name__,
                stack_trace=stack_trace,
                context=context or ErrorContext(),
            )
            
            # 添加额外数据
            if additional_data:
                if error_record.context.additional_data is None:
                    error_record.context.additional_data = {}
                error_record.context.additional_data.update(additional_data)
            
            # 检查是否为重复错误
            existing_error_id = await self._check_duplicate_error(error_record)
            if existing_error_id:
                await self._increment_error_count(existing_error_id)
                return existing_error_id
            
            # 存储错误记录
            await self._store_error_record(error_record)
            
            # 触发错误处理器
            await self._trigger_error_handlers(error_record)
            
            # 记录日志
            self.logger.error(
                f"错误已捕获 [ID: {error_id}] [严重程度: {severity.value}] "
                f"[分类: {category.value}] {str(exception)}"
            )
            
            return error_id
            
        except Exception as e:
            self.logger.error(f"捕获错误时发生异常: {e}")
            return None
    
    async def capture_exception_from_context(
        self,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        context: Optional[ErrorContext] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """从当前异常上下文捕获错误"""
        exc_type, exc_value, exc_traceback = sys.exc_info()
        if exc_value:
            return await self.capture_error(
                exc_value, severity, category, context, additional_data
            )
        return None
    
    def _get_stack_trace(self, exception: Exception) -> str:
        """获取堆栈跟踪"""
        try:
            stack_trace = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
            
            # 限制堆栈跟踪长度
            if len(stack_trace) > self.max_stack_trace_length:
                stack_trace = stack_trace[:self.max_stack_trace_length] + "\n... (truncated)"
            
            return stack_trace
        except Exception:
            return "无法获取堆栈跟踪"
    
    async def _check_rate_limit(self) -> bool:
        """检查错误频率限制"""
        try:
            current_minute = datetime.now().replace(second=0, microsecond=0)
            key = f"error_count:{current_minute.isoformat()}"
            
            current_count = await self.redis_client.get(key)
            current_count = int(current_count) if current_count else 0
            
            if current_count >= self.max_errors_per_minute:
                return False
            
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, 60)
            
            return True
        except Exception:
            return True  # 如果检查失败，允许记录错误
    
    async def _check_duplicate_error(self, error_record: ErrorRecord) -> Optional[str]:
        """检查重复错误"""
        try:
            # 创建错误指纹
            fingerprint = self._create_error_fingerprint(error_record)
            
            # 查找相同指纹的错误
            existing_error_id = await self.redis_client.get(f"error_fingerprint:{fingerprint}")
            
            return existing_error_id
        except Exception:
            return None
    
    def _create_error_fingerprint(self, error_record: ErrorRecord) -> str:
        """创建错误指纹"""
        # 使用异常类型、消息和部分堆栈跟踪创建指纹
        stack_lines = error_record.stack_trace.split('\n')[:5]  # 只取前5行
        fingerprint_data = {
            'exception_type': error_record.exception_type,
            'message': error_record.message[:200],  # 只取前200个字符
            'stack_sample': '\n'.join(stack_lines)
        }
        
        fingerprint = json.dumps(fingerprint_data, sort_keys=True)
        return str(hash(fingerprint))
    
    async def _increment_error_count(self, error_id: str):
        """增加错误计数"""
        try:
            error_key = f"error:{error_id}"
            error_data = await self.redis_client.get(error_key)
            
            if error_data:
                error_record = json.loads(error_data)
                error_record['occurrence_count'] += 1
                error_record['last_occurrence'] = datetime.now().isoformat()
                
                await self.redis_client.set(
                    error_key, 
                    json.dumps(error_record), 
                    ex=self.error_cache_ttl
                )
        except Exception as e:
            self.logger.error(f"增加错误计数失败: {e}")
    
    async def _store_error_record(self, error_record: ErrorRecord):
        """存储错误记录"""
        try:
            # 转换为字典
            error_dict = asdict(error_record)
            error_dict['timestamp'] = error_record.timestamp.isoformat()
            error_dict['first_occurrence'] = error_record.first_occurrence.isoformat()
            error_dict['last_occurrence'] = error_record.last_occurrence.isoformat()
            error_dict['severity'] = error_record.severity.value
            error_dict['category'] = error_record.category.value
            
            # 存储到Redis
            error_key = f"error:{error_record.id}"
            await self.redis_client.set(
                error_key, 
                json.dumps(error_dict), 
                ex=self.error_cache_ttl
            )
            
            # 添加到错误列表
            await self.redis_client.lpush("error_list", error_record.id)
            
            # 设置错误指纹映射
            fingerprint = self._create_error_fingerprint(error_record)
            await self.redis_client.set(
                f"error_fingerprint:{fingerprint}", 
                error_record.id, 
                ex=self.error_cache_ttl
            )
            
            # 存储到数据库（可选）
            await self._store_to_database(error_record)
            
        except Exception as e:
            self.logger.error(f"存储错误记录失败: {e}")
    
    async def _store_to_database(self, error_record: ErrorRecord):
        """存储错误记录到数据库"""
        try:
            db_service = get_enhanced_db_service()
            
            # 创建错误表（如果不存在）
            create_table_query = """
                CREATE TABLE IF NOT EXISTS error_logs (
                    id VARCHAR(36) PRIMARY KEY,
                    timestamp DATETIME,
                    severity VARCHAR(20),
                    category VARCHAR(50),
                    message TEXT,
                    exception_type VARCHAR(100),
                    stack_trace TEXT,
                    context_data JSON,
                    resolved BOOLEAN DEFAULT FALSE,
                    occurrence_count INT DEFAULT 1,
                    first_occurrence DATETIME,
                    last_occurrence DATETIME,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            db_service.execute_update(create_table_query)
            
            # 插入错误记录
            context_json = json.dumps(asdict(error_record.context))
            insert_query = """
                INSERT INTO error_logs 
                (id, timestamp, severity, category, message, exception_type, 
                 stack_trace, context_data, occurrence_count, first_occurrence, last_occurrence)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                error_record.id,
                error_record.timestamp,
                error_record.severity.value,
                error_record.category.value,
                error_record.message,
                error_record.exception_type,
                error_record.stack_trace,
                context_json,
                error_record.occurrence_count,
                error_record.first_occurrence,
                error_record.last_occurrence
            )
            
            db_service.execute_update(insert_query, params)
            
        except Exception as e:
            self.logger.error(f"存储错误到数据库失败: {e}")
    
    async def _trigger_error_handlers(self, error_record: ErrorRecord):
        """触发错误处理器"""
        try:
            handlers = self.error_handlers.get(error_record.category, [])
            for handler in handlers:
                try:
                    await handler(error_record)
                except Exception as e:
                    self.logger.error(f"错误处理器执行失败: {e}")
        except Exception as e:
            self.logger.error(f"触发错误处理器失败: {e}")
    
    async def _handle_database_error(self, error_record: ErrorRecord):
        """处理数据库错误"""
        self.logger.warning(f"数据库错误: {error_record.message}")
        # 可以添加数据库重连逻辑等
    
    async def _handle_network_error(self, error_record: ErrorRecord):
        """处理网络错误"""
        self.logger.warning(f"网络错误: {error_record.message}")
        # 可以添加重试逻辑等
    
    async def _handle_system_error(self, error_record: ErrorRecord):
        """处理系统错误"""
        self.logger.error(f"系统错误: {error_record.message}")
        # 可以添加系统监控告警等
    
    async def get_error_by_id(self, error_id: str) -> Optional[ErrorRecord]:
        """根据ID获取错误记录"""
        try:
            error_key = f"error:{error_id}"
            error_data = await self.redis_client.get(error_key)
            
            if error_data:
                error_dict = json.loads(error_data)
                return self._dict_to_error_record(error_dict)
            
            return None
        except Exception as e:
            self.logger.error(f"获取错误记录失败: {e}")
            return None
    
    async def get_recent_errors(self, limit: int = 50) -> List[ErrorRecord]:
        """获取最近的错误记录"""
        try:
            error_ids = await self.redis_client.lrange("error_list", 0, limit - 1)
            errors = []
            
            for error_id in error_ids:
                error_record = await self.get_error_by_id(error_id)
                if error_record:
                    errors.append(error_record)
            
            return errors
        except Exception as e:
            self.logger.error(f"获取最近错误记录失败: {e}")
            return []
    
    async def get_errors_by_category(self, category: ErrorCategory, limit: int = 50) -> List[ErrorRecord]:
        """根据分类获取错误记录"""
        try:
            all_errors = await self.get_recent_errors(limit * 2)  # 获取更多以便过滤
            category_errors = [
                error for error in all_errors 
                if error.category == category
            ][:limit]
            
            return category_errors
        except Exception as e:
            self.logger.error(f"根据分类获取错误记录失败: {e}")
            return []
    
    async def resolve_error(self, error_id: str, resolution_notes: str, resolved_by: str) -> bool:
        """解决错误"""
        try:
            error_record = await self.get_error_by_id(error_id)
            if not error_record:
                return False
            
            error_record.resolved = True
            error_record.resolution_notes = resolution_notes
            error_record.resolved_at = datetime.now()
            error_record.resolved_by = resolved_by
            
            # 更新Redis中的记录
            await self._store_error_record(error_record)
            
            self.logger.info(f"错误 {error_id} 已解决")
            return True
        except Exception as e:
            self.logger.error(f"解决错误失败: {e}")
            return False
    
    async def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        try:
            recent_errors = await self.get_recent_errors(1000)
            
            # 按分类统计
            category_stats = {}
            for category in ErrorCategory:
                category_stats[category.value] = len([
                    e for e in recent_errors if e.category == category
                ])
            
            # 按严重程度统计
            severity_stats = {}
            for severity in ErrorSeverity:
                severity_stats[severity.value] = len([
                    e for e in recent_errors if e.severity == severity
                ])
            
            # 解决状态统计
            resolved_count = len([e for e in recent_errors if e.resolved])
            unresolved_count = len(recent_errors) - resolved_count
            
            # 最近24小时错误趋势
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            recent_24h_errors = [
                e for e in recent_errors 
                if e.timestamp >= last_24h
            ]
            
            return {
                'total_errors': len(recent_errors),
                'resolved_errors': resolved_count,
                'unresolved_errors': unresolved_count,
                'last_24h_errors': len(recent_24h_errors),
                'category_distribution': category_stats,
                'severity_distribution': severity_stats,
                'resolution_rate': resolved_count / len(recent_errors) if recent_errors else 0
            }
        except Exception as e:
            self.logger.error(f"获取错误统计失败: {e}")
            return {}
    
    def _dict_to_error_record(self, error_dict: Dict[str, Any]) -> ErrorRecord:
        """将字典转换为错误记录对象"""
        context_dict = error_dict.get('context', {})
        context = ErrorContext(**context_dict)
        
        return ErrorRecord(
            id=error_dict['id'],
            timestamp=datetime.fromisoformat(error_dict['timestamp']),
            severity=ErrorSeverity(error_dict['severity']),
            category=ErrorCategory(error_dict['category']),
            message=error_dict['message'],
            exception_type=error_dict['exception_type'],
            stack_trace=error_dict['stack_trace'],
            context=context,
            resolved=error_dict.get('resolved', False),
            resolution_notes=error_dict.get('resolution_notes'),
            resolved_at=datetime.fromisoformat(error_dict['resolved_at']) if error_dict.get('resolved_at') else None,
            resolved_by=error_dict.get('resolved_by'),
            occurrence_count=error_dict.get('occurrence_count', 1),
            first_occurrence=datetime.fromisoformat(error_dict['first_occurrence']),
            last_occurrence=datetime.fromisoformat(error_dict['last_occurrence'])
        )
    
    async def _cleanup_daemon(self):
        """清理守护进程"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时运行一次
                await self._cleanup_old_errors()
            except Exception as e:
                self.logger.error(f"清理守护进程错误: {e}")
    
    async def _cleanup_old_errors(self):
        """清理旧的错误记录"""
        try:
            # 清理超过7天的错误记录
            cutoff_time = datetime.now() - timedelta(days=7)
            
            error_ids = await self.redis_client.lrange("error_list", 0, -1)
            cleaned_count = 0
            
            for error_id in error_ids:
                error_record = await self.get_error_by_id(error_id)
                if error_record and error_record.timestamp < cutoff_time:
                    # 删除错误记录
                    await self.redis_client.delete(f"error:{error_id}")
                    await self.redis_client.lrem("error_list", 1, error_id)
                    cleaned_count += 1
            
            if cleaned_count > 0:
                self.logger.info(f"清理了 {cleaned_count} 条旧错误记录")
                
        except Exception as e:
            self.logger.error(f"清理旧错误记录失败: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查Redis连接
            redis_ok = await self.redis_client.ping()
            
            # 获取基本统计
            stats = await self.get_error_statistics()
            
            return {
                'status': 'healthy' if redis_ok else 'unhealthy',
                'redis_connection': redis_ok,
                'error_statistics': stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# 全局错误追踪服务实例
_error_tracking_service = None

async def get_error_tracking_service() -> ErrorTrackingService:
    """获取错误追踪服务实例"""
    global _error_tracking_service
    if _error_tracking_service is None:
        _error_tracking_service = ErrorTrackingService()
        await _error_tracking_service.initialize()
    return _error_tracking_service

# 便捷函数
async def capture_error(
    exception: Exception,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    context: Optional[ErrorContext] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """捕获错误的便捷函数"""
    service = await get_error_tracking_service()
    return await service.capture_error(exception, severity, category, context, additional_data)

async def capture_current_exception(
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    context: Optional[ErrorContext] = None,
    additional_data: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """捕获当前异常的便捷函数"""
    service = await get_error_tracking_service()
    return await service.capture_exception_from_context(severity, category, context, additional_data)