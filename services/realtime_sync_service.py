#!/usr/bin/env python3
"""
实时数据同步服务
在Redis缓存和数据库之间实现实时数据同步
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from config.redis_config import get_redis_client
from core.chat_history_service import ChatHistoryService
from core.enhanced_db_service import get_enhanced_db_service

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyncOperation(Enum):
    """同步操作类型"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    BATCH_UPDATE = "batch_update"


class SyncDirection(Enum):
    """同步方向"""
    CACHE_TO_DB = "cache_to_db"
    DB_TO_CACHE = "db_to_cache"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class SyncTask:
    """同步任务"""
    task_id: str
    operation: SyncOperation
    direction: SyncDirection
    table_name: str
    data: Dict[str, Any]
    cache_key: str
    created_at: datetime
    priority: int = 1  # 1=高优先级, 2=中优先级, 3=低优先级
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "operation": self.operation.value,
            "direction": self.direction.value,
            "table_name": self.table_name,
            "data": self.data,
            "cache_key": self.cache_key,
            "created_at": self.created_at.isoformat(),
            "priority": self.priority,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SyncTask':
        """从字典创建实例"""
        return cls(
            task_id=data["task_id"],
            operation=SyncOperation(data["operation"]),
            direction=SyncDirection(data["direction"]),
            table_name=data["table_name"],
            data=data["data"],
            cache_key=data["cache_key"],
            created_at=datetime.fromisoformat(data["created_at"]),
            priority=data.get("priority", 1),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )


class RealtimeSyncService:
    """实时数据同步服务"""
    
    def __init__(self, sync_interval: int = 5, batch_size: int = 100):
        """
        初始化实时同步服务
        
        Args:
            sync_interval: 同步间隔（秒）
            batch_size: 批处理大小
        """
        self.sync_interval = sync_interval
        self.batch_size = batch_size
        self.redis_client = None
        self.chat_service = None
        self.db_service = None
        self.is_running = False
        
        # 队列键
        self.SYNC_QUEUE_KEY = "realtime_sync_queue"
        self.FAILED_QUEUE_KEY = "realtime_sync_failed"
        self.PROCESSING_KEY = "realtime_sync_processing"
        
        # 缓存键前缀
        self.CHAT_RECORD_PREFIX = "chat_record:"
        self.DEVICE_STATS_PREFIX = "device_stats:"
        self.SESSION_PREFIX = "session:"
        
        # 统计信息
        self.stats = {
            "total_tasks": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "cache_to_db_syncs": 0,
            "db_to_cache_syncs": 0,
            "last_sync_time": None,
            "queue_size": 0,
            "processing_time_avg": 0.0
        }
        
        # 同步处理器映射
        self.sync_handlers = {
            "chat_records": self._sync_chat_records,
            "device_stats": self._sync_device_stats,
            "sessions": self._sync_sessions
        }
    
    async def initialize(self):
        """初始化服务"""
        try:
            self.redis_client = await get_redis_client()
            self.chat_service = ChatHistoryService()
            self.db_service = get_enhanced_db_service()
            
            logger.info("🚀 实时数据同步服务初始化完成")
            return True
        except Exception as e:
            logger.error(f"❌ 实时数据同步服务初始化失败: {e}")
            return False
    
    async def add_sync_task(self, operation: SyncOperation, direction: SyncDirection, 
                           table_name: str, data: Dict[str, Any], 
                           cache_key: str, priority: int = 1) -> str:
        """添加同步任务"""
        task_id = f"{table_name}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        task = SyncTask(
            task_id=task_id,
            operation=operation,
            direction=direction,
            table_name=table_name,
            data=data,
            cache_key=cache_key,
            created_at=datetime.now(),
            priority=priority
        )
        
        # 添加到队列（按优先级排序）
        await self.redis_client.client.zadd(
            self.SYNC_QUEUE_KEY, 
            {json.dumps(task.to_dict()): priority}
        )
        
        self.stats["total_tasks"] += 1
        self.stats["queue_size"] += 1
        
        logger.debug(f"📝 添加同步任务: {task_id} ({operation.value} -> {direction.value})")
        return task_id
    
    async def sync_chat_record_to_cache(self, record_data: Dict[str, Any]) -> str:
        """同步聊天记录到缓存"""
        cache_key = f"{self.CHAT_RECORD_PREFIX}{record_data.get('id', '')}"
        return await self.add_sync_task(
            operation=SyncOperation.UPDATE,
            direction=SyncDirection.DB_TO_CACHE,
            table_name="chat_records",
            data=record_data,
            cache_key=cache_key,
            priority=1
        )
    
    async def sync_chat_record_to_db(self, record_data: Dict[str, Any]) -> str:
        """同步聊天记录到数据库"""
        cache_key = f"{self.CHAT_RECORD_PREFIX}{record_data.get('id', '')}"
        return await self.add_sync_task(
            operation=SyncOperation.CREATE,
            direction=SyncDirection.CACHE_TO_DB,
            table_name="chat_records",
            data=record_data,
            cache_key=cache_key,
            priority=1
        )
    
    async def sync_device_stats_to_cache(self, device_id: str, stats_data: Dict[str, Any]) -> str:
        """同步设备统计到缓存"""
        cache_key = f"{self.DEVICE_STATS_PREFIX}{device_id}"
        return await self.add_sync_task(
            operation=SyncOperation.UPDATE,
            direction=SyncDirection.DB_TO_CACHE,
            table_name="device_stats",
            data={"device_id": device_id, **stats_data},
            cache_key=cache_key,
            priority=2
        )
    
    async def sync_session_to_cache(self, session_data: Dict[str, Any]) -> str:
        """同步会话到缓存"""
        cache_key = f"{self.SESSION_PREFIX}{session_data.get('session_id', '')}"
        return await self.add_sync_task(
            operation=SyncOperation.UPDATE,
            direction=SyncDirection.DB_TO_CACHE,
            table_name="sessions",
            data=session_data,
            cache_key=cache_key,
            priority=2
        )
    
    async def start_sync_daemon(self):
        """启动同步守护进程"""
        if self.is_running:
            logger.warning("实时同步守护进程已在运行")
            return
        
        self.is_running = True
        logger.info("🚀 启动实时数据同步守护进程")
        
        # 在后台启动同步循环
        asyncio.create_task(self._sync_daemon_loop())
    
    async def _sync_daemon_loop(self):
        """同步守护进程循环"""
        while self.is_running:
            try:
                await self._process_sync_queue()
                await asyncio.sleep(self.sync_interval)
            except Exception as e:
                logger.error(f"❌ 同步守护进程错误: {e}")
                await asyncio.sleep(10)  # 失败后等待10秒重试
    
    async def stop_sync_daemon(self):
        """停止同步守护进程"""
        self.is_running = False
        logger.info("🛑 停止实时数据同步守护进程")
    
    async def _process_sync_queue(self):
        """处理同步队列"""
        try:
            # 获取队列大小
            queue_size = await self.redis_client.client.zcard(self.SYNC_QUEUE_KEY)
            self.stats["queue_size"] = queue_size
            
            if queue_size == 0:
                return
            
            # 批量获取任务（按优先级排序）
            tasks_data = await self.redis_client.client.zrange(
                self.SYNC_QUEUE_KEY, 0, self.batch_size - 1, withscores=True
            )
            
            if not tasks_data:
                return
            
            logger.info(f"🔄 处理 {len(tasks_data)} 个同步任务...")
            
            processed_tasks = []
            start_time = time.time()
            
            for task_json, priority in tasks_data:
                try:
                    task_data = json.loads(task_json)
                    task = SyncTask.from_dict(task_data)
                    
                    # 处理任务
                    success = await self._process_sync_task(task)
                    
                    if success:
                        self.stats["successful_syncs"] += 1
                        processed_tasks.append(task_json)
                    else:
                        # 增加重试次数
                        task.retry_count += 1
                        if task.retry_count >= task.max_retries:
                            # 移到失败队列
                            await self.redis_client.client.lpush(
                                self.FAILED_QUEUE_KEY, 
                                json.dumps(task.to_dict())
                            )
                            processed_tasks.append(task_json)
                            self.stats["failed_syncs"] += 1
                            logger.error(f"❌ 任务失败，已达最大重试次数: {task.task_id}")
                        else:
                            # 重新加入队列
                            await self.redis_client.client.zadd(
                                self.SYNC_QUEUE_KEY,
                                {json.dumps(task.to_dict()): task.priority + task.retry_count}
                            )
                            processed_tasks.append(task_json)
                            logger.warning(f"⚠️ 任务重试 ({task.retry_count}/{task.max_retries}): {task.task_id}")
                
                except Exception as e:
                    logger.error(f"❌ 处理任务失败: {e}")
                    processed_tasks.append(task_json)
            
            # 从队列中移除已处理的任务
            if processed_tasks:
                await self.redis_client.client.zrem(self.SYNC_QUEUE_KEY, *processed_tasks)
            
            # 更新统计
            processing_time = time.time() - start_time
            self.stats["processing_time_avg"] = (
                self.stats["processing_time_avg"] * 0.8 + processing_time * 0.2
            )
            self.stats["last_sync_time"] = datetime.now().isoformat()
            
            logger.info(f"✅ 处理完成，耗时: {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ 处理同步队列失败: {e}")
    
    async def _process_sync_task(self, task: SyncTask) -> bool:
        """处理单个同步任务"""
        try:
            handler = self.sync_handlers.get(task.table_name)
            if not handler:
                logger.error(f"❌ 未找到处理器: {task.table_name}")
                return False
            
            return await handler(task)
            
        except Exception as e:
            logger.error(f"❌ 处理同步任务失败 {task.task_id}: {e}")
            return False
    
    async def _sync_chat_records(self, task: SyncTask) -> bool:
        """同步聊天记录"""
        try:
            if task.direction == SyncDirection.CACHE_TO_DB:
                # 缓存到数据库
                if task.operation == SyncOperation.CREATE:
                    # 插入新记录
                    await self.chat_service.write_chat_record(
                        device_id=task.data.get("device_id"),
                        session_id=task.data.get("session_id"),
                        message_type=task.data.get("message_type"),
                        content=task.data.get("content"),
                        metadata=task.data.get("metadata", {})
                    )
                elif task.operation == SyncOperation.UPDATE:
                    # 更新记录（如果支持）
                    pass
                
                self.stats["cache_to_db_syncs"] += 1
                
            elif task.direction == SyncDirection.DB_TO_CACHE:
                # 数据库到缓存
                await self.redis_client.set_with_ttl(
                    task.cache_key,
                    task.data,
                    ttl=3600  # 1小时
                )
                
                self.stats["db_to_cache_syncs"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 同步聊天记录失败: {e}")
            return False
    
    async def _sync_device_stats(self, task: SyncTask) -> bool:
        """同步设备统计"""
        try:
            if task.direction == SyncDirection.DB_TO_CACHE:
                # 数据库到缓存
                await self.redis_client.set_with_ttl(
                    task.cache_key,
                    task.data,
                    ttl=1800  # 30分钟
                )
                
                self.stats["db_to_cache_syncs"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 同步设备统计失败: {e}")
            return False
    
    async def _sync_sessions(self, task: SyncTask) -> bool:
        """同步会话数据"""
        try:
            if task.direction == SyncDirection.DB_TO_CACHE:
                # 数据库到缓存
                await self.redis_client.set_with_ttl(
                    task.cache_key,
                    task.data,
                    ttl=3600  # 1小时
                )
                
                self.stats["db_to_cache_syncs"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 同步会话数据失败: {e}")
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        try:
            queue_size = await self.redis_client.client.zcard(self.SYNC_QUEUE_KEY)
            failed_size = await self.redis_client.client.llen(self.FAILED_QUEUE_KEY)
            processing_size = await self.redis_client.client.llen(self.PROCESSING_KEY)
            
            return {
                "queue_size": queue_size,
                "failed_size": failed_size,
                "processing_size": processing_size,
                "is_running": self.is_running
            }
        except Exception as e:
            logger.error(f"❌ 获取队列状态失败: {e}")
            return {"error": str(e)}
    
    async def get_failed_tasks(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取失败的任务"""
        try:
            failed_tasks_json = await self.redis_client.client.lrange(
                self.FAILED_QUEUE_KEY, 0, limit - 1
            )
            
            failed_tasks = []
            for task_json in failed_tasks_json:
                task_data = json.loads(task_json)
                failed_tasks.append(task_data)
            
            return failed_tasks
        except Exception as e:
            logger.error(f"❌ 获取失败任务失败: {e}")
            return []
    
    async def retry_failed_tasks(self) -> int:
        """重试失败的任务"""
        try:
            failed_tasks_json = await self.redis_client.client.lrange(
                self.FAILED_QUEUE_KEY, 0, -1
            )
            
            retry_count = 0
            for task_json in failed_tasks_json:
                task_data = json.loads(task_json)
                task = SyncTask.from_dict(task_data)
                
                # 重置重试次数
                task.retry_count = 0
                
                # 重新加入队列
                await self.redis_client.client.zadd(
                    self.SYNC_QUEUE_KEY,
                    {json.dumps(task.to_dict()): task.priority}
                )
                retry_count += 1
            
            # 清空失败队列
            await self.redis_client.client.delete(self.FAILED_QUEUE_KEY)
            
            logger.info(f"🔄 重试 {retry_count} 个失败任务")
            return retry_count
            
        except Exception as e:
            logger.error(f"❌ 重试失败任务失败: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "service_status": "running" if self.is_running else "stopped",
            "sync_interval": self.sync_interval,
            "batch_size": self.batch_size,
            "stats": self.stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查Redis连接
            await self.redis_client.client.ping()
            
            # 获取队列状态
            queue_status = await self.get_queue_status()
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "sync_daemon_running": self.is_running,
                "queue_status": queue_status,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# 全局实时同步服务实例
_realtime_sync_service = None


def get_realtime_sync_service() -> RealtimeSyncService:
    """获取全局实时同步服务实例"""
    global _realtime_sync_service
    if _realtime_sync_service is None:
        _realtime_sync_service = RealtimeSyncService()
    return _realtime_sync_service


async def start_realtime_sync_daemon():
    """启动实时同步守护进程"""
    service = get_realtime_sync_service()
    await service.initialize()
    await service.start_sync_daemon()


async def stop_realtime_sync_daemon():
    """停止实时同步守护进程"""
    service = get_realtime_sync_service()
    await service.stop_sync_daemon()


if __name__ == "__main__":
    # 测试代码
    async def test_realtime_sync():
        service = RealtimeSyncService()
        await service.initialize()
        
        # 添加测试任务
        task_id = await service.sync_chat_record_to_cache({
            "id": "test_record_001",
            "device_id": "test_device_001",
            "session_id": "test_session_001",
            "message_type": "user",
            "content": "测试消息",
            "created_at": datetime.now().isoformat()
        })
        
        print(f"添加任务: {task_id}")
        
        # 获取队列状态
        status = await service.get_queue_status()
        print(f"队列状态: {status}")
        
        # 获取统计信息
        stats = service.get_stats()
        print(f"统计信息: {stats}")
    
    asyncio.run(test_realtime_sync())