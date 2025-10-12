#!/usr/bin/env python3
"""
数据同步服务
在xiaozhi-server和ESP32服务器之间同步聊天记录
支持双向同步、冲突解决和实时通知
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from core.chat_history_service import ChatHistoryService
from core.enhanced_db_service import get_enhanced_db_service
from core.esp32_api_client import get_esp32_api_client

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyncDirection(Enum):
    """同步方向"""
    TO_ESP32 = "to_esp32"
    FROM_ESP32 = "from_esp32"
    BIDIRECTIONAL = "bidirectional"


class SyncStatus(Enum):
    """同步状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SyncRecord:
    """同步记录"""
    id: str
    direction: SyncDirection
    status: SyncStatus
    record_count: int
    start_time: datetime
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['direction'] = self.direction.value
        data['status'] = self.status.value
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data


class DataSyncService:
    """数据同步服务"""
    
    def __init__(self, sync_interval: int = 30):
        """
        初始化数据同步服务
        
        Args:
            sync_interval: 同步间隔（秒）
        """
        self.sync_interval = sync_interval
        self.chat_service = ChatHistoryService()
        self.db_service = get_enhanced_db_service()
        self.esp32_client = get_esp32_api_client()
        
        # 同步状态
        self.is_running = False
        self.last_sync_time = None
        self.sync_history: List[SyncRecord] = []
        
        # 性能统计
        self.stats = {
            "total_syncs": 0,
            "successful_syncs": 0,
            "failed_syncs": 0,
            "total_records_synced": 0,
            "average_sync_time": 0.0,
            "last_sync_time": None,
            "next_sync_time": None
        }
        
        logger.info(f"数据同步服务初始化完成，同步间隔: {sync_interval}秒")
    
    async def start_sync_daemon(self):
        """启动数据同步守护进程"""
        if self.is_running:
            logger.warning("数据同步守护进程已在运行")
            return
        
        self.is_running = True
        logger.info("🚀 启动数据同步守护进程")
        
        # 在后台启动守护进程
        asyncio.create_task(self._sync_daemon_loop())
    
    async def _sync_daemon_loop(self):
        """数据同步守护进程循环"""
        while self.is_running:
            try:
                await self.sync_recent_records()
                
                # 更新下次同步时间
                self.stats["next_sync_time"] = (
                    datetime.now() + timedelta(seconds=self.sync_interval)
                ).isoformat()
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"❌ 数据同步守护进程错误: {e}")
                self.stats["failed_syncs"] += 1
                await asyncio.sleep(60)  # 失败后等待1分钟重试
    
    async def stop_sync_daemon(self):
        """停止数据同步守护进程"""
        self.is_running = False
        logger.info("⏹️ 数据同步守护进程已停止")
    
    async def sync_recent_records(self) -> SyncRecord:
        """同步最近的聊天记录"""
        sync_record = SyncRecord(
            id=f"sync_{int(time.time())}",
            direction=SyncDirection.FROM_ESP32,
            status=SyncStatus.IN_PROGRESS,
            record_count=0,
            start_time=datetime.now()
        )
        
        try:
            logger.info("🔄 开始同步最近的聊天记录")
            
            # 获取最后同步时间
            cutoff_time = self.last_sync_time or (datetime.now() - timedelta(hours=1))
            
            # 从ESP32服务器获取新记录
            new_records = await self.get_records_from_esp32(cutoff_time)
            
            if new_records:
                # 批量插入本地数据库
                await self.batch_insert_records(new_records)
                sync_record.record_count = len(new_records)
                
                logger.info(f"✅ 成功同步 {len(new_records)} 条聊天记录")
            else:
                logger.info("📝 没有新的聊天记录需要同步")
            
            # 更新同步状态
            sync_record.status = SyncStatus.COMPLETED
            sync_record.end_time = datetime.now()
            
            # 更新统计信息
            self.stats["total_syncs"] += 1
            self.stats["successful_syncs"] += 1
            self.stats["total_records_synced"] += sync_record.record_count
            self.stats["last_sync_time"] = datetime.now().isoformat()
            
            # 计算平均同步时间
            sync_duration = (sync_record.end_time - sync_record.start_time).total_seconds()
            if self.stats["total_syncs"] > 0:
                self.stats["average_sync_time"] = (
                    (self.stats["average_sync_time"] * (self.stats["total_syncs"] - 1) + sync_duration) 
                    / self.stats["total_syncs"]
                )
            
            self.last_sync_time = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ 同步失败: {e}")
            sync_record.status = SyncStatus.FAILED
            sync_record.error_message = str(e)
            sync_record.end_time = datetime.now()
            
            self.stats["failed_syncs"] += 1
        
        # 记录同步历史
        self.sync_history.append(sync_record)
        
        # 只保留最近100条同步记录
        if len(self.sync_history) > 100:
            self.sync_history = self.sync_history[-100:]
        
        return sync_record
    
    async def get_records_from_esp32(self, since_time: datetime) -> List[Dict[str, Any]]:
        """从ESP32服务器获取聊天记录"""
        try:
            # 调用ESP32 API获取聊天记录
            records = await self.esp32_client.get_chat_history_since(since_time)
            return records
        except Exception as e:
            logger.error(f"❌ 从ESP32服务器获取记录失败: {e}")
            return []
    
    async def batch_insert_records(self, records: List[Dict[str, Any]]):
        """批量插入聊天记录到本地数据库"""
        try:
            for record in records:
                # 使用upsert避免重复插入
                await self.chat_service.upsert_chat_record(record)
            
            logger.info(f"✅ 批量插入 {len(records)} 条记录成功")
            
        except Exception as e:
            logger.error(f"❌ 批量插入记录失败: {e}")
            raise
    
    async def sync_to_esp32(self, device_id: str, since_time: Optional[datetime] = None) -> SyncRecord:
        """将本地记录同步到ESP32服务器"""
        sync_record = SyncRecord(
            id=f"sync_to_esp32_{int(time.time())}",
            direction=SyncDirection.TO_ESP32,
            status=SyncStatus.IN_PROGRESS,
            record_count=0,
            start_time=datetime.now()
        )
        
        try:
            logger.info(f"🔄 开始将设备 {device_id} 的记录同步到ESP32服务器")
            
            # 获取本地记录
            cutoff_time = since_time or (datetime.now() - timedelta(hours=1))
            local_records = await self.chat_service.get_chat_history(
                device_id=device_id,
                since_time=cutoff_time
            )
            
            if local_records:
                # 发送到ESP32服务器
                await self.send_records_to_esp32(local_records)
                sync_record.record_count = len(local_records)
                
                logger.info(f"✅ 成功同步 {len(local_records)} 条记录到ESP32服务器")
            else:
                logger.info("📝 没有新的记录需要同步到ESP32服务器")
            
            sync_record.status = SyncStatus.COMPLETED
            sync_record.end_time = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ 同步到ESP32服务器失败: {e}")
            sync_record.status = SyncStatus.FAILED
            sync_record.error_message = str(e)
            sync_record.end_time = datetime.now()
        
        self.sync_history.append(sync_record)
        return sync_record
    
    async def send_records_to_esp32(self, records: List[Dict[str, Any]]):
        """发送记录到ESP32服务器"""
        try:
            # 调用ESP32 API发送记录
            await self.esp32_client.sync_chat_records(records)
            logger.info(f"✅ 发送 {len(records)} 条记录到ESP32服务器成功")
            
        except Exception as e:
            logger.error(f"❌ 发送记录到ESP32服务器失败: {e}")
            raise
    
    async def force_full_sync(self, device_id: str) -> SyncRecord:
        """强制全量同步指定设备的记录"""
        logger.info(f"🔄 开始强制全量同步设备 {device_id}")
        
        # 获取所有本地记录
        all_records = self.chat_service.get_chat_history(device_id=device_id)
        logger.info(f"📊 找到 {len(all_records)} 条本地记录")
        
        # 同步到ESP32服务器
        return await self.sync_to_esp32(device_id, since_time=None)
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """获取同步统计信息"""
        return {
            "service_status": "running" if self.is_running else "stopped",
            "sync_interval": self.sync_interval,
            "stats": self.stats,
            "recent_syncs": [record.to_dict() for record in self.sync_history[-10:]]
        }
    
    def get_sync_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取同步历史"""
        return [record.to_dict() for record in self.sync_history[-limit:]]


# 全局数据同步服务实例
_data_sync_service = None


def get_data_sync_service() -> DataSyncService:
    """获取数据同步服务实例"""
    global _data_sync_service
    if _data_sync_service is None:
        _data_sync_service = DataSyncService()
    return _data_sync_service


async def start_data_sync_daemon():
    """启动数据同步守护进程"""
    sync_service = get_data_sync_service()
    await sync_service.start_sync_daemon()


async def stop_data_sync_daemon():
    """停止数据同步守护进程"""
    sync_service = get_data_sync_service()
    await sync_service.stop_sync_daemon()


if __name__ == "__main__":
    # 测试数据同步服务
    async def test_sync_service():
        sync_service = DataSyncService(sync_interval=10)
        
        # 测试单次同步
        result = await sync_service.sync_recent_records()
        print(f"同步结果: {result.to_dict()}")
        
        # 获取统计信息
        stats = sync_service.get_sync_stats()
        print(f"统计信息: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    asyncio.run(test_sync_service())