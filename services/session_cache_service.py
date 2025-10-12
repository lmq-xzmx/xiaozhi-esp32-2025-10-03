#!/usr/bin/env python3
"""
会话状态缓存服务
实现会话状态的Redis缓存、实时数据同步和会话管理
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from config.redis_config import get_redis_client
from core.chat_history_service import ChatHistoryService
from core.enhanced_db_service import get_enhanced_db_service

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """会话状态枚举"""
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"


@dataclass
class SessionState:
    """会话状态数据类"""
    session_id: str
    device_id: str
    status: SessionStatus
    created_at: datetime
    last_activity: datetime
    message_count: int = 0
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "device_id": self.device_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "message_count": self.message_count,
            "context": self.context,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionState':
        """从字典创建实例"""
        return cls(
            session_id=data["session_id"],
            device_id=data["device_id"],
            status=SessionStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activity=datetime.fromisoformat(data["last_activity"]),
            message_count=data.get("message_count", 0),
            context=data.get("context", {}),
            metadata=data.get("metadata", {})
        )


class SessionCacheService:
    """会话状态缓存服务"""
    
    def __init__(self, session_ttl: int = 3600, cleanup_interval: int = 300):
        """
        初始化会话缓存服务
        
        Args:
            session_ttl: 会话TTL（秒），默认1小时
            cleanup_interval: 清理间隔（秒），默认5分钟
        """
        self.session_ttl = session_ttl
        self.cleanup_interval = cleanup_interval
        self.redis_client = None
        self.chat_service = None
        self.db_service = None
        self.is_running = False
        
        # 缓存键前缀
        self.SESSION_PREFIX = "session:"
        self.DEVICE_SESSIONS_PREFIX = "device_sessions:"
        self.ACTIVE_SESSIONS_KEY = "active_sessions"
        
        # 统计信息
        self.stats = {
            "total_sessions": 0,
            "active_sessions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "sessions_created": 0,
            "sessions_expired": 0,
            "last_cleanup": None
        }
    
    async def initialize(self):
        """初始化服务"""
        try:
            self.redis_client = await get_redis_client()
            self.chat_service = ChatHistoryService()
            self.db_service = get_enhanced_db_service()
            
            logger.info("🚀 会话缓存服务初始化完成")
            return True
        except Exception as e:
            logger.error(f"❌ 会话缓存服务初始化失败: {e}")
            return False
    
    async def create_session(self, device_id: str, session_id: Optional[str] = None) -> SessionState:
        """创建新会话"""
        if not session_id:
            session_id = f"{device_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        now = datetime.now()
        session_state = SessionState(
            session_id=session_id,
            device_id=device_id,
            status=SessionStatus.ACTIVE,
            created_at=now,
            last_activity=now,
            message_count=0,
            context={
                "conversation_history": [],
                "user_preferences": {},
                "current_topic": None
            },
            metadata={
                "created_by": "session_cache_service",
                "version": "1.0"
            }
        )
        
        # 保存到Redis
        await self._save_session_to_cache(session_state)
        
        # 添加到设备会话列表
        await self._add_session_to_device(device_id, session_id)
        
        # 添加到活跃会话集合
        await self.redis_client.client.sadd(self.ACTIVE_SESSIONS_KEY, session_id)
        
        # 更新统计
        self.stats["sessions_created"] += 1
        self.stats["total_sessions"] += 1
        self.stats["active_sessions"] += 1
        
        logger.info(f"✅ 创建会话: {session_id} (设备: {device_id})")
        return session_state
    
    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """获取会话状态"""
        try:
            # 尝试从缓存获取
            cache_key = f"{self.SESSION_PREFIX}{session_id}"
            cached_data = await self.redis_client.client.get(cache_key)
            
            if cached_data:
                self.stats["cache_hits"] += 1
                session_data = json.loads(cached_data)
                return SessionState.from_dict(session_data)
            
            self.stats["cache_misses"] += 1
            logger.warning(f"⚠️ 会话不存在或已过期: {session_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ 获取会话失败 {session_id}: {e}")
            return None
    
    async def update_session(self, session_id: str, **updates) -> bool:
        """更新会话状态"""
        try:
            session_state = await self.get_session(session_id)
            if not session_state:
                logger.warning(f"⚠️ 尝试更新不存在的会话: {session_id}")
                return False
            
            # 更新字段
            for key, value in updates.items():
                if hasattr(session_state, key):
                    setattr(session_state, key, value)
            
            # 更新最后活动时间
            session_state.last_activity = datetime.now()
            
            # 保存到缓存
            await self._save_session_to_cache(session_state)
            
            logger.debug(f"🔄 更新会话: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 更新会话失败 {session_id}: {e}")
            return False
    
    async def update_session_context(self, session_id: str, context_updates: Dict[str, Any]) -> bool:
        """更新会话上下文"""
        try:
            session_state = await self.get_session(session_id)
            if not session_state:
                return False
            
            # 合并上下文
            session_state.context.update(context_updates)
            session_state.last_activity = datetime.now()
            
            # 保存到缓存
            await self._save_session_to_cache(session_state)
            
            logger.debug(f"🔄 更新会话上下文: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 更新会话上下文失败 {session_id}: {e}")
            return False
    
    async def increment_message_count(self, session_id: str) -> bool:
        """增加消息计数"""
        try:
            session_state = await self.get_session(session_id)
            if not session_state:
                return False
            
            session_state.message_count += 1
            session_state.last_activity = datetime.now()
            
            await self._save_session_to_cache(session_state)
            return True
            
        except Exception as e:
            logger.error(f"❌ 增加消息计数失败 {session_id}: {e}")
            return False
    
    async def get_device_sessions(self, device_id: str) -> List[SessionState]:
        """获取设备的所有会话"""
        try:
            cache_key = f"{self.DEVICE_SESSIONS_PREFIX}{device_id}"
            session_ids = await self.redis_client.client.smembers(cache_key)
            
            sessions = []
            for session_id in session_ids:
                session_state = await self.get_session(session_id)
                if session_state:
                    sessions.append(session_state)
            
            # 按最后活动时间排序
            sessions.sort(key=lambda x: x.last_activity, reverse=True)
            return sessions
            
        except Exception as e:
            logger.error(f"❌ 获取设备会话失败 {device_id}: {e}")
            return []
    
    async def get_active_sessions(self) -> List[SessionState]:
        """获取所有活跃会话"""
        try:
            session_ids = await self.redis_client.client.smembers(self.ACTIVE_SESSIONS_KEY)
            
            sessions = []
            for session_id in session_ids:
                session_state = await self.get_session(session_id)
                if session_state and session_state.status == SessionStatus.ACTIVE:
                    sessions.append(session_state)
            
            return sessions
            
        except Exception as e:
            logger.error(f"❌ 获取活跃会话失败: {e}")
            return []
    
    async def terminate_session(self, session_id: str) -> bool:
        """终止会话"""
        try:
            session_state = await self.get_session(session_id)
            if not session_state:
                return False
            
            # 更新状态
            session_state.status = SessionStatus.TERMINATED
            session_state.last_activity = datetime.now()
            
            # 保存到缓存（短期保留）
            await self._save_session_to_cache(session_state, ttl=300)  # 5分钟后清理
            
            # 从活跃会话中移除
            await self.redis_client.client.srem(self.ACTIVE_SESSIONS_KEY, session_id)
            
            # 更新统计
            self.stats["active_sessions"] = max(0, self.stats["active_sessions"] - 1)
            
            logger.info(f"🔚 终止会话: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 终止会话失败 {session_id}: {e}")
            return False
    
    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        try:
            logger.info("🧹 开始清理过期会话...")
            
            # 获取所有活跃会话
            session_ids = await self.redis_client.client.smembers(self.ACTIVE_SESSIONS_KEY)
            expired_count = 0
            
            for session_id in session_ids:
                session_state = await self.get_session(session_id)
                if not session_state:
                    # 会话已不存在，从活跃列表中移除
                    await self.redis_client.client.srem(self.ACTIVE_SESSIONS_KEY, session_id)
                    expired_count += 1
                    continue
                
                # 检查是否过期
                time_since_activity = datetime.now() - session_state.last_activity
                if time_since_activity.total_seconds() > self.session_ttl:
                    # 标记为过期
                    session_state.status = SessionStatus.EXPIRED
                    await self._save_session_to_cache(session_state, ttl=300)  # 短期保留
                    
                    # 从活跃列表中移除
                    await self.redis_client.client.srem(self.ACTIVE_SESSIONS_KEY, session_id)
                    expired_count += 1
            
            # 更新统计
            self.stats["sessions_expired"] += expired_count
            self.stats["active_sessions"] = max(0, self.stats["active_sessions"] - expired_count)
            self.stats["last_cleanup"] = datetime.now().isoformat()
            
            logger.info(f"✅ 清理完成，过期会话数: {expired_count}")
            
        except Exception as e:
            logger.error(f"❌ 清理过期会话失败: {e}")
    
    async def start_cleanup_daemon(self):
        """启动清理守护进程"""
        if self.is_running:
            logger.warning("清理守护进程已在运行")
            return
        
        self.is_running = True
        logger.info("🚀 启动会话清理守护进程")
        
        # 在后台启动清理循环
        asyncio.create_task(self._cleanup_daemon_loop())
    
    async def _cleanup_daemon_loop(self):
        """清理守护进程循环"""
        while self.is_running:
            try:
                await self.cleanup_expired_sessions()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"❌ 清理守护进程错误: {e}")
                await asyncio.sleep(60)  # 失败后等待1分钟重试
    
    async def stop_cleanup_daemon(self):
        """停止清理守护进程"""
        self.is_running = False
        logger.info("🛑 停止会话清理守护进程")
    
    async def _save_session_to_cache(self, session_state: SessionState, ttl: Optional[int] = None):
        """保存会话到缓存"""
        cache_key = f"{self.SESSION_PREFIX}{session_state.session_id}"
        session_data = json.dumps(session_state.to_dict(), ensure_ascii=False)
        
        if ttl is None:
            ttl = self.session_ttl
        
        await self.redis_client.client.setex(cache_key, ttl, session_data)
    
    async def _add_session_to_device(self, device_id: str, session_id: str):
        """添加会话到设备会话列表"""
        cache_key = f"{self.DEVICE_SESSIONS_PREFIX}{device_id}"
        await self.redis_client.client.sadd(cache_key, session_id)
        await self.redis_client.client.expire(cache_key, self.session_ttl * 2)  # 设备会话列表保留更长时间
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "service_status": "running" if self.is_running else "stopped",
            "session_ttl": self.session_ttl,
            "cleanup_interval": self.cleanup_interval,
            "stats": self.stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 检查Redis连接
            await self.redis_client.client.ping()
            
            # 获取活跃会话数
            active_count = await self.redis_client.client.scard(self.ACTIVE_SESSIONS_KEY)
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "active_sessions": active_count,
                "cleanup_daemon_running": self.is_running,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# 全局会话缓存服务实例
_session_cache_service = None


def get_session_cache_service() -> SessionCacheService:
    """获取全局会话缓存服务实例"""
    global _session_cache_service
    if _session_cache_service is None:
        _session_cache_service = SessionCacheService()
    return _session_cache_service


async def start_session_cache_daemon():
    """启动会话缓存守护进程"""
    service = get_session_cache_service()
    await service.initialize()
    await service.start_cleanup_daemon()


async def stop_session_cache_daemon():
    """停止会话缓存守护进程"""
    service = get_session_cache_service()
    await service.stop_cleanup_daemon()


if __name__ == "__main__":
    # 测试代码
    async def test_session_cache():
        service = SessionCacheService()
        await service.initialize()
        
        # 创建测试会话
        session = await service.create_session("test_device_001")
        print(f"创建会话: {session.session_id}")
        
        # 更新会话
        await service.update_session_context(session.session_id, {
            "current_topic": "天气查询",
            "user_preferences": {"language": "zh-CN"}
        })
        
        # 获取会话
        retrieved_session = await service.get_session(session.session_id)
        print(f"获取会话: {retrieved_session.to_dict()}")
        
        # 获取统计信息
        stats = service.get_stats()
        print(f"统计信息: {stats}")
    
    asyncio.run(test_session_cache())