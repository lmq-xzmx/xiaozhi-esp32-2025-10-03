#!/usr/bin/env python3
"""
WebSocket服务器 - 处理设备连接和聊天记录
集成聊天记录功能，支持实时消息处理和记录
"""

import asyncio
import json
import logging
import uuid
import base64
from datetime import datetime
from typing import Dict, Optional, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 导入自定义模块
from core.chat_history_service import ChatHistoryService
from core.enhanced_db_service import get_enhanced_db_service
from core.esp32_api_client import get_esp32_api_client
from services.data_sync_service import get_data_sync_service
from services.session_cache_service import get_session_cache_service
from services.realtime_sync_service import get_realtime_sync_service
from services.monitoring_service import get_monitoring_service
from services.error_tracking_service import get_error_tracking_service, ErrorSeverity, ErrorCategory, ErrorContext
from services.enhanced_logging_service import get_logging_service, LogLevel, LogCategory, LogContext

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.device_sessions: Dict[str, str] = {}  # device_id -> session_id
        self.session_devices: Dict[str, str] = {}  # session_id -> device_id
        self.chat_service = ChatHistoryService()
        self.session_cache_service = get_session_cache_service()
        
    async def connect(self, websocket: WebSocket, device_id: str) -> str:
        """建立WebSocket连接"""
        # 注意：websocket.accept()应该在调用此方法之前已经被调用
        
        # 生成会话ID
        session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        # 存储连接信息
        self.active_connections[session_id] = websocket
        self.device_sessions[device_id] = session_id
        self.session_devices[session_id] = device_id
        
        # 在缓存中创建会话状态
        await self.session_cache_service.create_session(session_id, device_id)
        
        logger.info(f"设备 {device_id} 已连接，会话ID: {session_id}")
        return session_id
    
    async def disconnect(self, session_id: str):
        """断开WebSocket连接"""
        if session_id in self.active_connections:
            device_id = self.session_devices.get(session_id)
            
            # 清理连接信息
            del self.active_connections[session_id]
            if device_id:
                self.device_sessions.pop(device_id, None)
            self.session_devices.pop(session_id, None)
            
            # 终止缓存中的会话
            await self.session_cache_service.terminate_session(session_id)
            
            logger.info(f"会话 {session_id} (设备 {device_id}) 已断开")
    
    async def send_message(self, session_id: str, message: dict):
        """发送消息到指定会话"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message))
                return True
            except Exception as e:
                logger.error(f"发送消息失败 {session_id}: {e}")
                return False
        return False
    
    async def broadcast_to_device(self, device_id: str, message: dict):
        """向指定设备广播消息"""
        session_id = self.device_sessions.get(device_id)
        if session_id:
            return await self.send_message(session_id, message)
        return False

class XiaozhiWebSocketServer:
    """小智WebSocket服务器"""
    
    def __init__(self):
        self.connection_manager = WebSocketConnectionManager()
        self.chat_service = ChatHistoryService()
        self.esp32_client = get_esp32_api_client()
        self.data_sync_service = get_data_sync_service()
        self.session_cache_service = get_session_cache_service()
        self.realtime_sync_service = get_realtime_sync_service()
        # 异步服务将在 initialize() 方法中初始化
        self.monitoring_service = None
        self.error_tracking_service = None
        self.logging_service = None
        self._initialized = False
    
    async def initialize(self):
        """异步初始化服务"""
        if not self._initialized:
            # 同步服务
            self.monitoring_service = get_monitoring_service()
            # 异步服务
            self.error_tracking_service = await get_error_tracking_service()
            self.logging_service = await get_logging_service()
            self._initialized = True
        
    async def handle_hello_message(self, websocket: WebSocket, message: dict, device_id: str) -> str:
        """处理hello消息"""
        session_id = await self.connection_manager.connect(websocket, device_id)
        
        # 发送hello响应
        response = {
            "type": "hello",
            "transport": "websocket",
            "session_id": session_id,
            "audio_params": {
                "format": "opus",
                "sample_rate": 16000,
                "channels": 1,
                "frame_duration": 60
            }
        }
        
        await self.connection_manager.send_message(session_id, response)
        logger.info(f"已发送hello响应到设备 {device_id}")
        
        return session_id
    
    async def handle_listen_message(self, message: dict, session_id: str):
        """处理listen消息"""
        device_id = self.connection_manager.session_devices.get(session_id)
        state = message.get("state", "")
        
        logger.info(f"设备 {device_id} 监听状态: {state}")
        
        if state == "start":
            # 设备开始监听
            response = {
                "type": "listen",
                "state": "ready",
                "session_id": session_id
            }
            await self.connection_manager.send_message(session_id, response)
    
    async def handle_stt_result(self, message: dict, session_id: str):
        """处理STT结果并记录聊天"""
        device_id = self.connection_manager.session_devices.get(session_id)
        text = message.get("text", "")
        
        if text and device_id:
            # 记录用户输入的聊天记录
            try:
                # 从设备信息获取相关数据
                device_info = await self.get_device_info(device_id)
                
                chat_record = {
                    "mac_address": device_info.get("mac_address", device_id),
                    "agent_id": device_info.get("agent_id", "default_agent"),
                    "session_id": session_id,
                    "chat_type": "user_input",
                    "content": text,
                    "device_id": device_id,
                    "student_id": device_info.get("student_id", 1001)  # 默认用户ID
                }
                
                success = await self.chat_service.write_chat_record(chat_record)
                if success:
                    logger.info(f"用户输入记录成功: {text[:50]}...")
                else:
                    logger.error(f"用户输入记录失败: {text[:50]}...")
                    
            except Exception as e:
                logger.error(f"记录用户输入失败: {e}")
    
    async def handle_tts_message(self, message: dict, session_id: str):
        """处理TTS消息并记录聊天"""
        device_id = self.connection_manager.session_devices.get(session_id)
        state = message.get("state", "")
        text = message.get("text", "")
        
        if state == "sentence_start" and text and device_id:
            # 记录AI回复的聊天记录
            try:
                device_info = await self.get_device_info(device_id)
                
                chat_record = {
                    "mac_address": device_info.get("mac_address", device_id),
                    "agent_id": device_info.get("agent_id", "default_agent"),
                    "session_id": session_id,
                    "chat_type": "ai_response",
                    "content": text,
                    "device_id": device_id,
                    "student_id": device_info.get("student_id", 1001)
                }
                
                success = await self.chat_service.write_chat_record(chat_record)
                if success:
                    logger.info(f"AI回复记录成功: {text[:50]}...")
                else:
                    logger.error(f"AI回复记录失败: {text[:50]}...")
                    
            except Exception as e:
                logger.error(f"记录AI回复失败: {e}")
    
    async def handle_audio_message(self, message: dict, session_id: str):
        """处理音频消息 - 完整的VAD→ASR→LLM→TTS流程"""
        device_id = self.connection_manager.session_devices.get(session_id)
        if not device_id:
            logger.error(f"会话 {session_id} 没有关联的设备ID")
            return
        
        try:
            # 获取音频数据
            audio_data_b64 = message.get("audio_data", "")
            if not audio_data_b64:
                logger.error("音频消息缺少音频数据")
                return
            
            # 解码音频数据
            try:
                audio_data = base64.b64decode(audio_data_b64)
            except Exception as e:
                logger.error(f"音频数据解码失败: {e}")
                return
            
            logger.info(f"开始处理音频 - 设备: {device_id}, 数据长度: {len(audio_data)} bytes")
            
            # 发送处理开始通知
            await self.connection_manager.send_message(session_id, {
                "type": "audio_processing",
                "state": "started",
                "session_id": session_id
            })
            
            # 调用ESP32服务器进行完整音频处理
            result = await self.esp32_client.complete_audio_processing(
                audio_data=audio_data,
                device_id=device_id,
                session_id=session_id
            )
            
            if result["success"]:
                # 处理成功，保存聊天记录
                await self._save_audio_chat_records(result, device_id, session_id)
                
                # 发送处理完成通知和音频响应
                await self.connection_manager.send_message(session_id, {
                    "type": "audio_processing",
                    "state": "completed",
                    "session_id": session_id,
                    "user_text": result["user_text"],
                    "ai_response": result["ai_response"],
                    "audio_data": base64.b64encode(result["audio_data"]).decode('utf-8'),
                    "processing_steps": result["steps"]
                })
                
                logger.info(f"音频处理完成 - 设备: {device_id}")
            else:
                # 处理失败
                await self.connection_manager.send_message(session_id, {
                    "type": "audio_processing",
                    "state": "failed",
                    "session_id": session_id,
                    "error": result.get("error", "音频处理失败"),
                    "message": result.get("message", "未知错误")
                })
                
                logger.error(f"音频处理失败 - 设备: {device_id}: {result.get('message', '未知错误')}")
                
        except Exception as e:
            logger.error(f"音频消息处理异常 - 设备: {device_id}: {e}")
            
            # 发送错误通知
            await self.connection_manager.send_message(session_id, {
                "type": "audio_processing",
                "state": "error",
                "session_id": session_id,
                "error": str(e)
            })
    
    async def _save_audio_chat_records(self, result: dict, device_id: str, session_id: str):
        """保存音频处理产生的聊天记录"""
        try:
            device_info = await self.get_device_info(device_id)
            
            # 保存用户输入记录
            user_record = {
                "mac_address": device_info.get("mac_address", device_id),
                "agent_id": device_info.get("agent_id", "xiaozhi_agent"),
                "session_id": session_id,
                "chat_type": "user_input",
                "content": result["user_text"],
                "device_id": device_id,
                "student_id": device_info.get("student_id", 1001)
            }
            
            user_success = await self.chat_service.write_chat_record(user_record)
            if user_success:
                logger.info(f"用户音频输入记录成功: {result['user_text'][:50]}...")
            else:
                logger.error(f"用户音频输入记录失败: {result['user_text'][:50]}...")
            
            # 保存AI回复记录
            ai_record = {
                "mac_address": device_info.get("mac_address", device_id),
                "agent_id": device_info.get("agent_id", "xiaozhi_agent"),
                "session_id": session_id,
                "chat_type": "ai_response",
                "content": result["ai_response"],
                "device_id": device_id,
                "student_id": device_info.get("student_id", 1001)
            }
            
            ai_success = await self.chat_service.write_chat_record(ai_record)
            if ai_success:
                logger.info(f"AI音频回复记录成功: {result['ai_response'][:50]}...")
            else:
                logger.error(f"AI音频回复记录失败: {result['ai_response'][:50]}...")
                
        except Exception as e:
            logger.error(f"保存音频聊天记录失败: {e}")
    
    async def get_device_info(self, device_id: str) -> dict:
        """获取设备信息"""
        # 这里应该从数据库查询设备信息
        # 暂时返回默认信息
        return {
            "mac_address": device_id,
            "agent_id": "xiaozhi_agent",
            "student_id": 1001  # 默认学生ID
        }
    
    async def handle_websocket_connection(self, websocket: WebSocket, device_id: str):
        """处理WebSocket连接"""
        session_id = None
        
        try:
            # 接受WebSocket连接
            await websocket.accept()
            logger.info(f"WebSocket连接已接受 - 设备: {device_id}")
            
            # 等待hello消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "hello":
                session_id = await self.handle_hello_message(websocket, message, device_id)
            else:
                logger.error(f"期望hello消息，但收到: {message.get('type')}")
                await websocket.close()
                return
            
            # 处理后续消息
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    message_type = message.get("type", "")
                    
                    logger.info(f"收到消息类型: {message_type} from {device_id}")
                    
                    if message_type == "listen":
                        await self.handle_listen_message(message, session_id)
                    elif message_type == "stt":
                        await self.handle_stt_result(message, session_id)
                    elif message_type == "tts":
                        await self.handle_tts_message(message, session_id)
                    elif message_type == "audio":
                        await self.handle_audio_message(message, session_id)
                    elif message_type == "abort":
                        logger.info(f"设备 {device_id} 中止操作")
                    else:
                        logger.warning(f"未知消息类型: {message_type}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {e}")
                except Exception as e:
                    logger.error(f"处理消息错误: {e}")
                    break
                    
        except WebSocketDisconnect:
            logger.info(f"设备 {device_id} WebSocket连接断开")
        except Exception as e:
            logger.error(f"WebSocket连接错误: {e}")
        finally:
            if session_id:
                await self.connection_manager.disconnect(session_id)

# 创建FastAPI应用
app = FastAPI(title="Xiaozhi WebSocket Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局WebSocket服务器实例
websocket_server = XiaozhiWebSocketServer()

@app.websocket("/ws/{device_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str):
    """WebSocket端点"""
    await websocket_server.handle_websocket_connection(websocket, device_id)

@app.get("/health")
async def health_check():
    """基础健康检查"""
    return {
        "status": "healthy",
        "websocket_server": "running",
        "active_connections": len(websocket_server.connection_manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """详细健康检查"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "websocket": {
            "status": "running",
            "active_connections": len(websocket_server.connection_manager.active_connections),
            "device_sessions": len(websocket_server.connection_manager.device_sessions)
        }
    }
    
    # 检查增强数据库服务
    try:
        enhanced_db = get_enhanced_db_service()
        db_health = enhanced_db.health_check()
        health_status["services"]["enhanced_database"] = db_health
        if db_health["status"] != "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["enhanced_database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # 检查原始聊天服务
    try:
        chat_health = websocket_server.chat_service.health_check()
        health_status["services"]["chat_service"] = chat_health
    except Exception as e:
        health_status["services"]["chat_service"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # 检查ESP32 API客户端
    try:
        esp32_health = await websocket_server.esp32_client.health_check()
        health_status["services"]["esp32_client"] = esp32_health
        if esp32_health["status"] != "healthy":
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["esp32_client"] = {
            "status": "unhealthy",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # 检查Redis连接（如果可用）
    try:
        import redis
        redis_client = redis.Redis(host='xiaozhi-esp32-server-redis', port=6379, db=0, socket_timeout=5)
        redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/api/connections")
async def get_connections():
    """获取当前连接信息"""
    return {
        "active_connections": len(websocket_server.connection_manager.active_connections),
        "device_sessions": websocket_server.connection_manager.device_sessions,
        "session_devices": websocket_server.connection_manager.session_devices
    }

@app.get("/api/sync/status")
async def get_sync_status():
    """获取数据同步状态"""
    try:
        stats = websocket_server.data_sync_service.get_sync_stats()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取同步状态失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/sync/history")
async def get_sync_history(limit: int = 50):
    """获取数据同步历史"""
    try:
        history = websocket_server.data_sync_service.get_sync_history(limit)
        return {
            "status": "success",
            "data": history
        }
    except Exception as e:
        logger.error(f"获取同步历史失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/sync/force/{device_id}")
async def force_sync(device_id: str):
    """强制执行完整同步"""
    try:
        sync_record = await websocket_server.data_sync_service.force_full_sync(device_id)
        return {
            "status": "success",
            "data": sync_record.to_dict()
        }
    except Exception as e:
        logger.error(f"强制同步失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/sync/start")
async def start_sync_daemon():
    """启动数据同步守护进程"""
    try:
        await websocket_server.data_sync_service.start_sync_daemon()
        return {
            "status": "success",
            "message": "数据同步守护进程已启动"
        }
    except Exception as e:
        logger.error(f"启动同步守护进程失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/sync/stop")
async def stop_sync_daemon():
    """停止数据同步守护进程"""
    try:
        await websocket_server.data_sync_service.stop_sync_daemon()
        return {
            "status": "success",
            "message": "数据同步守护进程已停止"
        }
    except Exception as e:
        logger.error(f"停止同步守护进程失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# 会话缓存管理API
@app.get("/api/cache/sessions")
async def get_cached_sessions():
    """获取所有缓存的会话"""
    try:
        sessions = await websocket_server.session_cache_service.get_all_sessions()
        return {
            "status": "success",
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"获取缓存会话失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/cache/sessions/{session_id}")
async def get_session_state(session_id: str):
    """获取特定会话的状态"""
    try:
        session_state = await websocket_server.session_cache_service.get_session(session_id)
        if session_state:
            return {
                "status": "success",
                "session": session_state
            }
        else:
            return {
                "status": "error",
                "message": "会话不存在"
            }
    except Exception as e:
        logger.error(f"获取会话状态失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/cache/device/{device_id}/sessions")
async def get_device_sessions(device_id: str):
    """获取设备的所有会话"""
    try:
        sessions = await websocket_server.session_cache_service.get_device_sessions(device_id)
        return {
            "status": "success",
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"获取设备会话失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/cache/cleanup")
async def cleanup_expired_sessions():
    """清理过期的会话"""
    try:
        cleaned_count = await websocket_server.session_cache_service.cleanup_expired_sessions()
        return {
            "status": "success",
            "message": f"已清理 {cleaned_count} 个过期会话"
        }
    except Exception as e:
        logger.error(f"清理过期会话失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# 实时同步管理API
@app.get("/api/realtime-sync/status")
async def get_realtime_sync_status():
    """获取实时同步服务状态"""
    try:
        status = await websocket_server.realtime_sync_service.get_service_stats()
        return {
            "status": "success",
            "sync_status": status
        }
    except Exception as e:
        logger.error(f"获取实时同步状态失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/realtime-sync/queue")
async def get_sync_queue_status():
    """获取同步队列状态"""
    try:
        queue_status = await websocket_server.realtime_sync_service.get_queue_status()
        return {
            "status": "success",
            "queue": queue_status
        }
    except Exception as e:
        logger.error(f"获取同步队列状态失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/realtime-sync/start")
async def start_realtime_sync():
    """启动实时同步守护进程"""
    try:
        await websocket_server.realtime_sync_service.start_sync_daemon()
        return {
            "status": "success",
            "message": "实时同步守护进程已启动"
        }
    except Exception as e:
        logger.error(f"启动实时同步守护进程失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/realtime-sync/stop")
async def stop_realtime_sync():
    """停止实时同步守护进程"""
    try:
        await websocket_server.realtime_sync_service.stop_sync_daemon()
        return {
            "status": "success",
            "message": "实时同步守护进程已停止"
        }
    except Exception as e:
        logger.error(f"停止实时同步守护进程失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# 服务初始化检查装饰器
async def ensure_services_initialized():
    """确保异步服务已初始化"""
    if not websocket_server._initialized:
        await websocket_server.initialize()

# 监控系统API
@app.get("/api/monitoring/health")
async def get_monitoring_health():
    """获取监控系统健康状态"""
    try:
        await ensure_services_initialized()
        health = await websocket_server.monitoring_service.health_check()
        return {
            "status": "success",
            "data": health
        }
    except Exception as e:
        logger.error(f"获取监控健康状态失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/monitoring/metrics")
async def get_system_metrics():
    """获取系统指标"""
    try:
        await ensure_services_initialized()
        metrics = await websocket_server.monitoring_service.get_system_metrics()
        return {
            "status": "success",
            "data": metrics
        }
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/monitoring/alerts")
async def get_active_alerts():
    """获取活跃告警"""
    try:
        await ensure_services_initialized()
        alerts = await websocket_server.monitoring_service.get_active_alerts()
        return {
            "status": "success",
            "data": alerts
        }
    except Exception as e:
        logger.error(f"获取活跃告警失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/monitoring/start")
async def start_monitoring():
    """启动监控守护进程"""
    try:
        await ensure_services_initialized()
        await websocket_server.monitoring_service.start_monitoring_daemon()
        return {
            "status": "success",
            "message": "监控守护进程已启动"
        }
    except Exception as e:
        logger.error(f"启动监控守护进程失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/monitoring/stop")
async def stop_monitoring():
    """停止监控守护进程"""
    try:
        await ensure_services_initialized()
        await websocket_server.monitoring_service.stop_monitoring_daemon()
        return {
            "status": "success",
            "message": "监控守护进程已停止"
        }
    except Exception as e:
        logger.error(f"停止监控守护进程失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# 错误追踪API
@app.get("/api/errors/recent")
async def get_recent_errors(limit: int = 50):
    """获取最近的错误记录"""
    try:
        await ensure_services_initialized()
        errors = await websocket_server.error_tracking_service.get_recent_errors(limit)
        # 转换为字典格式
        error_dicts = []
        for error in errors:
            error_dict = {
                "id": error.id,
                "timestamp": error.timestamp.isoformat(),
                "severity": error.severity.value,
                "category": error.category.value,
                "message": error.message,
                "exception_type": error.exception_type,
                "resolved": error.resolved,
                "occurrence_count": error.occurrence_count
            }
            error_dicts.append(error_dict)
        
        return {
            "status": "success",
            "data": error_dicts
        }
    except Exception as e:
        logger.error(f"获取最近错误记录失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/errors/statistics")
async def get_error_statistics():
    """获取错误统计信息"""
    try:
        await ensure_services_initialized()
        stats = await websocket_server.error_tracking_service.get_error_statistics()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取错误统计信息失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/errors/{error_id}")
async def get_error_details(error_id: str):
    """获取错误详情"""
    try:
        await ensure_services_initialized()
        error = await websocket_server.error_tracking_service.get_error_by_id(error_id)
        if error:
            error_dict = {
                "id": error.id,
                "timestamp": error.timestamp.isoformat(),
                "severity": error.severity.value,
                "category": error.category.value,
                "message": error.message,
                "exception_type": error.exception_type,
                "stack_trace": error.stack_trace,
                "context": {
                    "user_id": error.context.user_id,
                    "session_id": error.context.session_id,
                    "device_id": error.context.device_id,
                    "request_id": error.context.request_id,
                    "endpoint": error.context.endpoint,
                    "method": error.context.method,
                    "user_agent": error.context.user_agent,
                    "ip_address": error.context.ip_address,
                    "additional_data": error.context.additional_data
                },
                "resolved": error.resolved,
                "resolution_notes": error.resolution_notes,
                "resolved_at": error.resolved_at.isoformat() if error.resolved_at else None,
                "resolved_by": error.resolved_by,
                "occurrence_count": error.occurrence_count,
                "first_occurrence": error.first_occurrence.isoformat() if error.first_occurrence else None,
                "last_occurrence": error.last_occurrence.isoformat() if error.last_occurrence else None
            }
            return {
                "status": "success",
                "data": error_dict
            }
        else:
            return {
                "status": "error",
                "message": "错误记录不存在"
            }
    except Exception as e:
        logger.error(f"获取错误详情失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/errors/{error_id}/resolve")
async def resolve_error(error_id: str, resolution_notes: str = "", resolved_by: str = "system"):
    """解决错误"""
    try:
        await ensure_services_initialized()
        success = await websocket_server.error_tracking_service.resolve_error(
            error_id, resolution_notes, resolved_by
        )
        if success:
            return {
                "status": "success",
                "message": "错误已标记为已解决"
            }
        else:
            return {
                "status": "error",
                "message": "错误记录不存在或已解决"
            }
    except Exception as e:
        logger.error(f"解决错误失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

# 日志系统API
@app.get("/api/logs/search")
async def search_logs(
    category: str = None,
    level: str = None,
    keyword: str = None,
    hours: int = 24,
    limit: int = 100
):
    """搜索日志"""
    try:
        await ensure_services_initialized()
        
        # 构建搜索条件
        filters = {}
        if category:
            filters['category'] = category
        if level:
            filters['level'] = level
        if keyword:
            filters['keyword'] = keyword
        
        # 计算时间范围
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 转换category和level为枚举类型
        from services.enhanced_logging_service import LogCategory, LogLevel
        
        category_enum = None
        if category:
            try:
                category_enum = LogCategory(category.lower())
            except ValueError:
                pass
        
        level_enum = None
        if level:
            try:
                level_enum = LogLevel(level.upper())
            except ValueError:
                pass
        
        logs = await websocket_server.logging_service.search_logs(
            category=category_enum,
            level=level_enum,
            start_time=start_time,
            end_time=end_time,
            keyword=keyword,
            limit=limit
        )
        
        return {
            "status": "success",
            "data": logs,
            "total": len(logs),
            "filters": filters,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            }
        }
    except Exception as e:
        logger.error(f"搜索日志失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/logs/analyze")
async def analyze_logs(category: str = None, hours: int = 24):
    """分析日志"""
    try:
        await ensure_services_initialized()
        
        # 计算时间范围
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 转换category为枚举类型
        from services.enhanced_logging_service import LogCategory
        
        category_enum = None
        if category:
            try:
                category_enum = LogCategory(category.lower())
            except ValueError:
                pass
        
        analysis = await websocket_server.logging_service.analyze_logs(
            category=category_enum,
            hours=hours
        )
        
        return {
            "status": "success",
            "data": analysis,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            }
        }
    except Exception as e:
        logger.error(f"分析日志失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/logs/statistics")
async def get_log_statistics():
    """获取日志统计信息"""
    try:
        await ensure_services_initialized()
        stats = await websocket_server.logging_service.get_log_statistics()
        return {
            "status": "success",
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取日志统计信息失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/logs/export")
async def export_logs(
    category: str = None,
    hours: int = 24,
    format: str = "json"
):
    """导出日志"""
    try:
        await ensure_services_initialized()
        
        # 计算时间范围
        from datetime import datetime, timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # 转换category为枚举类型
        from services.enhanced_logging_service import LogCategory
        
        category_enum = None
        if category:
            try:
                category_enum = LogCategory(category.lower())
            except ValueError:
                pass
        
        export_data = await websocket_server.logging_service.export_logs(
            category=category_enum,
            start_time=start_time,
            end_time=end_time,
            format=format
        )
        
        return {
            "status": "success",
            "data": export_data,
            "format": format,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": hours
            }
        }
    except Exception as e:
        logger.error(f"导出日志失败: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    print("=" * 60)
    print("小智WebSocket服务器启动中...")
    print("=" * 60)
    print("WebSocket端点: ws://localhost:8001/ws/{device_id}")
    print("健康检查: http://localhost:8001/health")
    print("连接信息: http://localhost:8001/api/connections")
    print("")
    print("数据同步API:")
    print("  状态: http://localhost:8001/api/sync/status")
    print("  历史: http://localhost:8001/api/sync/history")
    print("  强制同步: POST http://localhost:8001/api/sync/force/{device_id}")
    print("  启动守护进程: POST http://localhost:8001/api/sync/start")
    print("  停止守护进程: POST http://localhost:8001/api/sync/stop")
    print("")
    print("会话缓存API:")
    print("  所有会话: http://localhost:8001/api/cache/sessions")
    print("  会话状态: http://localhost:8001/api/cache/sessions/{session_id}")
    print("  设备会话: http://localhost:8001/api/cache/device/{device_id}/sessions")
    print("  清理过期: POST http://localhost:8001/api/cache/cleanup")
    print("")
    print("实时同步API:")
    print("  同步状态: http://localhost:8001/api/realtime-sync/status")
    print("  队列状态: http://localhost:8001/api/realtime-sync/queue")
    print("  启动同步: POST http://localhost:8001/api/realtime-sync/start")
    print("  停止同步: POST http://localhost:8001/api/realtime-sync/stop")
    print("")
    print("监控系统API:")
    print("  监控健康: http://localhost:8001/api/monitoring/health")
    print("  系统指标: http://localhost:8001/api/monitoring/metrics")
    print("  活跃告警: http://localhost:8001/api/monitoring/alerts")
    print("  启动监控: POST http://localhost:8001/api/monitoring/start")
    print("  停止监控: POST http://localhost:8001/api/monitoring/stop")
    print("")
    print("错误追踪API:")
    print("  最近错误: http://localhost:8001/api/errors/recent")
    print("  错误统计: http://localhost:8001/api/errors/statistics")
    print("  错误详情: http://localhost:8001/api/errors/{error_id}")
    print("  解决错误: POST http://localhost:8001/api/errors/{error_id}/resolve")
    print("")
    print("日志系统API:")
    print("  搜索日志: http://localhost:8001/api/logs/search")
    print("  分析日志: http://localhost:8001/api/logs/analyze")
    print("  日志统计: http://localhost:8001/api/logs/statistics")
    print("  导出日志: POST http://localhost:8001/api/logs/export")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)