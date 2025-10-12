#!/usr/bin/env python3
"""
独立的WebSocket测试服务器 - 专门测试聊天记录功能
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入聊天记录服务
try:
    from core.chat_history_service import ChatHistoryService
    logger.info("✅ 成功导入聊天记录服务")
except ImportError as e:
    logger.error(f"❌ 无法导入聊天记录服务: {e}")
    ChatHistoryService = None

class WebSocketTestManager:
    """WebSocket测试管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_info: Dict[str, Dict] = {}
        self.chat_service = ChatHistoryService() if ChatHistoryService else None
        
    async def connect(self, websocket: WebSocket, session_id: str, device_id: str, student_id: str):
        """建立WebSocket连接"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.session_info[session_id] = {
            "device_id": device_id,
            "student_id": student_id,
            "connected_at": datetime.now().isoformat(),
            "message_count": 0
        }
        logger.info(f"🔗 WebSocket连接建立: {session_id} (设备: {device_id}, 学生: {student_id})")
        
    def disconnect(self, session_id: str):
        """断开WebSocket连接"""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.session_info:
            del self.session_info[session_id]
        logger.info(f"🔌 WebSocket连接断开: {session_id}")
        
    async def send_message(self, session_id: str, message: dict):
        """发送消息到WebSocket"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text(json.dumps(message))
            
    async def record_user_input(self, session_id: str, text: str):
        """记录用户输入"""
        if not self.chat_service:
            logger.warning("⚠️ 聊天记录服务不可用")
            return
            
        session = self.session_info.get(session_id)
        if not session:
            logger.warning(f"⚠️ 会话信息不存在: {session_id}")
            return
            
        try:
            await self.chat_service.record_user_input(
                device_id=session["device_id"],
                student_id=session["student_id"],
                user_input=text,
                timestamp=datetime.now()
            )
            logger.info(f"📝 用户输入已记录: {text[:50]}...")
        except Exception as e:
            logger.error(f"❌ 记录用户输入失败: {e}")
            
    async def record_ai_response(self, session_id: str, response: str):
        """记录AI响应"""
        if not self.chat_service:
            logger.warning("⚠️ 聊天记录服务不可用")
            return
            
        session = self.session_info.get(session_id)
        if not session:
            logger.warning(f"⚠️ 会话信息不存在: {session_id}")
            return
            
        try:
            await self.chat_service.record_ai_response(
                device_id=session["device_id"],
                student_id=session["student_id"],
                ai_response=response,
                timestamp=datetime.now()
            )
            logger.info(f"🤖 AI响应已记录: {response[:50]}...")
        except Exception as e:
            logger.error(f"❌ 记录AI响应失败: {e}")

# 创建FastAPI应用
app = FastAPI(title="WebSocket Chat Test Server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建WebSocket管理器
manager = WebSocketTestManager()

@app.websocket("/ws/{device_id}/{student_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str, student_id: str, session_id: str):
    """WebSocket端点 - 支持聊天记录"""
    await manager.connect(websocket, session_id, device_id, student_id)
    
    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"📨 收到消息: {message}")
            
            # 处理不同类型的消息
            msg_type = message.get("type", "unknown")
            
            if msg_type == "user_input":
                # 用户输入
                text = message.get("text", "")
                await manager.record_user_input(session_id, text)
                
                # 模拟AI响应
                ai_response = f"收到您的消息: {text}"
                await manager.record_ai_response(session_id, ai_response)
                
                # 发送响应
                response = {
                    "type": "ai_response",
                    "text": ai_response,
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_message(session_id, response)
                
            elif msg_type == "ping":
                # 心跳检测
                pong = {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_message(session_id, pong)
                
            else:
                # 未知消息类型
                error_response = {
                    "type": "error",
                    "message": f"未知消息类型: {msg_type}",
                    "timestamp": datetime.now().isoformat()
                }
                await manager.send_message(session_id, error_response)
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"❌ WebSocket错误: {e}")
        manager.disconnect(session_id)

@app.get("/sessions")
async def get_active_sessions():
    """获取活跃会话列表"""
    return {
        "active_sessions": list(manager.session_info.keys()),
        "session_details": manager.session_info,
        "total_connections": len(manager.active_connections)
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    chat_service_status = "available" if manager.chat_service else "unavailable"
    return {
        "status": "healthy",
        "service": "websocket_test",
        "chat_service": chat_service_status,
        "active_connections": len(manager.active_connections)
    }

@app.post("/test/record_user_input")
async def test_record_user_input(device_id: str, student_id: str, text: str):
    """测试记录用户输入"""
    if not manager.chat_service:
        return {"success": False, "error": "聊天记录服务不可用"}
        
    try:
        await manager.chat_service.record_user_input(
            device_id=device_id,
            student_id=student_id,
            user_input=text,
            timestamp=datetime.now()
        )
        return {"success": True, "message": "用户输入记录成功"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/test/record_ai_response")
async def test_record_ai_response(device_id: str, student_id: str, response: str):
    """测试记录AI响应"""
    if not manager.chat_service:
        return {"success": False, "error": "聊天记录服务不可用"}
        
    try:
        await manager.chat_service.record_ai_response(
            device_id=device_id,
            student_id=student_id,
            ai_response=response,
            timestamp=datetime.now()
        )
        return {"success": True, "message": "AI响应记录成功"}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    logger.info("🎯 启动WebSocket聊天测试服务器 (端口: 8002)")
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=1)