#!/usr/bin/env python3
"""
简化的WebSocket测试脚本 - 测试聊天记录功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入聊天记录服务
try:
    from core.chat_history_service import ChatHistoryService
    logger.info("✅ 成功导入聊天记录服务")
    chat_service = ChatHistoryService()
except ImportError as e:
    logger.error(f"❌ 无法导入聊天记录服务: {e}")
    chat_service = None

app = FastAPI(title="WebSocket Chat Test", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_connections = {}
session_info = {}

@app.websocket("/ws/{device_id}/{student_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, device_id: str, student_id: str, session_id: str):
    await websocket.accept()
    active_connections[session_id] = websocket
    session_info[session_id] = {
        "device_id": device_id,
        "student_id": student_id,
        "connected_at": datetime.now().isoformat()
    }
    logger.info(f"🔗 WebSocket连接建立: {session_id} (设备: {device_id}, 学生: {student_id})")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            logger.info(f"📨 收到消息: {message}")
            
            msg_type = message.get("type", "unknown")
            
            if msg_type == "user_input":
                text = message.get("text", "")
                
                # 记录用户输入
                if chat_service:
                    try:
                        await chat_service.record_user_input(
                            device_id=device_id,
                            student_id=student_id,
                            user_input=text,
                            timestamp=datetime.now()
                        )
                        logger.info(f"📝 用户输入已记录: {text[:50]}...")
                    except Exception as e:
                        logger.error(f"❌ 记录用户输入失败: {e}")
                
                # 模拟AI响应
                ai_response = f"收到您的消息: {text}"
                
                # 记录AI响应
                if chat_service:
                    try:
                        await chat_service.record_ai_response(
                            device_id=device_id,
                            student_id=student_id,
                            ai_response=ai_response,
                            timestamp=datetime.now()
                        )
                        logger.info(f"🤖 AI响应已记录: {ai_response[:50]}...")
                    except Exception as e:
                        logger.error(f"❌ 记录AI响应失败: {e}")
                
                # 发送响应
                response = {
                    "type": "ai_response",
                    "text": ai_response,
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(response))
                
            elif msg_type == "ping":
                pong = {
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }
                await websocket.send_text(json.dumps(pong))
                
    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]
        if session_id in session_info:
            del session_info[session_id]
        logger.info(f"🔌 WebSocket连接断开: {session_id}")
    except Exception as e:
        logger.error(f"❌ WebSocket错误: {e}")

@app.get("/health")
async def health_check():
    chat_service_status = "available" if chat_service else "unavailable"
    return {
        "status": "healthy",
        "service": "websocket_test",
        "chat_service": chat_service_status,
        "active_connections": len(active_connections)
    }

@app.get("/sessions")
async def get_sessions():
    return {
        "active_sessions": list(session_info.keys()),
        "session_details": session_info,
        "total_connections": len(active_connections)
    }

if __name__ == "__main__":
    logger.info("🎯 启动WebSocket聊天测试服务器 (端口: 8004)")
    uvicorn.run(app, host="0.0.0.0", port=8004, workers=1)