#!/usr/bin/env python3
"""
WebSocket客户端测试脚本
测试音频处理和聊天记录保存功能
"""

import asyncio
import websockets
import json
import base64
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketTestClient:
    def __init__(self, uri: str, device_id: str):
        self.uri = uri
        self.device_id = device_id
        self.websocket = None
        
    async def connect(self):
        """连接到WebSocket服务器"""
        try:
            self.websocket = await websockets.connect(f"{self.uri}/ws/{self.device_id}")
            logger.info(f"已连接到WebSocket服务器: {self.uri}")
            return True
        except Exception as e:
            logger.error(f"连接失败: {e}")
            return False
    
    async def send_hello(self):
        """发送hello消息"""
        hello_message = {
            "type": "hello",
            "device_id": self.device_id,
            "timestamp": "2025-01-12T15:00:00Z"
        }
        
        await self.websocket.send(json.dumps(hello_message))
        logger.info("已发送hello消息")
        
        # 等待响应
        response = await self.websocket.recv()
        response_data = json.loads(response)
        logger.info(f"收到hello响应: {response_data}")
        return response_data.get("session_id")
    
    async def send_audio_message(self, session_id: str):
        """发送模拟音频消息"""
        # 创建模拟音频数据（实际应用中这会是真实的音频数据）
        fake_audio_data = b"fake_audio_data_for_testing_purposes_" + b"0" * 1000
        audio_b64 = base64.b64encode(fake_audio_data).decode('utf-8')
        
        audio_message = {
            "type": "audio",
            "session_id": session_id,
            "audio_data": audio_b64,
            "format": "opus",
            "timestamp": "2025-01-12T15:00:01Z"
        }
        
        await self.websocket.send(json.dumps(audio_message))
        logger.info("已发送音频消息")
        
        # 监听响应
        while True:
            try:
                response = await asyncio.wait_for(self.websocket.recv(), timeout=30.0)
                response_data = json.loads(response)
                logger.info(f"收到响应: {response_data.get('type')} - {response_data.get('state', '')}")
                
                if response_data.get("type") == "audio_processing":
                    state = response_data.get("state")
                    if state == "completed":
                        logger.info(f"音频处理完成!")
                        logger.info(f"用户文本: {response_data.get('user_text', 'N/A')}")
                        logger.info(f"AI回复: {response_data.get('ai_response', 'N/A')}")
                        break
                    elif state == "failed" or state == "error":
                        logger.error(f"音频处理失败: {response_data.get('error', 'Unknown error')}")
                        break
                        
            except asyncio.TimeoutError:
                logger.error("等待响应超时")
                break
            except Exception as e:
                logger.error(f"接收响应异常: {e}")
                break
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            logger.info("已断开WebSocket连接")

async def test_websocket_audio_processing():
    """测试WebSocket音频处理功能"""
    client = WebSocketTestClient("ws://localhost:8001", "test_device_001")
    
    try:
        # 连接
        if not await client.connect():
            return
        
        # 发送hello消息
        session_id = await client.send_hello()
        if not session_id:
            logger.error("未获取到session_id")
            return
        
        logger.info(f"获取到session_id: {session_id}")
        
        # 发送音频消息
        await client.send_audio_message(session_id)
        
    except Exception as e:
        logger.error(f"测试异常: {e}")
    finally:
        await client.disconnect()

if __name__ == "__main__":
    print("=" * 60)
    print("WebSocket音频处理测试")
    print("=" * 60)
    
    asyncio.run(test_websocket_audio_processing())