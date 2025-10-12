#!/usr/bin/env python3
"""
测试集成的聊天记录流程
验证ASR流式处理与聊天记录功能的集成
"""

import asyncio
import json
import logging
import time
import websockets
import requests
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedChatFlowTester:
    """集成聊天记录流程测试器"""
    
    def __init__(self):
        self.asr_base_url = "http://localhost:8001"
        self.asr_ws_url = "ws://localhost:8001"
        self.api_base_url = "http://localhost:8091"
        
    async def test_websocket_with_chat_recording(self):
        """测试WebSocket连接与聊天记录集成"""
        print("\n" + "="*60)
        print("测试WebSocket连接与聊天记录集成")
        print("="*60)
        
        # 测试参数
        device_id = "test_device_001"
        student_id = 1001  # 使用已存在的学生ID
        session_id = f"test_session_{int(time.time())}"
        
        try:
            # 连接WebSocket（使用带设备ID和学生ID的端点）
            ws_url = f"{self.asr_ws_url}/asr/stream/{device_id}/{student_id}/{session_id}"
            print(f"连接WebSocket: {ws_url}")
            
            async with websockets.connect(ws_url) as websocket:
                print("✅ WebSocket连接成功")
                
                # 发送设备信息
                device_info = {
                    "type": "device_info",
                    "device_id": device_id,
                    "student_id": student_id
                }
                await websocket.send(json.dumps(device_info))
                print(f"📤 发送设备信息: {device_info}")
                
                # 模拟发送音频数据（这里用空字节模拟）
                test_audio_chunks = [
                    b'\x00' * 640,  # 模拟音频块1
                    b'\x01' * 640,  # 模拟音频块2
                    b'\x02' * 640,  # 模拟音频块3
                ]
                
                for i, chunk in enumerate(test_audio_chunks):
                    await websocket.send(chunk)
                    print(f"📤 发送音频块 {i+1}")
                    
                    # 等待响应
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        result = json.loads(response)
                        print(f"📥 收到ASR结果: {result}")
                    except asyncio.TimeoutError:
                        print("⏰ 等待ASR结果超时（正常，因为是模拟音频）")
                    except Exception as e:
                        print(f"❌ 处理响应错误: {e}")
                
                # 测试记录AI响应
                await self.test_record_ai_response(session_id)
                
                # 获取会话信息
                await self.test_get_session_info(session_id)
                
        except Exception as e:
            print(f"❌ WebSocket测试失败: {e}")
    
    async def test_record_ai_response(self, session_id: str):
        """测试记录AI响应"""
        print(f"\n📝 测试记录AI响应 - 会话: {session_id}")
        
        try:
            url = f"{self.asr_base_url}/asr/record_ai_response/{session_id}"
            data = {
                "ai_response": "你好！我是小智助手，很高兴为您服务。"
            }
            
            response = requests.post(url, json=data)
            if response.status_code == 200:
                result = response.json()
                print(f"✅ AI响应记录成功: {result}")
            else:
                print(f"❌ AI响应记录失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ 记录AI响应错误: {e}")
    
    async def test_get_session_info(self, session_id: str):
        """测试获取会话信息"""
        print(f"\n📊 测试获取会话信息 - 会话: {session_id}")
        
        try:
            url = f"{self.asr_base_url}/asr/session/{session_id}"
            response = requests.get(url)
            
            if response.status_code == 200:
                session_info = response.json()
                print(f"✅ 会话信息: {session_info}")
            else:
                print(f"❌ 获取会话信息失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ 获取会话信息错误: {e}")
    
    async def test_get_active_sessions(self):
        """测试获取活跃会话列表"""
        print(f"\n📋 测试获取活跃会话列表")
        
        try:
            url = f"{self.asr_base_url}/asr/sessions"
            response = requests.get(url)
            
            if response.status_code == 200:
                sessions = response.json()
                print(f"✅ 活跃会话: {sessions}")
            else:
                print(f"❌ 获取活跃会话失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ 获取活跃会话错误: {e}")
    
    async def test_chat_service_health(self):
        """测试聊天记录服务健康检查"""
        print(f"\n🏥 测试聊天记录服务健康检查")
        
        try:
            url = f"{self.asr_base_url}/asr/chat_health"
            response = requests.get(url)
            
            if response.status_code == 200:
                health = response.json()
                print(f"✅ 聊天记录服务健康状态: {health}")
            else:
                print(f"❌ 聊天记录服务健康检查失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ 聊天记录服务健康检查错误: {e}")
    
    async def test_chat_records_api(self, device_id: str):
        """测试聊天记录API"""
        print(f"\n📚 测试聊天记录API - 设备: {device_id}")
        
        try:
            url = f"{self.api_base_url}/api/chat-records/{device_id}"
            response = requests.get(url)
            
            if response.status_code == 200:
                records = response.json()
                print(f"✅ 聊天记录: {json.dumps(records, indent=2, ensure_ascii=False)}")
            else:
                print(f"❌ 获取聊天记录失败: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ 获取聊天记录错误: {e}")
    
    async def run_complete_test(self):
        """运行完整的集成测试"""
        print("🚀 开始集成聊天记录流程测试")
        print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 测试聊天记录服务健康状态
        await self.test_chat_service_health()
        
        # 2. 测试WebSocket连接与聊天记录
        await self.test_websocket_with_chat_recording()
        
        # 3. 测试获取活跃会话
        await self.test_get_active_sessions()
        
        # 4. 测试聊天记录API
        await self.test_chat_records_api("test_device_001")
        
        print("\n" + "="*60)
        print("✅ 集成聊天记录流程测试完成")
        print("="*60)

async def main():
    """主函数"""
    tester = IntegratedChatFlowTester()
    await tester.run_complete_test()

if __name__ == "__main__":
    asyncio.run(main())