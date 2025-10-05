#!/usr/bin/env python3
"""
实时测试ASR、LLM、TTS模块功能
"""
import asyncio
import websockets
import json
import time
import requests
import base64
import wave
import struct

# 测试配置
SERVER_URL = "ws://localhost:8000/xiaozhi/v1/"
HTTP_SERVER_URL = "http://localhost:8000"

class ModuleTester:
    def __init__(self):
        self.test_results = {}
        
    def print_header(self, title):
        print(f"\n{'='*50}")
        print(f"  {title}")
        print(f"{'='*50}")
        
    def print_result(self, module, status, details=""):
        icon = "✅" if status else "❌"
        print(f"{icon} {module}: {'正常' if status else '异常'}")
        if details:
            print(f"   详情: {details}")
        self.test_results[module] = {"status": status, "details": details}
    
    def generate_test_audio(self, duration=2, sample_rate=16000):
        """生成测试音频数据 (PCM格式)"""
        import math
        frames = []
        for i in range(int(duration * sample_rate)):
            # 生成440Hz的正弦波
            value = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames.append(struct.pack('<h', value))
        return b''.join(frames)
    
    async def test_websocket_connection(self):
        """测试WebSocket连接"""
        self.print_header("测试WebSocket连接")
        try:
            async with websockets.connect(SERVER_URL) as websocket:
                # 发送初始化消息
                init_message = {
                    "type": "init",
                    "data": {
                        "session_id": "test_session_001",
                        "user_id": "test_user"
                    }
                }
                await websocket.send(json.dumps(init_message))
                
                # 等待响应
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == "init_success":
                    self.print_result("WebSocket连接", True, "连接成功")
                    return websocket
                else:
                    self.print_result("WebSocket连接", False, f"初始化失败: {response_data}")
                    return None
                    
        except Exception as e:
            self.print_result("WebSocket连接", False, f"连接失败: {e}")
            return None
    
    async def test_asr_module(self, websocket):
        """测试ASR模块"""
        self.print_header("测试ASR模块 (SenseVoiceStream)")
        try:
            # 生成测试音频
            test_audio = self.generate_test_audio(duration=3)
            
            # 发送音频数据
            audio_message = {
                "type": "audio",
                "data": {
                    "audio": base64.b64encode(test_audio).decode('utf-8'),
                    "format": "pcm",
                    "sample_rate": 16000
                }
            }
            
            await websocket.send(json.dumps(audio_message))
            print("   已发送测试音频数据...")
            
            # 等待ASR结果
            start_time = time.time()
            while time.time() - start_time < 10:  # 最多等待10秒
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") == "asr_result":
                        text = response_data.get("data", {}).get("text", "")
                        self.print_result("ASR模块", True, f"识别结果: '{text}'")
                        return True
                    elif response_data.get("type") == "error":
                        self.print_result("ASR模块", False, f"ASR错误: {response_data.get('data', {}).get('message', '')}")
                        return False
                        
                except asyncio.TimeoutError:
                    continue
                    
            self.print_result("ASR模块", False, "超时未收到ASR结果")
            return False
            
        except Exception as e:
            self.print_result("ASR模块", False, f"测试异常: {e}")
            return False
    
    async def test_llm_module(self, websocket):
        """测试LLM模块"""
        self.print_header("测试LLM模块")
        try:
            # 发送文本消息
            text_message = {
                "type": "text",
                "data": {
                    "text": "你好，请简单介绍一下你自己",
                    "session_id": "test_session_001"
                }
            }
            
            await websocket.send(json.dumps(text_message))
            print("   已发送测试文本...")
            
            # 等待LLM响应
            start_time = time.time()
            llm_response = ""
            
            while time.time() - start_time < 15:  # 最多等待15秒
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") == "llm_chunk":
                        chunk = response_data.get("data", {}).get("content", "")
                        llm_response += chunk
                        print(f"   LLM响应片段: {chunk}")
                        
                    elif response_data.get("type") == "llm_complete":
                        self.print_result("LLM模块", True, f"完整响应: '{llm_response[:100]}...'")
                        return True
                        
                    elif response_data.get("type") == "error":
                        self.print_result("LLM模块", False, f"LLM错误: {response_data.get('data', {}).get('message', '')}")
                        return False
                        
                except asyncio.TimeoutError:
                    continue
                    
            if llm_response:
                self.print_result("LLM模块", True, f"部分响应: '{llm_response[:100]}...'")
                return True
            else:
                self.print_result("LLM模块", False, "超时未收到LLM响应")
                return False
                
        except Exception as e:
            self.print_result("LLM模块", False, f"测试异常: {e}")
            return False
    
    async def test_tts_module(self, websocket):
        """测试TTS模块"""
        self.print_header("测试TTS模块 (EdgeTTS)")
        try:
            # 发送TTS请求
            tts_message = {
                "type": "tts",
                "data": {
                    "text": "这是一个TTS测试",
                    "session_id": "test_session_001"
                }
            }
            
            await websocket.send(json.dumps(tts_message))
            print("   已发送TTS请求...")
            
            # 等待TTS响应
            start_time = time.time()
            audio_chunks = 0
            
            while time.time() - start_time < 10:  # 最多等待10秒
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") == "tts_audio":
                        audio_chunks += 1
                        print(f"   收到TTS音频块: {audio_chunks}")
                        
                    elif response_data.get("type") == "tts_complete":
                        self.print_result("TTS模块", True, f"成功生成音频，共{audio_chunks}个音频块")
                        return True
                        
                    elif response_data.get("type") == "error":
                        self.print_result("TTS模块", False, f"TTS错误: {response_data.get('data', {}).get('message', '')}")
                        return False
                        
                except asyncio.TimeoutError:
                    continue
                    
            if audio_chunks > 0:
                self.print_result("TTS模块", True, f"部分成功，收到{audio_chunks}个音频块")
                return True
            else:
                self.print_result("TTS模块", False, "超时未收到TTS音频")
                return False
                
        except Exception as e:
            self.print_result("TTS模块", False, f"测试异常: {e}")
            return False
    
    def test_http_endpoints(self):
        """测试HTTP端点"""
        self.print_header("测试HTTP端点")
        try:
            # 测试健康检查
            response = requests.get(f"{HTTP_SERVER_URL}/health", timeout=5)
            if response.status_code == 200:
                self.print_result("HTTP健康检查", True, f"状态码: {response.status_code}")
            else:
                self.print_result("HTTP健康检查", False, f"状态码: {response.status_code}")
                
        except Exception as e:
            self.print_result("HTTP健康检查", False, f"请求失败: {e}")
    
    def check_current_config(self):
        """检查当前配置"""
        self.print_header("检查当前配置")
        try:
            # 检查本地配置文件
            with open('/root/xiaozhi-server/data/.config.yaml', 'r', encoding='utf-8') as f:
                import yaml
                config = yaml.safe_load(f)
                
            # 检查selected_module配置
            selected_modules = config.get('local_override', {}).get('selected_module', {})
            if not selected_modules:
                # 如果local_override中没有，检查根级别
                selected_modules = config.get('selected_module', {})
                
            print(f"   当前选中的模块:")
            for module, provider in selected_modules.items():
                print(f"     {module}: {provider}")
                
            # 检查是否从API读取配置
            read_from_api = config.get('read_config_from_api', False)
            print(f"   从API读取配置: {read_from_api}")
            
            if read_from_api:
                manager_api = config.get('manager-api', {})
                print(f"   Manager-API URL: {manager_api.get('url', 'N/A')}")
                
        except Exception as e:
            print(f"   配置检查失败: {e}")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("开始实测模块功能...")
        
        # 检查当前配置
        self.check_current_config()
        
        # 测试HTTP端点
        self.test_http_endpoints()
        
        # 测试WebSocket连接和各模块
        websocket = await self.test_websocket_connection()
        if websocket:
            try:
                await self.test_asr_module(websocket)
                await self.test_llm_module(websocket)
                await self.test_tts_module(websocket)
            finally:
                await websocket.close()
        
        # 输出测试总结
        self.print_header("测试总结")
        for module, result in self.test_results.items():
            icon = "✅" if result["status"] else "❌"
            print(f"{icon} {module}: {'正常' if result['status'] else '异常'}")
            if result["details"]:
                print(f"   {result['details']}")

if __name__ == "__main__":
    tester = ModuleTester()
    asyncio.run(tester.run_all_tests())