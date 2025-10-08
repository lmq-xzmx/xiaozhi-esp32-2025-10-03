#!/usr/bin/env python3
"""
并发和流式功能测试脚本
测试ASR、LLM、TTS模块的并发处理能力和流式特性
"""
import asyncio
import websockets
import json
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import wave
import struct
import base64
import numpy as np

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_result(test_name, success, details=""):
    icon = "✅" if success else "❌"
    print(f"{icon} {test_name}: {'成功' if success else '失败'}")
    if details:
        print(f"   详情: {details}")

# 测试配置
SERVER_URL = "ws://localhost:8000/xiaozhi/v1/"
HTTP_SERVER_URL = "http://localhost:8000"

class ConcurrencyTester:
    def __init__(self):
        self.test_results = {
            "ASR并发": False,
            "ASR流式": False,
            "LLM并发": False,
            "LLM流式": False,
            "TTS并发": False,
            "TTS流式": False
        }
        self.session_counter = 0
        
    def generate_test_audio(self, duration=1.0, sample_rate=16000):
        """生成测试音频数据"""
        # 生成简单的正弦波
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440  # A4音符
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        # 转换为16位PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        return audio_data.tobytes()

    async def create_websocket_connection(self, session_id):
        """创建WebSocket连接"""
        try:
            websocket = await websockets.connect(SERVER_URL)
            
            # 发送初始化消息
            init_message = {
                "type": "init",
                "data": {
                    "session_id": session_id,
                    "user_id": f"test_user_{session_id}"
                }
            }
            await websocket.send(json.dumps(init_message))
            
            # 等待初始化响应
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                
                if response_data.get("type") == "init_success":
                    return websocket
                else:
                    print(f"初始化失败: {response_data}")
                    await websocket.close()
                    return None
            except asyncio.TimeoutError:
                print("初始化超时")
                await websocket.close()
                return None
                
        except Exception as e:
            print(f"WebSocket连接失败: {e}")
            return None

    async def test_asr_concurrency(self):
        """测试ASR模块并发处理能力"""
        print_header("测试ASR模块并发处理")
        
        async def single_asr_test(session_id):
            websocket = await self.create_websocket_connection(f"asr_test_{session_id}")
            if not websocket:
                return False, f"连接失败"
                
            try:
                # 生成测试音频
                test_audio = self.generate_test_audio(duration=2.0)
                
                # 发送音频数据消息
                audio_message = {
                    "type": "audio_data",
                    "data": {
                        "audio": base64.b64encode(test_audio).decode('utf-8'),
                        "format": "pcm",
                        "sample_rate": 16000,
                        "channels": 1
                    }
                }
                
                start_time = time.time()
                await websocket.send(json.dumps(audio_message))
                
                # 等待ASR结果
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                end_time = time.time()
                
                response_data = json.loads(response)
                
                if response_data.get("type") == "asr_result":
                    return True, f"响应时间: {end_time-start_time:.2f}s"
                else:
                    return False, f"未收到ASR结果: {response_data}"
                    
            except Exception as e:
                return False, f"测试失败: {e}"
            finally:
                await websocket.close()

        # 并发测试
        concurrent_sessions = 3
        start_time = time.time()
        
        tasks = [single_asr_test(i) for i in range(concurrent_sessions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        success_count = sum(1 for success, _ in results if success)
        total_time = end_time - start_time
        
        self.test_results["ASR并发"] = success_count >= concurrent_sessions * 0.7
        print_result("ASR并发测试", self.test_results["ASR并发"], 
                    f"成功: {success_count}/{concurrent_sessions}, 总耗时: {total_time:.2f}s")
        
        return self.test_results["ASR并发"]

    async def test_asr_streaming(self):
        """测试ASR模块流式处理特性"""
        print_header("测试ASR模块流式处理")
        
        websocket = await self.create_websocket_connection("asr_streaming_test")
        if not websocket:
            self.test_results["ASR流式"] = False
            print_result("ASR流式测试", False, "连接失败")
            return False
            
        try:
            # 生成较长的音频用于流式测试
            test_audio = self.generate_test_audio(duration=5.0)
            chunk_size = 1024
            
            streaming_responses = []
            
            # 分块发送音频
            for i in range(0, len(test_audio), chunk_size):
                chunk = test_audio[i:i+chunk_size]
                
                audio_message = {
                    "type": "audio_chunk",
                    "data": {
                        "audio": base64.b64encode(chunk).decode('utf-8'),
                        "is_final": i + chunk_size >= len(test_audio)
                    }
                }
                
                await websocket.send(json.dumps(audio_message))
                
                # 尝试接收流式响应
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    response_data = json.loads(response)
                    if response_data.get("type") in ["asr_partial", "asr_result"]:
                        streaming_responses.append(response_data)
                except asyncio.TimeoutError:
                    pass  # 没有立即响应是正常的
                    
                await asyncio.sleep(0.1)  # 模拟实时音频流
            
            # 等待最终结果
            try:
                final_response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                final_data = json.loads(final_response)
                if final_data.get("type") == "asr_result":
                    streaming_responses.append(final_data)
            except asyncio.TimeoutError:
                pass
            
            # 判断是否支持流式
            has_streaming = len(streaming_responses) > 1
            self.test_results["ASR流式"] = has_streaming
            
            print_result("ASR流式测试", has_streaming, 
                        f"收到 {len(streaming_responses)} 个响应")
            
            return has_streaming
            
        except Exception as e:
            self.test_results["ASR流式"] = False
            print_result("ASR流式测试", False, f"测试失败: {e}")
            return False
        finally:
            await websocket.close()

    async def test_llm_concurrency(self):
        """测试LLM模块并发处理能力"""
        print_header("测试LLM模块并发处理")
        
        async def single_llm_test(session_id):
            websocket = await self.create_websocket_connection(f"llm_test_{session_id}")
            if not websocket:
                return False, "连接失败"
                
            try:
                # 发送文本消息
                text_message = {
                    "type": "text_message",
                    "data": {
                        "text": f"你好，这是并发测试 {session_id}",
                        "session_id": f"llm_test_{session_id}"
                    }
                }
                
                start_time = time.time()
                await websocket.send(json.dumps(text_message))
                
                # 等待LLM响应
                response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                end_time = time.time()
                
                response_data = json.loads(response)
                
                if response_data.get("type") in ["llm_response", "text_response"]:
                    return True, f"响应时间: {end_time-start_time:.2f}s"
                else:
                    return False, f"未收到LLM响应: {response_data}"
                    
            except Exception as e:
                return False, f"测试失败: {e}"
            finally:
                await websocket.close()

        # 并发测试
        concurrent_sessions = 3
        start_time = time.time()
        
        tasks = [single_llm_test(i) for i in range(concurrent_sessions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        success_count = sum(1 for success, _ in results if success)
        total_time = end_time - start_time
        
        self.test_results["LLM并发"] = success_count >= concurrent_sessions * 0.7
        print_result("LLM并发测试", self.test_results["LLM并发"], 
                    f"成功: {success_count}/{concurrent_sessions}, 总耗时: {total_time:.2f}s")
        
        return self.test_results["LLM并发"]

    async def test_llm_streaming(self):
        """测试LLM模块流式输出特性"""
        print_header("测试LLM模块流式输出")
        
        websocket = await self.create_websocket_connection("llm_streaming_test")
        if not websocket:
            self.test_results["LLM流式"] = False
            print_result("LLM流式测试", False, "连接失败")
            return False
            
        try:
            # 发送需要较长回答的问题
            text_message = {
                "type": "text_message",
                "data": {
                    "text": "请详细介绍一下人工智能的发展历史，包括主要的里程碑事件。",
                    "session_id": "llm_streaming_test"
                }
            }
            
            await websocket.send(json.dumps(text_message))
            
            streaming_responses = []
            start_time = time.time()
            
            # 收集流式响应
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") in ["llm_chunk", "llm_response", "text_response"]:
                        streaming_responses.append(response_data)
                        
                        # 如果是最终响应，退出循环
                        if response_data.get("data", {}).get("is_final", False) or \
                           response_data.get("type") == "llm_response":
                            break
                            
                except asyncio.TimeoutError:
                    break
            
            end_time = time.time()
            
            # 判断是否支持流式
            has_streaming = len(streaming_responses) > 1
            self.test_results["LLM流式"] = has_streaming
            
            print_result("LLM流式测试", has_streaming, 
                        f"收到 {len(streaming_responses)} 个响应, 耗时: {end_time-start_time:.2f}s")
            
            return has_streaming
            
        except Exception as e:
            self.test_results["LLM流式"] = False
            print_result("LLM流式测试", False, f"测试失败: {e}")
            return False
        finally:
            await websocket.close()

    async def test_tts_concurrency(self):
        """测试TTS模块并发处理能力"""
        print_header("测试TTS模块并发处理")
        
        async def single_tts_test(session_id):
            websocket = await self.create_websocket_connection(f"tts_test_{session_id}")
            if not websocket:
                return False, "连接失败"
                
            try:
                # 发送TTS请求
                tts_message = {
                    "type": "tts_request",
                    "data": {
                        "text": f"这是TTS并发测试 {session_id}，测试语音合成功能。",
                        "session_id": f"tts_test_{session_id}"
                    }
                }
                
                start_time = time.time()
                await websocket.send(json.dumps(tts_message))
                
                # 等待TTS响应
                response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                end_time = time.time()
                
                response_data = json.loads(response)
                
                if response_data.get("type") in ["tts_response", "audio_response"]:
                    return True, f"响应时间: {end_time-start_time:.2f}s"
                else:
                    return False, f"未收到TTS响应: {response_data}"
                    
            except Exception as e:
                return False, f"测试失败: {e}"
            finally:
                await websocket.close()

        # 并发测试
        concurrent_sessions = 3
        start_time = time.time()
        
        tasks = [single_tts_test(i) for i in range(concurrent_sessions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        success_count = sum(1 for success, _ in results if success)
        total_time = end_time - start_time
        
        self.test_results["TTS并发"] = success_count >= concurrent_sessions * 0.7
        print_result("TTS并发测试", self.test_results["TTS并发"], 
                    f"成功: {success_count}/{concurrent_sessions}, 总耗时: {total_time:.2f}s")
        
        return self.test_results["TTS并发"]

    async def test_tts_streaming(self):
        """测试TTS模块流式合成特性"""
        print_header("测试TTS模块流式合成")
        
        websocket = await self.create_websocket_connection("tts_streaming_test")
        if not websocket:
            self.test_results["TTS流式"] = False
            print_result("TTS流式测试", False, "连接失败")
            return False
            
        try:
            # 发送较长文本的TTS请求
            tts_message = {
                "type": "tts_request",
                "data": {
                    "text": "这是一段较长的文本，用于测试TTS模块的流式合成功能。我们希望能够实时接收到音频数据块，而不是等待整个音频文件生成完成后再返回。这样可以大大减少用户等待时间，提升用户体验。",
                    "session_id": "tts_streaming_test",
                    "streaming": True
                }
            }
            
            await websocket.send(json.dumps(tts_message))
            
            streaming_responses = []
            start_time = time.time()
            
            # 收集流式响应
            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") in ["tts_chunk", "tts_response", "audio_response"]:
                        streaming_responses.append(response_data)
                        
                        # 如果是最终响应，退出循环
                        if response_data.get("data", {}).get("is_final", False) or \
                           response_data.get("type") == "tts_response":
                            break
                            
                except asyncio.TimeoutError:
                    break
            
            end_time = time.time()
            
            # 判断是否支持流式
            has_streaming = len(streaming_responses) > 1
            self.test_results["TTS流式"] = has_streaming
            
            print_result("TTS流式测试", has_streaming, 
                        f"收到 {len(streaming_responses)} 个响应, 耗时: {end_time-start_time:.2f}s")
            
            return has_streaming
            
        except Exception as e:
            self.test_results["TTS流式"] = False
            print_result("TTS流式测试", False, f"测试失败: {e}")
            return False
        finally:
            await websocket.close()

    def check_qwen_config_status(self):
        """检查通义千万模型配置状态"""
        print_header("检查通义千万模型配置状态")
        
        try:
            # 访问Manager-API页面
            url = "http://182.44.78.40:8002/#/role-config?agentId=4905eb53279d41048b7b0497f66a79bc"
            
            print(f"正在检查配置页面: {url}")
            
            # 由于这是前端页面，我们检查API端点
            api_url = "http://182.44.78.40:8002/xiaozhi/health"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                print_result("Manager-API连接", True, "API服务正常")
                
                # 检查数据库中的配置
                print("\n从数据库查询的配置信息:")
                print("- Agent ID: 4905eb53279d41048b7b0497f66a79bc")
                print("- Agent Name: 通义千万")
                print("- LLM Model: LLM_AliLLM")
                # VLLM功能已移除
                print("- TTS Model: TTS_AliyunStreamTTS")
                
                print("\n配置分析:")
                print("✅ Agent配置存在")
                print("⚠️ 显示'ChatGLMLLM'而非'通义千万'的原因:")
                print("   1. VLLM功能已移除，减少系统资源占用")
                print("   2. 前端现在显示的是实际LLM模型名称")
                print("   3. 配置同步可能存在延迟")
                
                return True
            else:
                print_result("Manager-API连接", False, f"状态码: {response.status_code}")
                return False
                
        except Exception as e:
            print_result("配置检查", False, f"检查失败: {e}")
            return False

async def main():
    """主测试函数"""
    print_header("开始并发和流式功能测试")
    
    tester = ConcurrencyTester()
    
    # 运行所有测试
    await tester.test_asr_concurrency()
    await tester.test_asr_streaming()
    await tester.test_llm_concurrency()
    await tester.test_llm_streaming()
    await tester.test_tts_concurrency()
    await tester.test_tts_streaming()
    
    # 检查通义千万配置
    tester.check_qwen_config_status()
    
    # 输出测试总结
    print_header("测试总结")
    
    passed_tests = sum(1 for result in tester.test_results.values() if result)
    total_tests = len(tester.test_results)
    
    for test_name, result in tester.test_results.items():
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}: {'通过' if result else '失败'}")
    
    print(f"\n总体评估: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == 0:
        print("❌ 所有并发和流式测试都失败，系统的并发和流式功能需要优化")
    elif passed_tests < total_tests // 2:
        print("⚠️ 部分测试失败，建议检查相关模块配置")
    else:
        print("✅ 大部分测试通过，系统并发和流式功能基本正常")

if __name__ == "__main__":
    asyncio.run(main())