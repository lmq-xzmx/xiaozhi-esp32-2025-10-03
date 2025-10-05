#!/usr/bin/env python3
"""
ASR并发性能测试脚本
测试多个设备同时进行语音识别的性能
"""
import asyncio
import websockets
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# 测试配置
SERVER_URL = "ws://localhost:8000/xiaozhi/v1/"
NUM_CONCURRENT_CONNECTIONS = 3  # 模拟3个设备同时连接
TEST_DURATION = 5  # 测试持续时间（秒）

class ASRConcurrencyTester:
    def __init__(self):
        self.results = []
        self.lock = threading.Lock()
    
    async def test_single_connection(self, connection_id):
        """测试单个连接的ASR性能"""
        try:
            uri = f"{SERVER_URL}?device_id=test_device_{connection_id}"
            
            async with websockets.connect(uri) as websocket:
                print(f"连接 {connection_id}: 已建立WebSocket连接")
                
                # 发送初始化消息
                init_message = {
                    "type": "init",
                    "device_id": f"test_device_{connection_id}",
                    "timestamp": time.time()
                }
                await websocket.send(json.dumps(init_message))
                
                # 模拟音频数据发送
                start_time = time.time()
                message_count = 0
                
                while time.time() - start_time < TEST_DURATION:
                    # 模拟音频数据包
                    audio_message = {
                        "type": "audio_data",
                        "device_id": f"test_device_{connection_id}",
                        "timestamp": time.time(),
                        "data": "fake_audio_data_" + str(message_count)
                    }
                    
                    send_time = time.time()
                    await websocket.send(json.dumps(audio_message))
                    
                    # 等待响应
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                        receive_time = time.time()
                        
                        with self.lock:
                            self.results.append({
                                'connection_id': connection_id,
                                'message_count': message_count,
                                'latency': receive_time - send_time,
                                'timestamp': receive_time
                            })
                        
                        message_count += 1
                        
                    except asyncio.TimeoutError:
                        print(f"连接 {connection_id}: 消息 {message_count} 超时")
                    
                    # 控制发送频率
                    await asyncio.sleep(0.5)
                
                print(f"连接 {connection_id}: 测试完成，发送了 {message_count} 条消息")
                
        except Exception as e:
            print(f"连接 {connection_id}: 发生错误 - {e}")
    
    async def run_concurrent_test(self):
        """运行并发测试"""
        print(f"开始ASR并发测试...")
        print(f"并发连接数: {NUM_CONCURRENT_CONNECTIONS}")
        print(f"测试持续时间: {TEST_DURATION}秒")
        print("-" * 50)
        
        # 创建并发任务
        tasks = []
        for i in range(NUM_CONCURRENT_CONNECTIONS):
            task = asyncio.create_task(self.test_single_connection(i + 1))
            tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分析结果
        self.analyze_results()
    
    def analyze_results(self):
        """分析测试结果"""
        if not self.results:
            print("❌ 没有收到任何响应，可能存在严重的并发问题")
            return
        
        print("\n" + "=" * 50)
        print("📊 测试结果分析")
        print("=" * 50)
        
        # 按连接分组统计
        connection_stats = {}
        for result in self.results:
            conn_id = result['connection_id']
            if conn_id not in connection_stats:
                connection_stats[conn_id] = []
            connection_stats[conn_id].append(result['latency'])
        
        total_messages = len(self.results)
        total_latency = sum(r['latency'] for r in self.results)
        avg_latency = total_latency / total_messages if total_messages > 0 else 0
        
        print(f"✅ 总消息数: {total_messages}")
        print(f"✅ 平均延迟: {avg_latency:.3f}秒")
        
        # 各连接统计
        for conn_id, latencies in connection_stats.items():
            count = len(latencies)
            avg_lat = sum(latencies) / count if count > 0 else 0
            max_lat = max(latencies) if latencies else 0
            min_lat = min(latencies) if latencies else 0
            
            print(f"📱 连接 {conn_id}: {count}条消息, 平均延迟 {avg_lat:.3f}s, 最大 {max_lat:.3f}s, 最小 {min_lat:.3f}s")
        
        # 并发性能评估
        if len(connection_stats) == NUM_CONCURRENT_CONNECTIONS:
            print(f"🎉 并发测试成功！所有 {NUM_CONCURRENT_CONNECTIONS} 个连接都能正常工作")
            if avg_latency < 2.0:
                print("⚡ 性能良好：平均延迟小于2秒")
            else:
                print("⚠️  性能一般：平均延迟较高，可能需要进一步优化")
        else:
            print(f"⚠️  部分连接失败：只有 {len(connection_stats)}/{NUM_CONCURRENT_CONNECTIONS} 个连接成功")

async def main():
    tester = ASRConcurrencyTester()
    await tester.run_concurrent_test()

if __name__ == "__main__":
    print("🚀 ASR并发性能测试工具")
    print("测试目标：验证ASR瓶颈修复效果")
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
