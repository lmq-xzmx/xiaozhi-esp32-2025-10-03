#!/usr/bin/env python3
"""
ASR服务快速验证脚本
用于快速检查ASR优化后的服务是否正常运行
"""

import requests
import json
import time
import base64
import numpy as np

def generate_test_audio(duration_ms=1000, sample_rate=16000):
    """生成测试音频数据"""
    samples = int(duration_ms * sample_rate / 1000)
    t = np.linspace(0, duration_ms/1000, samples)
    frequency = 440  # A4音符
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()

def test_asr_service(base_url="http://localhost:8000"):
    """测试ASR服务"""
    print("🔍 开始ASR服务快速验证...")
    
    # 1. 健康检查
    print("\n1. 健康检查...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ 服务健康检查通过")
        else:
            print(f"   ❌ 健康检查失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ 无法连接到服务: {e}")
        return False
    
    # 2. 获取服务统计
    print("\n2. 获取服务统计...")
    try:
        response = requests.get(f"{base_url}/asr/stats", timeout=10)
        if response.status_code == 200:
            stats = response.json()
            print("   ✅ 服务统计获取成功:")
            print(f"      - 活跃连接: {stats.get('active_connections', 'N/A')}")
            print(f"      - 队列长度: {stats.get('queue_length', 'N/A')}")
            print(f"      - 内存使用: {stats.get('memory_usage_mb', 'N/A')}MB")
        else:
            print(f"   ⚠️  统计信息获取失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  统计信息获取异常: {e}")
    
    # 3. ASR识别测试
    print("\n3. ASR识别测试...")
    try:
        # 生成测试音频数据
        test_audio_bytes = generate_test_audio()
        test_audio_b64 = base64.b64encode(test_audio_bytes).decode('utf-8')
        
        # 发送ASR请求
        asr_data = {
            "session_id": "test_session_001",
            "audio_data": test_audio_b64,
            "sample_rate": 16000,
            "language": "zh",
            "priority": 2,
            "timestamp": time.time()
        }
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(f"{base_url}/asr/recognize", json=asr_data, headers=headers, timeout=15)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ ASR识别成功")
            print(f"      - 识别文本: {result.get('text', 'N/A')}")
            print(f"      - 置信度: {result.get('confidence', 'N/A')}")
            print(f"      - 处理时间: {result.get('processing_time', 'N/A')}s")
            print(f"      - 音频大小: {result.get('audio_size', 'N/A')} bytes")
            print(f"      - 缓存命中: {result.get('cached', False)}")
        else:
            print(f"   ❌ ASR识别失败: HTTP {response.status_code}")
            print(f"      响应内容: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ ASR识别异常: {e}")
        return False
    
    # 4. 简单并发测试
    print("\n4. 简单并发测试...")
    try:
        import concurrent.futures
        import threading
        
        def send_request(session_id):
            audio_data_bytes = generate_test_audio(500)  # 0.5秒音频
            audio_b64 = base64.b64encode(audio_data_bytes).decode('utf-8')
            
            request_data = {
                "session_id": session_id,
                "audio_data": audio_b64,
                "sample_rate": 16000,
                "language": "zh",
                "priority": 2,
                "timestamp": time.time()
            }
            
            start_time = time.time()
            response = requests.post(f"{base_url}/asr/recognize", json=request_data, timeout=10)
            end_time = time.time()
            
            return {
                "success": response.status_code == 200,
                "latency": end_time - start_time,
                "session_id": session_id
            }
        
        # 发送5个并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_request, f"concurrent_test_{i}") for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        success_count = sum(1 for r in results if r["success"])
        avg_latency = sum(r["latency"] for r in results if r["success"]) / max(success_count, 1)
        
        print(f"   📊 并发测试结果: {success_count}/5 成功")
        print(f"      平均延迟: {avg_latency:.3f}s")
        
        if success_count >= 4:  # 至少80%成功率
            print("   ✅ 并发测试通过")
        else:
            print("   ⚠️  并发测试部分失败")
            
    except Exception as e:
        print(f"   ⚠️  并发测试异常: {e}")
    
    print("\n🎉 ASR服务快速验证完成!")
    return True

if __name__ == "__main__":
    test_asr_service()