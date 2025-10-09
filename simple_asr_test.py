#!/usr/bin/env python3
"""
简化的ASR测试脚本
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

def test_asr_optimization():
    """测试ASR优化效果"""
    base_url = "http://localhost:8001"
    
    print("🚀 ASR优化效果测试")
    print("=" * 50)
    
    # 1. 健康检查
    print("\n1. 服务健康检查...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ 服务状态: {health_data.get('status')}")
            print(f"   📊 优化配置: {health_data.get('optimization')}")
            print(f"   🔄 当前并发: {health_data.get('current_concurrent')}")
            print(f"   📈 最大并发: {health_data.get('max_concurrent')}")
        else:
            print(f"   ❌ 健康检查失败: {response.status_code}")
            return
    except Exception as e:
        print(f"   ❌ 健康检查异常: {e}")
        return
    
    # 2. 获取初始统计
    print("\n2. 获取服务统计...")
    try:
        response = requests.get(f"{base_url}/asr/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"   📊 批处理大小: {stats['processor']['batch_size']}")
            print(f"   💾 缓存大小: {stats['processor']['cache_size']}")
            print(f"   📈 总请求数: {stats['service']['total_requests']}")
            print(f"   🎯 缓存命中: {stats['service']['cache_hits']}")
        else:
            print(f"   ⚠️  统计获取失败: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  统计获取异常: {e}")
    
    # 3. 单次ASR测试
    print("\n3. 单次ASR识别测试...")
    try:
        # 生成测试音频
        audio_bytes = generate_test_audio(1000)  # 1秒音频
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # 构建请求
        request_data = {
            "session_id": "optimization_test_001",
            "audio_data": audio_b64,
            "sample_rate": 16000,
            "language": "zh",
            "priority": 2,
            "timestamp": time.time()
        }
        
        # 发送请求
        start_time = time.time()
        response = requests.post(
            f"{base_url}/asr/recognize", 
            json=request_data, 
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ ASR识别成功")
            print(f"      📝 识别文本: {result.get('text')}")
            print(f"      🎯 置信度: {result.get('confidence'):.3f}")
            print(f"      ⏱️  处理时间: {result.get('processing_time'):.3f}s")
            print(f"      📦 音频大小: {result.get('audio_size')} bytes")
            print(f"      💾 缓存命中: {result.get('cached')}")
            print(f"      🌐 总延迟: {end_time - start_time:.3f}s")
        else:
            print(f"   ❌ ASR识别失败: {response.status_code}")
            print(f"      响应: {response.text}")
    except Exception as e:
        print(f"   ❌ ASR识别异常: {e}")
    
    # 4. 缓存测试
    print("\n4. 缓存效果测试...")
    try:
        # 重复相同请求测试缓存
        start_time = time.time()
        response = requests.post(
            f"{base_url}/asr/recognize", 
            json=request_data, 
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 缓存测试成功")
            print(f"      💾 缓存命中: {result.get('cached')}")
            print(f"      ⏱️  处理时间: {result.get('processing_time'):.3f}s")
            print(f"      🌐 总延迟: {end_time - start_time:.3f}s")
        else:
            print(f"   ❌ 缓存测试失败: {response.status_code}")
    except Exception as e:
        print(f"   ❌ 缓存测试异常: {e}")
    
    # 5. 并发测试
    print("\n5. 并发性能测试...")
    try:
        import concurrent.futures
        import threading
        
        def send_concurrent_request(session_id):
            """发送并发请求"""
            try:
                audio_bytes = generate_test_audio(500)  # 0.5秒音频
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                
                request_data = {
                    "session_id": f"concurrent_test_{session_id}",
                    "audio_data": audio_b64,
                    "sample_rate": 16000,
                    "language": "zh",
                    "priority": 2,
                    "timestamp": time.time()
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{base_url}/asr/recognize", 
                    json=request_data, 
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "success": True,
                        "session_id": session_id,
                        "latency": end_time - start_time,
                        "processing_time": result.get('processing_time', 0),
                        "cached": result.get('cached', False)
                    }
                else:
                    return {"success": False, "session_id": session_id, "error": response.status_code}
            except Exception as e:
                return {"success": False, "session_id": session_id, "error": str(e)}
        
        # 并发测试 - 10个请求
        concurrent_count = 10
        print(f"   🔄 发送 {concurrent_count} 个并发请求...")
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_count) as executor:
            futures = [executor.submit(send_concurrent_request, i) for i in range(concurrent_count)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        end_time = time.time()
        
        # 分析结果
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if successful_requests:
            avg_latency = sum(r["latency"] for r in successful_requests) / len(successful_requests)
            avg_processing = sum(r["processing_time"] for r in successful_requests) / len(successful_requests)
            cache_hits = sum(1 for r in successful_requests if r["cached"])
            
            print(f"   ✅ 并发测试完成")
            print(f"      📊 成功请求: {len(successful_requests)}/{concurrent_count}")
            print(f"      ⏱️  平均延迟: {avg_latency:.3f}s")
            print(f"      🔧 平均处理时间: {avg_processing:.3f}s")
            print(f"      💾 缓存命中: {cache_hits}/{len(successful_requests)}")
            print(f"      🚀 总耗时: {end_time - start_time:.3f}s")
            print(f"      📈 吞吐量: {len(successful_requests)/(end_time - start_time):.2f} req/s")
        
        if failed_requests:
            print(f"   ⚠️  失败请求: {len(failed_requests)}")
            for req in failed_requests[:3]:  # 只显示前3个错误
                print(f"      ❌ Session {req['session_id']}: {req['error']}")
    
    except Exception as e:
        print(f"   ❌ 并发测试异常: {e}")
    
    # 6. 最终统计
    print("\n6. 最终服务统计...")
    try:
        response = requests.get(f"{base_url}/asr/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print(f"   📊 总请求数: {stats['service']['total_requests']}")
            print(f"   🎯 缓存命中: {stats['service']['cache_hits']}")
            print(f"   📈 缓存命中率: {stats['processor']['cache_hit_rate']:.3f}")
            print(f"   ⏱️  平均处理时间: {stats['processor']['avg_processing_time']:.3f}s")
            print(f"   💾 当前缓存大小: {stats['processor']['cache_size']}")
        else:
            print(f"   ⚠️  统计获取失败: {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  统计获取异常: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 ASR优化效果测试完成！")

if __name__ == "__main__":
    test_asr_optimization()