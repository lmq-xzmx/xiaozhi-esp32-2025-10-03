#!/usr/bin/env python3
"""
16GB服务器高并发ASR性能测试
测试80-100并发请求的处理能力
"""

import asyncio
import aiohttp
import time
import json
import base64
import numpy as np
from typing import List, Dict, Any
import statistics

def generate_test_audio(duration_ms=1000, sample_rate=16000, frequency=440):
    """生成测试音频数据"""
    samples = int(duration_ms * sample_rate / 1000)
    t = np.linspace(0, duration_ms/1000, samples)
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()

async def send_asr_request(session: aiohttp.ClientSession, session_id: str, base_url: str) -> Dict[str, Any]:
    """发送单个ASR请求"""
    try:
        # 生成测试音频
        audio_bytes = generate_test_audio(1000, frequency=440 + (hash(session_id) % 200))
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        request_data = {
            "session_id": session_id,
            "audio_data": audio_b64,
            "sample_rate": 16000,
            "language": "zh",
            "priority": 2,
            "timestamp": time.time()
        }
        
        start_time = time.time()
        async with session.post(f"{base_url}/asr/recognize", json=request_data) as response:
            end_time = time.time()
            
            if response.status == 200:
                result = await response.json()
                return {
                    "success": True,
                    "session_id": session_id,
                    "latency": end_time - start_time,
                    "processing_time": result.get("processing_time", 0),
                    "cached": result.get("cached", False),
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", 0)
                }
            else:
                return {
                    "success": False,
                    "session_id": session_id,
                    "error": f"HTTP {response.status}",
                    "latency": end_time - start_time
                }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "error": str(e),
            "latency": 0
        }

async def concurrent_test(base_url: str, concurrent_count: int, test_name: str) -> Dict[str, Any]:
    """执行并发测试"""
    print(f"\n🚀 开始 {test_name} - {concurrent_count}并发测试")
    
    connector = aiohttp.TCPConnector(limit=200, limit_per_host=200)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 创建并发任务
        tasks = []
        for i in range(concurrent_count):
            session_id = f"16gb_test_{concurrent_count}_{i:03d}"
            task = send_asr_request(session, session_id, base_url)
            tasks.append(task)
        
        # 执行并发测试
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 分析结果
        successful_results = [r for r in results if isinstance(r, dict) and r.get("success", False)]
        failed_results = [r for r in results if isinstance(r, dict) and not r.get("success", False)]
        exception_results = [r for r in results if isinstance(r, Exception)]
        
        total_duration = end_time - start_time
        success_count = len(successful_results)
        failure_count = len(failed_results) + len(exception_results)
        success_rate = (success_count / concurrent_count) * 100
        
        # 计算性能指标
        if successful_results:
            latencies = [r["latency"] for r in successful_results]
            processing_times = [r["processing_time"] for r in successful_results]
            
            avg_latency = statistics.mean(latencies)
            p95_latency = statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0]
            avg_processing_time = statistics.mean(processing_times)
            throughput = success_count / total_duration
            
            cache_hits = sum(1 for r in successful_results if r.get("cached", False))
            cache_hit_rate = (cache_hits / success_count) * 100
        else:
            avg_latency = p95_latency = avg_processing_time = throughput = cache_hit_rate = 0
        
        return {
            "test_name": test_name,
            "concurrent_count": concurrent_count,
            "total_duration": total_duration,
            "success_count": success_count,
            "failure_count": failure_count,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "avg_processing_time": avg_processing_time,
            "throughput": throughput,
            "cache_hit_rate": cache_hit_rate,
            "successful_results": successful_results[:5],  # 只保留前5个成功结果作为样本
            "failed_results": failed_results[:3]  # 只保留前3个失败结果
        }

async def main():
    base_url = "http://localhost:8001"
    
    print("🎯 16GB服务器ASR高并发性能测试")
    print("=" * 60)
    
    # 测试配置
    test_configs = [
        (50, "中等并发测试"),
        (80, "目标并发测试"),
        (100, "极限并发测试"),
        (120, "超限并发测试")
    ]
    
    all_results = []
    
    for concurrent_count, test_name in test_configs:
        try:
            result = await concurrent_test(base_url, concurrent_count, test_name)
            all_results.append(result)
            
            # 打印测试结果
            print(f"\n📊 {test_name} 结果:")
            print(f"   并发数: {result['concurrent_count']}")
            print(f"   总耗时: {result['total_duration']:.2f}s")
            print(f"   成功率: {result['success_rate']:.1f}% ({result['success_count']}/{result['concurrent_count']})")
            print(f"   平均延迟: {result['avg_latency']:.3f}s")
            print(f"   P95延迟: {result['p95_latency']:.3f}s")
            print(f"   平均处理时间: {result['avg_processing_time']:.3f}s")
            print(f"   吞吐量: {result['throughput']:.1f} req/s")
            print(f"   缓存命中率: {result['cache_hit_rate']:.1f}%")
            
            if result['failure_count'] > 0:
                print(f"   ⚠️  失败数: {result['failure_count']}")
                if result['failed_results']:
                    print(f"   失败原因: {result['failed_results'][0].get('error', 'Unknown')}")
            
            # 等待一下再进行下一个测试
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
    
    # 生成总结报告
    print(f"\n📈 16GB服务器ASR性能测试总结")
    print("=" * 60)
    
    for result in all_results:
        if result['success_rate'] >= 95:
            status = "✅ 优秀"
        elif result['success_rate'] >= 80:
            status = "⚠️  良好"
        else:
            status = "❌ 需优化"
        
        print(f"{status} {result['concurrent_count']}并发: "
              f"成功率{result['success_rate']:.1f}%, "
              f"吞吐量{result['throughput']:.1f}req/s, "
              f"延迟{result['avg_latency']:.3f}s")
    
    # 找出最佳性能点
    best_result = max(all_results, key=lambda x: x['throughput'] if x['success_rate'] >= 90 else 0)
    if best_result['success_rate'] >= 90:
        print(f"\n🏆 最佳性能配置: {best_result['concurrent_count']}并发")
        print(f"   推荐设备支持数: {int(best_result['concurrent_count'] * 0.8)}-{best_result['concurrent_count']}台")
        print(f"   预期吞吐量: {best_result['throughput']:.1f} req/s")

if __name__ == "__main__":
    asyncio.run(main())