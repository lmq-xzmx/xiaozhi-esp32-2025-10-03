#!/usr/bin/env python3
"""
25并发ASR压力测试
验证优化配置的并发处理能力
"""

import asyncio
import aiohttp
import time
import json
import base64
import numpy as np
from typing import List, Dict, Any

# 生成测试音频数据
def generate_test_audio(duration_seconds: float = 2.0, sample_rate: int = 16000) -> bytes:
    """生成测试用的正弦波音频"""
    t = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), False)
    frequency = 440  # A4音符
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()

async def send_asr_request(session: aiohttp.ClientSession, session_id: str, audio_data: str) -> Dict[str, Any]:
    """发送ASR请求"""
    url = "http://localhost:8001/asr/recognize"
    payload = {
        "session_id": session_id,
        "audio_data": audio_data,
        "sample_rate": 16000,
        "language": "zh",
        "priority": 2,
        "timestamp": time.time()
    }
    
    start_time = time.time()
    try:
        async with session.post(url, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                end_time = time.time()
                return {
                    "success": True,
                    "session_id": session_id,
                    "latency": end_time - start_time,
                    "processing_time": result.get("processing_time", 0),
                    "cache_hit": result.get("cache_hit", False),
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", 0)
                }
            else:
                return {
                    "success": False,
                    "session_id": session_id,
                    "error": f"HTTP {response.status}",
                    "latency": time.time() - start_time
                }
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "error": str(e),
            "latency": time.time() - start_time
        }

async def concurrent_test(num_concurrent: int = 25) -> Dict[str, Any]:
    """并发测试"""
    print(f"🔥 开始 {num_concurrent} 并发ASR测试...")
    
    # 生成测试音频
    audio_bytes = generate_test_audio()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # 创建会话
    async with aiohttp.ClientSession() as session:
        # 创建并发任务
        tasks = []
        for i in range(num_concurrent):
            session_id = f"concurrent_test_{i}"
            task = send_asr_request(session, session_id, audio_b64)
            tasks.append(task)
        
        # 执行并发测试
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # 统计结果
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        total_time = end_time - start_time
        avg_latency = sum(r["latency"] for r in successful) / len(successful) if successful else 0
        avg_processing_time = sum(r["processing_time"] for r in successful) / len(successful) if successful else 0
        cache_hits = sum(1 for r in successful if r.get("cache_hit", False))
        
        return {
            "total_requests": num_concurrent,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / num_concurrent * 100,
            "total_time": total_time,
            "avg_latency": avg_latency,
            "avg_processing_time": avg_processing_time,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / len(successful) * 100 if successful else 0,
            "throughput": len(successful) / total_time if total_time > 0 else 0,
            "errors": [r["error"] for r in failed]
        }

async def get_service_stats() -> Dict[str, Any]:
    """获取服务统计"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8001/asr/stats") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

async def main():
    """主函数"""
    print("🚀 ASR 25并发压力测试")
    print("=" * 50)
    
    # 获取初始统计
    print("📊 获取初始服务统计...")
    initial_stats = await get_service_stats()
    print(f"   📈 最大并发: {initial_stats.get('max_concurrent', 'unknown')}")
    print(f"   📊 批处理大小: {initial_stats.get('batch_size', 'unknown')}")
    print(f"   📈 总请求数: {initial_stats.get('total_requests', 'unknown')}")
    print()
    
    # 执行并发测试
    test_results = await concurrent_test(25)
    
    # 显示结果
    print("📊 测试结果:")
    print(f"   🎯 总请求数: {test_results['total_requests']}")
    print(f"   ✅ 成功请求: {test_results['successful_requests']}")
    print(f"   ❌ 失败请求: {test_results['failed_requests']}")
    print(f"   📈 成功率: {test_results['success_rate']:.1f}%")
    print(f"   ⏱️  平均延迟: {test_results['avg_latency']:.3f}s")
    print(f"   🔧 平均处理时间: {test_results['avg_processing_time']:.3f}s")
    print(f"   💾 缓存命中: {test_results['cache_hits']}/{test_results['successful_requests']}")
    print(f"   📈 缓存命中率: {test_results['cache_hit_rate']:.1f}%")
    print(f"   🚀 总耗时: {test_results['total_time']:.3f}s")
    print(f"   📊 吞吐量: {test_results['throughput']:.2f} req/s")
    
    if test_results['errors']:
        print(f"   ⚠️  错误信息: {test_results['errors'][:3]}")  # 只显示前3个错误
    
    print()
    
    # 获取最终统计
    print("📊 获取最终服务统计...")
    final_stats = await get_service_stats()
    print(f"   📈 总请求数: {final_stats.get('total_requests', 'unknown')}")
    print(f"   🎯 缓存命中: {final_stats.get('cache_hits', 'unknown')}")
    print(f"   💾 当前缓存大小: {final_stats.get('cache_size', 'unknown')}")
    
    print()
    print("🎉 25并发压力测试完成！")

if __name__ == "__main__":
    asyncio.run(main())