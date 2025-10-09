#!/usr/bin/env python3
"""
ASR支撑能力翻倍测试 - 160并发压力测试
验证4核16GB服务器在极限优化配置下的ASR性能
"""

import asyncio
import aiohttp
import time
import json
import base64
import numpy as np
from typing import List, Dict, Any
import statistics
import sys

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
        async with session.post(f"{base_url}/asr/recognize", json=request_data, timeout=aiohttp.ClientTimeout(total=10)) as response:
            end_time = time.time()
            
            if response.status == 200:
                result = await response.json()
                return {
                    "success": True,
                    "session_id": session_id,
                    "latency": end_time - start_time,
                    "processing_time": result.get("processing_time", 0),
                    "text": result.get("text", ""),
                    "confidence": result.get("confidence", 0),
                    "cache_hit": result.get("cache_hit", False)
                }
            else:
                return {
                    "success": False,
                    "session_id": session_id,
                    "latency": end_time - start_time,
                    "error": f"HTTP {response.status}",
                    "processing_time": 0
                }
                
    except Exception as e:
        return {
            "success": False,
            "session_id": session_id,
            "latency": 0,
            "error": str(e),
            "processing_time": 0
        }

async def concurrent_test(base_url: str, concurrent_count: int, test_name: str) -> Dict[str, Any]:
    """执行并发测试"""
    print(f"\n🔄 {test_name} - {concurrent_count}并发测试")
    print("=" * 60)
    
    connector = aiohttp.TCPConnector(limit=concurrent_count + 10, limit_per_host=concurrent_count + 10)
    timeout = aiohttp.ClientTimeout(total=30)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # 创建并发任务
        tasks = []
        for i in range(concurrent_count):
            session_id = f"{test_name}_{i:03d}"
            task = send_asr_request(session, session_id, base_url)
            tasks.append(task)
        
        # 执行并发测试
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 统计结果
        successful_results = []
        failed_results = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_results.append({"error": str(result), "success": False})
            elif result.get("success", False):
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        # 计算统计数据
        total_requests = len(results)
        successful_requests = len(successful_results)
        failed_requests = len(failed_results)
        success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
        
        latencies = [r["latency"] for r in successful_results if r["latency"] > 0]
        processing_times = [r["processing_time"] for r in successful_results if r["processing_time"] > 0]
        cache_hits = sum(1 for r in successful_results if r.get("cache_hit", False))
        
        total_time = end_time - start_time
        throughput = successful_requests / total_time if total_time > 0 else 0
        
        # 输出结果
        print(f"📊 测试结果:")
        print(f"   总请求数: {total_requests}")
        print(f"   成功请求: {successful_requests}")
        print(f"   失败请求: {failed_requests}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   总耗时: {total_time:.3f}s")
        print(f"   吞吐量: {throughput:.1f} req/s")
        
        if latencies:
            print(f"   平均延迟: {statistics.mean(latencies):.3f}s")
            print(f"   P50延迟: {statistics.median(latencies):.3f}s")
            print(f"   P95延迟: {np.percentile(latencies, 95):.3f}s")
            print(f"   P99延迟: {np.percentile(latencies, 99):.3f}s")
        
        if processing_times:
            print(f"   平均处理时间: {statistics.mean(processing_times):.3f}s")
        
        print(f"   缓存命中: {cache_hits}/{successful_requests}")
        
        # 失败分析
        if failed_requests > 0:
            print(f"\n❌ 失败分析:")
            error_types = {}
            for result in failed_results:
                error = result.get("error", "Unknown")
                error_types[error] = error_types.get(error, 0) + 1
            
            for error, count in error_types.items():
                print(f"   {error}: {count}次")
        
        return {
            "concurrent_count": concurrent_count,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": success_rate,
            "total_time": total_time,
            "throughput": throughput,
            "avg_latency": statistics.mean(latencies) if latencies else 0,
            "p95_latency": np.percentile(latencies, 95) if latencies else 0,
            "avg_processing_time": statistics.mean(processing_times) if processing_times else 0,
            "cache_hits": cache_hits
        }

async def get_service_stats(base_url: str) -> Dict[str, Any]:
    """获取服务统计信息"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/asr/stats") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

async def main():
    """主测试函数"""
    base_url = "http://localhost:8001"
    
    print("🚀 ASR支撑能力翻倍测试 - 160并发压力测试")
    print("=" * 80)
    
    # 检查服务状态
    print("1. 检查服务状态...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"   ✅ 服务状态: {health.get('status', 'unknown')}")
                    print(f"   📊 最大并发: {health.get('max_concurrent', 'unknown')}")
                else:
                    print(f"   ❌ 服务不可用: HTTP {response.status}")
                    sys.exit(1)
    except Exception as e:
        print(f"   ❌ 连接失败: {e}")
        sys.exit(1)
    
    # 获取初始统计
    print("\n2. 获取初始统计...")
    initial_stats = await get_service_stats(base_url)
    if "error" not in initial_stats:
        print(f"   📊 当前配置: 批处理={initial_stats['processor']['batch_size']}, 最大并发={initial_stats['service']['max_concurrent']}")
        print(f"   📈 历史请求: {initial_stats['service']['total_requests']}")
    
    # 渐进式并发测试
    test_scenarios = [
        (40, "基线测试"),
        (80, "当前配置测试"),
        (120, "中等负载测试"),
        (160, "目标并发测试"),
        (200, "极限压力测试")
    ]
    
    results = []
    
    for concurrent_count, test_name in test_scenarios:
        try:
            result = await concurrent_test(base_url, concurrent_count, test_name)
            results.append(result)
            
            # 如果成功率低于80%，停止更高并发的测试
            if result["success_rate"] < 80:
                print(f"\n⚠️  成功率低于80%，停止更高并发测试")
                break
                
            # 等待系统恢复
            if concurrent_count < 200:
                print("   ⏳ 等待系统恢复...")
                await asyncio.sleep(3)
                
        except Exception as e:
            print(f"   ❌ 测试失败: {e}")
            break
    
    # 最终统计
    print("\n" + "=" * 80)
    print("📈 ASR支撑能力翻倍测试总结")
    print("=" * 80)
    
    print("\n📊 各并发级别性能对比:")
    print(f"{'并发数':<8} {'成功率':<8} {'吞吐量':<12} {'平均延迟':<10} {'P95延迟':<10} {'状态':<10}")
    print("-" * 70)
    
    for result in results:
        concurrent = result["concurrent_count"]
        success_rate = result["success_rate"]
        throughput = result["throughput"]
        avg_latency = result["avg_latency"]
        p95_latency = result["p95_latency"]
        
        # 状态判断
        if success_rate >= 95 and avg_latency < 0.6:
            status = "✅ 优秀"
        elif success_rate >= 90 and avg_latency < 0.8:
            status = "🟡 良好"
        elif success_rate >= 80:
            status = "🟠 可用"
        else:
            status = "❌ 不稳定"
        
        print(f"{concurrent:<8} {success_rate:<7.1f}% {throughput:<11.1f} {avg_latency:<9.3f}s {p95_latency:<9.3f}s {status}")
    
    # 找出最佳配置
    stable_results = [r for r in results if r["success_rate"] >= 95]
    if stable_results:
        best_result = max(stable_results, key=lambda x: x["concurrent_count"])
        print(f"\n🏆 推荐生产配置:")
        print(f"   最大稳定并发: {best_result['concurrent_count']}台设备")
        print(f"   预期吞吐量: {best_result['throughput']:.1f} req/s")
        print(f"   平均延迟: {best_result['avg_latency']:.3f}s")
        print(f"   成功率: {best_result['success_rate']:.1f}%")
        
        # 计算相对于80并发的提升
        baseline_80 = next((r for r in results if r["concurrent_count"] == 80), None)
        if baseline_80:
            improvement = (best_result["concurrent_count"] / 80 - 1) * 100
            throughput_improvement = (best_result["throughput"] / baseline_80["throughput"] - 1) * 100
            print(f"\n📈 相对于80并发的提升:")
            print(f"   并发能力提升: +{improvement:.0f}%")
            print(f"   吞吐量提升: +{throughput_improvement:.0f}%")
            
            if improvement >= 100:
                print("   🎉 成功实现ASR支撑能力翻倍目标！")
            elif improvement >= 50:
                print("   🎯 显著提升ASR支撑能力！")
            else:
                print("   📊 有一定提升，但未达到翻倍目标")
    
    # 获取最终统计
    print(f"\n3. 获取最终统计...")
    final_stats = await get_service_stats(base_url)
    if "error" not in final_stats:
        total_requests = final_stats['service']['total_requests']
        cache_hits = final_stats['service']['cache_hits']
        cache_hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
        print(f"   📊 总处理请求: {total_requests}")
        print(f"   💾 缓存命中率: {cache_hit_rate:.1f}%")
        print(f"   📦 当前缓存大小: {final_stats['processor']['cache_size']}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")