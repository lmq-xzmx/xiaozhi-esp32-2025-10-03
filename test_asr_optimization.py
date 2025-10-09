#!/usr/bin/env python3
"""
ASR优化配置验证测试
测试新的配置：MAX_CONCURRENT=25, BATCH_SIZE=10, CACHE_SIZE_MB=768
"""

import asyncio
import aiohttp
import time
import json
import base64
import numpy as np
from typing import List, Dict

class ASROptimizationTester:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_audio = self.generate_test_audio()
        
    def generate_test_audio(self) -> str:
        """生成测试音频数据"""
        # 生成2秒的测试音频 (16kHz, 16bit)
        duration = 2.0
        sample_rate = 16000
        samples = int(duration * sample_rate)
        
        # 生成正弦波音频
        t = np.linspace(0, duration, samples)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440Hz正弦波
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 转换为base64
        audio_bytes = audio_int16.tobytes()
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    async def health_check(self) -> Dict:
        """健康检查"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as response:
                return await response.json()
    
    async def get_stats(self) -> Dict:
        """获取服务统计"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/asr/stats") as response:
                return await response.json()
    
    async def single_asr_request(self, session_id: str) -> Dict:
        """单个ASR请求"""
        async with aiohttp.ClientSession() as session:
            data = {
                "session_id": session_id,
                "audio_data": self.test_audio,
                "sample_rate": 16000,
                "language": "zh",
                "priority": 2
            }
            
            start_time = time.time()
            async with session.post(f"{self.base_url}/asr/recognize", json=data) as response:
                result = await response.json()
                end_time = time.time()
                
                result['total_latency'] = end_time - start_time
                return result
    
    async def concurrent_test(self, num_requests: int) -> List[Dict]:
        """并发测试"""
        print(f"🔄 发送 {num_requests} 个并发请求...")
        
        start_time = time.time()
        tasks = []
        
        for i in range(num_requests):
            task = self.single_asr_request(f"test_session_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        # 统计结果
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed_results = [r for r in results if not isinstance(r, dict) or not r.get('success')]
        
        total_time = end_time - start_time
        throughput = len(successful_results) / total_time if total_time > 0 else 0
        
        if successful_results:
            avg_latency = sum(r['total_latency'] for r in successful_results) / len(successful_results)
            avg_processing_time = sum(r.get('processing_time', 0) for r in successful_results) / len(successful_results)
        else:
            avg_latency = 0
            avg_processing_time = 0
        
        print(f"   ✅ 并发测试完成")
        print(f"      📊 成功请求: {len(successful_results)}/{num_requests}")
        print(f"      ❌ 失败请求: {len(failed_results)}")
        print(f"      ⏱️  平均延迟: {avg_latency:.3f}s")
        print(f"      🔧 平均处理时间: {avg_processing_time:.3f}s")
        print(f"      🚀 总耗时: {total_time:.3f}s")
        print(f"      📈 吞吐量: {throughput:.2f} req/s")
        
        return results
    
    async def stress_test(self) -> Dict:
        """压力测试 - 测试25并发能力"""
        print("\n🔥 压力测试 - 25并发请求")
        print("=" * 50)
        
        # 测试25个并发请求
        results_25 = await self.concurrent_test(25)
        
        print("\n🔥 极限测试 - 30并发请求")
        print("=" * 50)
        
        # 测试30个并发请求（超过配置限制）
        results_30 = await self.concurrent_test(30)
        
        return {
            'test_25_concurrent': results_25,
            'test_30_concurrent': results_30
        }
    
    async def run_optimization_test(self):
        """运行完整的优化测试"""
        print("🚀 ASR优化配置验证测试")
        print("=" * 50)
        print("📋 测试配置:")
        print("   - MAX_CONCURRENT: 25")
        print("   - BATCH_SIZE: 10") 
        print("   - CACHE_SIZE_MB: 768")
        print("   - WORKER_THREADS: 4")
        print("=" * 50)
        
        # 1. 健康检查
        print("\n1. 服务健康检查...")
        try:
            health = await self.health_check()
            print(f"   ✅ 服务状态: {health.get('status', 'unknown')}")
        except Exception as e:
            print(f"   ❌ 健康检查失败: {e}")
            return
        
        # 2. 获取初始统计
        print("\n2. 获取服务统计...")
        try:
            stats = await self.get_stats()
            print(f"   📊 批处理大小: {stats.get('batch_size', 'unknown')}")
            print(f"   🔄 最大并发: {stats.get('max_concurrent', 'unknown')}")
            print(f"   💾 缓存大小: {stats.get('cache_size', 'unknown')}")
            print(f"   📈 总请求数: {stats.get('total_requests', 'unknown')}")
        except Exception as e:
            print(f"   ❌ 统计获取失败: {e}")
        
        # 3. 基准测试 - 10并发
        print("\n3. 基准测试 - 10并发请求")
        print("=" * 50)
        await self.concurrent_test(10)
        
        # 4. 压力测试
        await self.stress_test()
        
        # 5. 最终统计
        print("\n5. 最终服务统计...")
        try:
            final_stats = await self.get_stats()
            print(f"   📊 总请求数: {final_stats.get('total_requests', 'unknown')}")
            print(f"   🎯 缓存命中: {final_stats.get('cache_hits', 'unknown')}")
            cache_hit_rate = final_stats.get('cache_hit_rate', 0)
            print(f"   📈 缓存命中率: {cache_hit_rate:.3f}")
            print(f"   💾 当前缓存大小: {final_stats.get('cache_size', 'unknown')}")
        except Exception as e:
            print(f"   ❌ 最终统计获取失败: {e}")
        
        print("\n" + "=" * 50)
        print("🎉 ASR优化配置验证测试完成！")

async def main():
    tester = ASROptimizationTester()
    await tester.run_optimization_test()

if __name__ == "__main__":
    asyncio.run(main())