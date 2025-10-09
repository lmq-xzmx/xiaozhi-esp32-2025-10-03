#!/usr/bin/env python3
"""
优化后性能测试脚本
测试ASR服务的并发能力、响应时间和资源使用情况
"""

import asyncio
import aiohttp
import time
import json
import base64
import numpy as np
import psutil
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """测试结果数据类"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    throughput: float
    error_rate: float
    cache_hit_rate: float
    total_test_time: float

class OptimizedPerformanceTester:
    """优化后性能测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.test_audio = self._generate_test_audio()
        
    def _generate_test_audio(self) -> str:
        """生成测试音频数据（模拟1秒16kHz音频）"""
        # 生成1秒的正弦波音频数据
        sample_rate = 16000
        duration = 1.0
        frequency = 440  # A4音符
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # 转换为16位整数
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # 转换为bytes并编码为base64
        audio_bytes = audio_int16.tobytes()
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    async def test_single_request(self, session: aiohttp.ClientSession, 
                                session_id: str) -> Tuple[bool, float, Dict]:
        """测试单个ASR请求"""
        start_time = time.time()
        
        try:
            # 使用正确的请求格式
            request_data = {
                "session_id": session_id,
                "audio_data": self.test_audio,
                "sample_rate": 16000,
                "language": "zh",
                "priority": 2,
                "timestamp": time.time()
            }
            
            async with session.post(
                f"{self.base_url}/asr/recognize",
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    return True, response_time, result
                else:
                    logger.error(f"请求失败: {response.status}")
                    return False, response_time, {}
                    
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"请求异常: {e}")
            return False, response_time, {}
    
    async def test_concurrent_requests(self, concurrent_count: int, 
                                     total_requests: int) -> TestResult:
        """测试并发请求"""
        logger.info(f"开始并发测试: {concurrent_count}并发, 总请求数: {total_requests}")
        
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        response_times = []
        cache_hits = 0
        
        # 创建HTTP会话
        connector = aiohttp.TCPConnector(limit=concurrent_count * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            # 创建信号量控制并发数
            semaphore = asyncio.Semaphore(concurrent_count)
            
            async def make_request(request_id: int):
                async with semaphore:
                    session_id = f"test_session_{request_id}_{int(time.time())}"
                    success, response_time, result = await self.test_single_request(session, session_id)
                    
                    nonlocal successful_requests, failed_requests, cache_hits
                    
                    if success:
                        successful_requests += 1
                        response_times.append(response_time)
                        
                        # 检查是否命中缓存
                        if result.get('cached', False):
                            cache_hits += 1
                    else:
                        failed_requests += 1
            
            # 创建所有请求任务
            tasks = [make_request(i) for i in range(total_requests)]
            
            # 执行所有请求
            await asyncio.gather(*tasks, return_exceptions=True)
        
        total_test_time = time.time() - start_time
        
        # 计算统计数据
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = np.percentile(response_times, 95)
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = 0
        
        throughput = successful_requests / total_test_time if total_test_time > 0 else 0
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        cache_hit_rate = cache_hits / successful_requests if successful_requests > 0 else 0
        
        return TestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            throughput=throughput,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate,
            total_test_time=total_test_time
        )
    
    async def test_cache_effectiveness(self) -> Dict:
        """测试缓存效果"""
        logger.info("测试缓存效果...")
        
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            # 第一次请求（应该不命中缓存）
            session_id = f"cache_test_{int(time.time())}"
            success1, time1, result1 = await self.test_single_request(session, session_id)
            
            # 等待一秒
            await asyncio.sleep(1)
            
            # 第二次相同请求（应该命中缓存）
            success2, time2, result2 = await self.test_single_request(session, session_id)
            
            return {
                'first_request': {
                    'success': success1,
                    'response_time': time1,
                    'cached': result1.get('cached', False)
                },
                'second_request': {
                    'success': success2,
                    'response_time': time2,
                    'cached': result2.get('cached', False)
                },
                'cache_speedup': time1 / time2 if time2 > 0 else 0
            }
    
    async def get_system_stats(self) -> Dict:
        """获取系统资源使用统计"""
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'memory_available': psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
            'disk_usage': psutil.disk_usage('/').percent
        }
    
    async def get_service_stats(self) -> Dict:
        """获取服务统计信息"""
        try:
            connector = aiohttp.TCPConnector(limit=1)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.get(f"{self.base_url}/asr/stats", 
                                     timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {}
        except Exception as e:
            logger.error(f"获取服务统计失败: {e}")
            return {}
    
    def print_test_results(self, result: TestResult, test_name: str):
        """打印测试结果"""
        print(f"\n🚀 {test_name} 测试结果")
        print("=" * 60)
        print(f"📊 总请求数: {result.total_requests}")
        print(f"✅ 成功请求: {result.successful_requests}")
        print(f"❌ 失败请求: {result.failed_requests}")
        print(f"📈 成功率: {(1 - result.error_rate) * 100:.1f}%")
        print(f"⏱️  平均响应时间: {result.avg_response_time:.3f}s")
        print(f"⚡ 最快响应时间: {result.min_response_time:.3f}s")
        print(f"🐌 最慢响应时间: {result.max_response_time:.3f}s")
        print(f"📊 P95响应时间: {result.p95_response_time:.3f}s")
        print(f"🚀 吞吐量: {result.throughput:.2f} req/s")
        print(f"💾 缓存命中率: {result.cache_hit_rate * 100:.1f}%")
        print(f"🕐 总测试时间: {result.total_test_time:.2f}s")
    
    async def run_comprehensive_test(self):
        """运行综合性能测试"""
        print("🎯 开始优化后性能测试")
        print("=" * 60)
        
        # 获取测试前系统状态
        print("📊 测试前系统状态:")
        system_stats_before = await self.get_system_stats()
        for key, value in system_stats_before.items():
            print(f"   {key}: {value}")
        
        # 测试1: 单个请求基准测试
        print("\n🔍 测试1: 单个请求基准测试")
        single_result = await self.test_concurrent_requests(1, 1)
        self.print_test_results(single_result, "单个请求基准")
        
        # 测试2: 缓存效果测试
        print("\n💾 测试2: 缓存效果测试")
        cache_result = await self.test_cache_effectiveness()
        print(f"第一次请求: {cache_result['first_request']['response_time']:.3f}s (缓存: {cache_result['first_request']['cached']})")
        print(f"第二次请求: {cache_result['second_request']['response_time']:.3f}s (缓存: {cache_result['second_request']['cached']})")
        print(f"缓存加速比: {cache_result['cache_speedup']:.1f}x")
        
        # 测试3: 中等并发测试 (15并发)
        print("\n🔥 测试3: 中等并发测试 (15并发)")
        medium_result = await self.test_concurrent_requests(15, 30)
        self.print_test_results(medium_result, "中等并发")
        
        # 测试4: 高并发测试 (30并发)
        print("\n🚀 测试4: 高并发测试 (30并发)")
        high_result = await self.test_concurrent_requests(30, 60)
        self.print_test_results(high_result, "高并发")
        
        # 测试5: 极限并发测试 (40并发)
        print("\n⚡ 测试5: 极限并发测试 (40并发)")
        extreme_result = await self.test_concurrent_requests(40, 80)
        self.print_test_results(extreme_result, "极限并发")
        
        # 获取测试后系统状态
        print("\n📊 测试后系统状态:")
        system_stats_after = await self.get_system_stats()
        for key, value in system_stats_after.items():
            print(f"   {key}: {value}")
        
        # 获取服务统计
        print("\n🔧 服务统计信息:")
        service_stats = await self.get_service_stats()
        if service_stats:
            for key, value in service_stats.items():
                print(f"   {key}: {value}")
        
        # 性能对比总结
        print("\n📈 性能对比总结")
        print("=" * 60)
        print(f"单个请求延迟: {single_result.avg_response_time:.3f}s")
        print(f"中等并发(15): {medium_result.throughput:.2f} req/s, 成功率: {(1-medium_result.error_rate)*100:.1f}%")
        print(f"高并发(30): {high_result.throughput:.2f} req/s, 成功率: {(1-high_result.error_rate)*100:.1f}%")
        print(f"极限并发(40): {extreme_result.throughput:.2f} req/s, 成功率: {(1-extreme_result.error_rate)*100:.1f}%")
        
        # 推荐配置
        if high_result.error_rate < 0.05 and high_result.avg_response_time < 1.0:
            print("\n✅ 推荐: 系统可以稳定支持30并发")
        elif medium_result.error_rate < 0.05 and medium_result.avg_response_time < 1.0:
            print("\n⚠️  推荐: 系统可以稳定支持15并发")
        else:
            print("\n❌ 建议: 需要进一步优化配置")

async def main():
    """主函数"""
    tester = OptimizedPerformanceTester()
    
    # 等待服务启动
    print("⏳ 等待服务启动...")
    await asyncio.sleep(10)
    
    try:
        await tester.run_comprehensive_test()
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    asyncio.run(main())