#!/usr/bin/env python3
"""
并发设备性能测试工具
测试不同设备数量下的ASR性能表现，重点关注首字延迟
"""

import asyncio
import time
import json
import os
import random
import statistics
from typing import List, Dict, Tuple
import aiohttp
from tabulate import tabulate
from datetime import datetime
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConcurrentDeviceTester:
    def __init__(self):
        self.base_url = "http://localhost:8001"  # ASR服务地址
        self.test_audio_files = self._load_test_audio_files()
        self.results = {}
        
    def _load_test_audio_files(self) -> List[bytes]:
        """加载测试音频文件"""
        audio_root = os.path.join(os.getcwd(), "config", "assets")
        test_files = []
        
        if os.path.exists(audio_root):
            for file_name in os.listdir(audio_root):
                if file_name.endswith(('.wav', '.pcm')):
                    file_path = os.path.join(audio_root, file_name)
                    if os.path.getsize(file_path) > 50 * 1024:  # 至少50KB
                        with open(file_path, 'rb') as f:
                            test_files.append(f.read())
        
        # 如果没有找到音频文件，生成模拟数据
        if not test_files:
            logger.warning("未找到测试音频文件，使用模拟数据")
            for i in range(5):
                # 生成模拟音频数据（实际应该是真实的音频文件）
                test_files.append(b"fake_audio_data_" + str(i).encode() * 1000)
        
        return test_files
    
    async def _simulate_device_request(self, device_id: int, session: aiohttp.ClientSession) -> Dict:
        """模拟单个设备的ASR请求"""
        audio_data = random.choice(self.test_audio_files)
        
        # 记录开始时间
        start_time = time.time()
        first_response_time = None
        total_time = None
        success = False
        error_msg = None
        
        try:
            # 准备ASR请求数据，匹配simple_asr_test_service的API格式
            import base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            request_data = {
                "session_id": f"device_{device_id}_{int(time.time())}",
                "audio_data": audio_base64,
                "sample_rate": 16000,
                "language": "zh",
                "priority": 2,
                "timestamp": time.time()
            }
            
            async with session.post(
                f"{self.base_url}/asr/recognize", 
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    # 记录首次响应时间（首字延迟）
                    first_response_time = time.time() - start_time
                    
                    result = await response.json()
                    total_time = time.time() - start_time
                    success = True
                    
                    return {
                        'device_id': device_id,
                        'success': success,
                        'first_response_time': first_response_time,
                        'total_time': total_time,
                        'result': result.get('text', ''),
                        'error': None
                    }
                else:
                    error_msg = f"HTTP {response.status}: {await response.text()}"
                    
        except asyncio.TimeoutError:
            error_msg = "请求超时"
        except Exception as e:
            error_msg = str(e)
        
        return {
            'device_id': device_id,
            'success': success,
            'first_response_time': first_response_time,
            'total_time': total_time,
            'result': '',
            'error': error_msg
        }
    
    async def _test_concurrent_devices(self, device_count: int, test_duration: int = 60) -> Dict:
        """测试指定数量设备的并发性能"""
        logger.info(f"开始测试 {device_count} 台设备并发性能，测试时长 {test_duration} 秒")
        
        results = []
        start_test_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - start_test_time < test_duration:
                # 创建并发任务
                tasks = []
                for device_id in range(device_count):
                    task = self._simulate_device_request(device_id, session)
                    tasks.append(task)
                
                # 执行并发请求
                batch_start = time.time()
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                batch_duration = time.time() - batch_start
                
                # 处理结果
                for result in batch_results:
                    if isinstance(result, dict):
                        result['batch_duration'] = batch_duration
                        results.append(result)
                    else:
                        logger.error(f"任务执行异常: {result}")
                
                # 短暂休息避免过度压力
                await asyncio.sleep(0.1)
        
        return self._analyze_results(device_count, results)
    
    def _analyze_results(self, device_count: int, results: List[Dict]) -> Dict:
        """分析测试结果"""
        if not results:
            return {
                'device_count': device_count,
                'total_requests': 0,
                'success_rate': 0,
                'avg_first_response_time': 0,
                'avg_total_time': 0,
                'p95_first_response_time': 0,
                'p99_first_response_time': 0,
                'errors': []
            }
        
        successful_results = [r for r in results if r['success']]
        total_requests = len(results)
        success_count = len(successful_results)
        success_rate = (success_count / total_requests) * 100 if total_requests > 0 else 0
        
        # 计算延迟统计
        first_response_times = [r['first_response_time'] for r in successful_results if r['first_response_time'] is not None]
        total_times = [r['total_time'] for r in successful_results if r['total_time'] is not None]
        
        avg_first_response_time = statistics.mean(first_response_times) if first_response_times else 0
        avg_total_time = statistics.mean(total_times) if total_times else 0
        
        p95_first_response_time = statistics.quantiles(first_response_times, n=20)[18] if len(first_response_times) >= 20 else (max(first_response_times) if first_response_times else 0)
        p99_first_response_time = statistics.quantiles(first_response_times, n=100)[98] if len(first_response_times) >= 100 else (max(first_response_times) if first_response_times else 0)
        
        # 收集错误信息
        errors = [r['error'] for r in results if r['error'] is not None]
        error_summary = {}
        for error in errors:
            error_summary[error] = error_summary.get(error, 0) + 1
        
        return {
            'device_count': device_count,
            'total_requests': total_requests,
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_first_response_time': avg_first_response_time,
            'avg_total_time': avg_total_time,
            'p95_first_response_time': p95_first_response_time,
            'p99_first_response_time': p99_first_response_time,
            'error_summary': error_summary,
            'throughput': success_count / 60 if success_count > 0 else 0  # 每秒成功请求数
        }
    
    def _print_results(self):
        """打印测试结果"""
        print("\n" + "=" * 80)
        print("🚀 并发设备性能测试报告")
        print("=" * 80)
        
        if not self.results:
            print("❌ 没有测试结果")
            return
        
        # 创建结果表格
        headers = [
            "设备数量", "总请求数", "成功率(%)", "首字延迟(ms)", 
            "总延迟(ms)", "P95延迟(ms)", "P99延迟(ms)", "吞吐量(req/s)", "状态"
        ]
        
        table_data = []
        for device_count in sorted(self.results.keys()):
            result = self.results[device_count]
            
            # 判断性能状态
            if result['success_rate'] >= 95 and result['avg_first_response_time'] < 1.0:
                status = "✅ 优秀"
            elif result['success_rate'] >= 90 and result['avg_first_response_time'] < 2.0:
                status = "🟡 良好"
            elif result['success_rate'] >= 80:
                status = "🟠 一般"
            else:
                status = "❌ 差"
            
            table_data.append([
                f"{device_count}台",
                result['total_requests'],
                f"{result['success_rate']:.1f}",
                f"{result['avg_first_response_time']*1000:.0f}",
                f"{result['avg_total_time']*1000:.0f}",
                f"{result['p95_first_response_time']*1000:.0f}",
                f"{result['p99_first_response_time']*1000:.0f}",
                f"{result['throughput']:.1f}",
                status
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # 打印详细分析
        print("\n📊 性能分析:")
        for device_count in sorted(self.results.keys()):
            result = self.results[device_count]
            print(f"\n{device_count}台设备:")
            print(f"  • 成功率: {result['success_rate']:.1f}% ({result['success_count']}/{result['total_requests']})")
            print(f"  • 首字延迟: 平均{result['avg_first_response_time']*1000:.0f}ms, P95={result['p95_first_response_time']*1000:.0f}ms")
            print(f"  • 吞吐量: {result['throughput']:.1f} 请求/秒")
            
            if result['error_summary']:
                print(f"  • 错误统计:")
                for error, count in result['error_summary'].items():
                    print(f"    - {error}: {count}次")
        
        # 性能建议
        print("\n💡 优化建议:")
        self._generate_optimization_suggestions()
    
    def _generate_optimization_suggestions(self):
        """生成优化建议"""
        if not self.results:
            return
        
        # 分析性能趋势
        device_counts = sorted(self.results.keys())
        
        for i, device_count in enumerate(device_counts):
            result = self.results[device_count]
            
            if result['avg_first_response_time'] > 2.0:  # 首字延迟超过2秒
                print(f"🔴 {device_count}台设备首字延迟严重 ({result['avg_first_response_time']*1000:.0f}ms):")
                print("   - 建议增加ASR_MAX_CONCURRENT参数")
                print("   - 建议优化ASR_BATCH_SIZE配置")
                print("   - 考虑启用ASR缓存和预处理优化")
                
            elif result['success_rate'] < 90:  # 成功率低于90%
                print(f"🟡 {device_count}台设备成功率偏低 ({result['success_rate']:.1f}%):")
                print("   - 检查服务器资源使用情况")
                print("   - 增加队列容量和超时时间")
                print("   - 考虑启用负载均衡")
        
        # 通用优化建议
        print("\n🎯 通用优化策略:")
        print("1. 首字延迟优化:")
        print("   - 启用ASR预热机制")
        print("   - 优化模型加载和初始化")
        print("   - 使用更快的推理引擎")
        print("2. 并发性能优化:")
        print("   - 调整ASR_MAX_CONCURRENT和ASR_BATCH_SIZE")
        print("   - 启用内存池和零拷贝优化")
        print("   - 配置合适的工作线程数")
        print("3. 系统资源优化:")
        print("   - 监控CPU和内存使用率")
        print("   - 优化Redis缓存配置")
        print("   - 考虑硬件升级或横向扩展")
    
    async def run_full_test(self):
        """运行完整的并发性能测试"""
        test_scenarios = [10, 20, 30]  # 测试10台、20台、30台设备
        
        print("🚀 开始并发设备性能测试")
        print(f"测试场景: {test_scenarios} 台设备")
        print("每个场景测试时长: 60秒")
        print("-" * 50)
        
        for device_count in test_scenarios:
            print(f"\n📱 正在测试 {device_count} 台设备...")
            result = await self._test_concurrent_devices(device_count, test_duration=60)
            self.results[device_count] = result
            
            # 实时显示结果
            print(f"✅ {device_count}台设备测试完成:")
            print(f"   成功率: {result['success_rate']:.1f}%")
            print(f"   首字延迟: {result['avg_first_response_time']*1000:.0f}ms")
            print(f"   吞吐量: {result['throughput']:.1f} req/s")
        
        # 打印完整报告
        self._print_results()

async def main():
    """主函数"""
    tester = ConcurrentDeviceTester()
    await tester.run_full_test()

if __name__ == "__main__":
    asyncio.run(main())