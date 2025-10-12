#!/usr/bin/env python3
"""
真实场景压力测试工具
模拟持续负载、资源竞争和真实使用模式
"""

import asyncio
import time
import json
import os
import random
import statistics
import psutil
import threading
from typing import List, Dict, Tuple
import aiohttp
from tabulate import tabulate
from datetime import datetime
import logging

# 设置日志级别
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticStressTester:
    def __init__(self):
        self.base_url = "http://localhost:8001"
        self.test_audio_files = self._load_test_audio_files()
        self.results = {}
        self.system_stats = []
        self.monitoring_active = False
        
    def _load_test_audio_files(self) -> List[bytes]:
        """加载测试音频文件"""
        audio_root = os.path.join(os.getcwd(), "config", "assets")
        test_files = []
        
        if os.path.exists(audio_root):
            for file_name in os.listdir(audio_root):
                if file_name.endswith(('.wav', '.pcm')):
                    file_path = os.path.join(audio_root, file_name)
                    if os.path.getsize(file_path) > 50 * 1024:
                        with open(file_path, 'rb') as f:
                            test_files.append(f.read())
        
        if not test_files:
            logger.warning("未找到测试音频文件，使用模拟数据")
            for i in range(5):
                test_files.append(b"fake_audio_data_" + str(i).encode() * 2000)
        
        return test_files
    
    def _start_system_monitoring(self):
        """启动系统资源监控"""
        self.monitoring_active = True
        
        def monitor():
            while self.monitoring_active:
                try:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    # 获取ASR服务进程信息
                    asr_process_info = None
                    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                        try:
                            if 'python' in proc.info['name'].lower() and any(
                                'asr' in cmd.lower() for cmd in proc.cmdline()
                            ):
                                asr_process_info = {
                                    'pid': proc.info['pid'],
                                    'cpu_percent': proc.cpu_percent(),
                                    'memory_percent': proc.memory_percent()
                                }
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    self.system_stats.append({
                        'timestamp': time.time(),
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory.percent,
                        'memory_available_gb': memory.available / (1024**3),
                        'asr_process': asr_process_info
                    })
                    
                except Exception as e:
                    logger.error(f"监控错误: {e}")
                
                time.sleep(2)  # 每2秒监控一次
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _stop_system_monitoring(self):
        """停止系统资源监控"""
        self.monitoring_active = False
    
    async def _simulate_realistic_device(self, device_id: int, session: aiohttp.ClientSession, 
                                       test_duration: int, request_interval: float) -> List[Dict]:
        """模拟真实设备的使用模式"""
        device_results = []
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < test_duration:
            request_start = time.time()
            
            try:
                # 模拟真实的音频数据
                audio_data = random.choice(self.test_audio_files)
                
                import base64
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                
                request_data = {
                    "session_id": f"device_{device_id}_req_{request_count}",
                    "audio_data": audio_base64,
                    "sample_rate": 16000,
                    "language": "zh",
                    "priority": random.choice([1, 2, 3]),  # 随机优先级
                    "timestamp": time.time()
                }
                
                async with session.post(
                    f"{self.base_url}/asr/recognize",
                    json=request_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=aiohttp.ClientTimeout(total=10)  # 更短的超时时间
                ) as response:
                    
                    response_time = time.time() - request_start
                    
                    if response.status == 200:
                        result = await response.json()
                        device_results.append({
                            'device_id': device_id,
                            'request_id': request_count,
                            'success': True,
                            'response_time': response_time,
                            'timestamp': request_start,
                            'result': result.get('text', ''),
                            'error': None
                        })
                    else:
                        error_text = await response.text()
                        device_results.append({
                            'device_id': device_id,
                            'request_id': request_count,
                            'success': False,
                            'response_time': response_time,
                            'timestamp': request_start,
                            'result': '',
                            'error': f"HTTP {response.status}: {error_text}"
                        })
                        
            except asyncio.TimeoutError:
                device_results.append({
                    'device_id': device_id,
                    'request_id': request_count,
                    'success': False,
                    'response_time': time.time() - request_start,
                    'timestamp': request_start,
                    'result': '',
                    'error': "请求超时"
                })
            except Exception as e:
                device_results.append({
                    'device_id': device_id,
                    'request_id': request_count,
                    'success': False,
                    'response_time': time.time() - request_start,
                    'timestamp': request_start,
                    'result': '',
                    'error': str(e)
                })
            
            request_count += 1
            
            # 模拟真实的请求间隔（带随机性）
            actual_interval = request_interval + random.uniform(-0.1, 0.1)
            if actual_interval > 0:
                await asyncio.sleep(actual_interval)
        
        return device_results
    
    async def _test_realistic_scenario(self, device_count: int, test_duration: int = 300) -> Dict:
        """测试真实场景下的性能"""
        logger.info(f"开始真实场景测试: {device_count}台设备，持续{test_duration}秒")
        
        # 启动系统监控
        self._start_system_monitoring()
        
        # 模拟真实的请求频率：每个设备每2-5秒发送一次请求
        request_interval = random.uniform(2.0, 5.0)
        
        all_results = []
        
        async with aiohttp.ClientSession() as session:
            # 创建设备任务
            tasks = []
            for device_id in range(device_count):
                task = self._simulate_realistic_device(
                    device_id, session, test_duration, request_interval
                )
                tasks.append(task)
            
            # 执行所有设备的并发测试
            device_results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 收集所有结果
            for device_results in device_results_list:
                if isinstance(device_results, list):
                    all_results.extend(device_results)
                else:
                    logger.error(f"设备任务异常: {device_results}")
        
        # 停止系统监控
        self._stop_system_monitoring()
        
        return self._analyze_realistic_results(device_count, all_results)
    
    def _analyze_realistic_results(self, device_count: int, results: List[Dict]) -> Dict:
        """分析真实场景测试结果"""
        if not results:
            return {
                'device_count': device_count,
                'total_requests': 0,
                'success_rate': 0,
                'avg_response_time': 0,
                'p95_response_time': 0,
                'p99_response_time': 0,
                'timeout_rate': 0,
                'error_summary': {},
                'system_performance': {}
            }
        
        successful_results = [r for r in results if r['success']]
        total_requests = len(results)
        success_count = len(successful_results)
        success_rate = (success_count / total_requests) * 100 if total_requests > 0 else 0
        
        # 计算响应时间统计
        response_times = [r['response_time'] for r in successful_results]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else (max(response_times) if response_times else 0)
        p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else (max(response_times) if response_times else 0)
        
        # 计算超时率
        timeout_count = len([r for r in results if r['error'] and '超时' in r['error']])
        timeout_rate = (timeout_count / total_requests) * 100 if total_requests > 0 else 0
        
        # 错误统计
        errors = [r['error'] for r in results if r['error'] is not None]
        error_summary = {}
        for error in errors:
            error_key = error[:50] + "..." if len(error) > 50 else error
            error_summary[error_key] = error_summary.get(error_key, 0) + 1
        
        # 系统性能分析
        system_performance = self._analyze_system_performance()
        
        return {
            'device_count': device_count,
            'total_requests': total_requests,
            'success_count': success_count,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time,
            'timeout_rate': timeout_rate,
            'error_summary': error_summary,
            'system_performance': system_performance
        }
    
    def _analyze_system_performance(self) -> Dict:
        """分析系统性能数据"""
        if not self.system_stats:
            return {}
        
        cpu_usage = [stat['cpu_percent'] for stat in self.system_stats]
        memory_usage = [stat['memory_percent'] for stat in self.system_stats]
        
        asr_cpu_usage = []
        asr_memory_usage = []
        for stat in self.system_stats:
            if stat['asr_process']:
                asr_cpu_usage.append(stat['asr_process']['cpu_percent'])
                asr_memory_usage.append(stat['asr_process']['memory_percent'])
        
        return {
            'avg_cpu_usage': statistics.mean(cpu_usage),
            'max_cpu_usage': max(cpu_usage),
            'avg_memory_usage': statistics.mean(memory_usage),
            'max_memory_usage': max(memory_usage),
            'avg_asr_cpu_usage': statistics.mean(asr_cpu_usage) if asr_cpu_usage else 0,
            'max_asr_cpu_usage': max(asr_cpu_usage) if asr_cpu_usage else 0,
            'avg_asr_memory_usage': statistics.mean(asr_memory_usage) if asr_memory_usage else 0,
            'max_asr_memory_usage': max(asr_memory_usage) if asr_memory_usage else 0
        }
    
    def _print_realistic_results(self):
        """打印真实场景测试结果"""
        print("\n" + "=" * 80)
        print("🔥 真实场景压力测试报告")
        print("=" * 80)
        
        if not self.results:
            print("❌ 没有测试结果")
            return
        
        # 创建结果表格
        headers = [
            "设备数量", "总请求数", "成功率(%)", "平均响应时间(ms)", 
            "P95响应时间(ms)", "P99响应时间(ms)", "超时率(%)", "状态"
        ]
        
        table_data = []
        for device_count in sorted(self.results.keys()):
            result = self.results[device_count]
            
            # 判断性能状态
            if result['success_rate'] >= 95 and result['avg_response_time'] < 1.0 and result['timeout_rate'] < 5:
                status = "✅ 优秀"
            elif result['success_rate'] >= 90 and result['avg_response_time'] < 2.0 and result['timeout_rate'] < 10:
                status = "🟡 良好"
            elif result['success_rate'] >= 80 and result['timeout_rate'] < 20:
                status = "🟠 一般"
            else:
                status = "❌ 差"
            
            table_data.append([
                f"{device_count}台",
                result['total_requests'],
                f"{result['success_rate']:.1f}",
                f"{result['avg_response_time']*1000:.0f}",
                f"{result['p95_response_time']*1000:.0f}",
                f"{result['p99_response_time']*1000:.0f}",
                f"{result['timeout_rate']:.1f}",
                status
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # 打印详细分析
        print("\n📊 详细性能分析:")
        for device_count in sorted(self.results.keys()):
            result = self.results[device_count]
            sys_perf = result['system_performance']
            
            print(f"\n{device_count}台设备:")
            print(f"  📈 请求统计:")
            print(f"    • 总请求数: {result['total_requests']}")
            print(f"    • 成功率: {result['success_rate']:.1f}% ({result['success_count']}/{result['total_requests']})")
            print(f"    • 超时率: {result['timeout_rate']:.1f}%")
            
            print(f"  ⏱️ 响应时间:")
            print(f"    • 平均: {result['avg_response_time']*1000:.0f}ms")
            print(f"    • P95: {result['p95_response_time']*1000:.0f}ms")
            print(f"    • P99: {result['p99_response_time']*1000:.0f}ms")
            
            if sys_perf:
                print(f"  💻 系统资源:")
                print(f"    • CPU使用率: 平均{sys_perf['avg_cpu_usage']:.1f}%, 峰值{sys_perf['max_cpu_usage']:.1f}%")
                print(f"    • 内存使用率: 平均{sys_perf['avg_memory_usage']:.1f}%, 峰值{sys_perf['max_memory_usage']:.1f}%")
                if sys_perf['avg_asr_cpu_usage'] > 0:
                    print(f"    • ASR进程CPU: 平均{sys_perf['avg_asr_cpu_usage']:.1f}%, 峰值{sys_perf['max_asr_cpu_usage']:.1f}%")
                    print(f"    • ASR进程内存: 平均{sys_perf['avg_asr_memory_usage']:.1f}%, 峰值{sys_perf['max_asr_memory_usage']:.1f}%")
            
            if result['error_summary']:
                print(f"  ❌ 错误统计:")
                for error, count in list(result['error_summary'].items())[:3]:  # 只显示前3个错误
                    print(f"    • {error}: {count}次")
        
        # 性能建议
        print("\n💡 针对性优化建议:")
        self._generate_realistic_optimization_suggestions()
    
    def _generate_realistic_optimization_suggestions(self):
        """生成基于真实测试的优化建议"""
        if not self.results:
            return
        
        for device_count in sorted(self.results.keys()):
            result = self.results[device_count]
            sys_perf = result['system_performance']
            
            print(f"\n🎯 {device_count}台设备优化建议:")
            
            # 响应时间优化
            if result['avg_response_time'] > 2.0:
                print("  🔴 响应时间过长:")
                print("    - 增加ASR_MAX_CONCURRENT参数")
                print("    - 优化ASR_BATCH_SIZE配置")
                print("    - 启用ASR缓存机制")
            
            # 超时率优化
            if result['timeout_rate'] > 10:
                print("  🟡 超时率偏高:")
                print("    - 增加请求超时时间")
                print("    - 优化队列管理策略")
                print("    - 检查网络延迟")
            
            # 系统资源优化
            if sys_perf and sys_perf['max_cpu_usage'] > 80:
                print("  ⚠️ CPU使用率过高:")
                print("    - 减少并发数量")
                print("    - 优化算法效率")
                print("    - 考虑CPU升级")
            
            if sys_perf and sys_perf['max_memory_usage'] > 85:
                print("  ⚠️ 内存使用率过高:")
                print("    - 减少缓存大小")
                print("    - 优化内存管理")
                print("    - 考虑内存升级")
    
    async def run_realistic_test(self):
        """运行真实场景测试"""
        test_scenarios = [10, 20, 30]
        test_duration = 180  # 3分钟测试
        
        print("🔥 开始真实场景压力测试")
        print(f"测试场景: {test_scenarios} 台设备")
        print(f"每个场景测试时长: {test_duration}秒")
        print("模拟真实使用模式: 每设备2-5秒间隔发送请求")
        print("-" * 50)
        
        for device_count in test_scenarios:
            print(f"\n📱 正在测试 {device_count} 台设备...")
            result = await self._test_realistic_scenario(device_count, test_duration)
            self.results[device_count] = result
            
            # 实时显示结果
            print(f"✅ {device_count}台设备测试完成:")
            print(f"   成功率: {result['success_rate']:.1f}%")
            print(f"   平均响应时间: {result['avg_response_time']*1000:.0f}ms")
            print(f"   超时率: {result['timeout_rate']:.1f}%")
        
        # 打印完整报告
        self._print_realistic_results()

async def main():
    """主函数"""
    tester = RealisticStressTester()
    await tester.run_realistic_test()

if __name__ == "__main__":
    asyncio.run(main())