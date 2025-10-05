#!/usr/bin/env python3
"""
小智ESP32服务器性能监控脚本
用于监控多设备并发时的系统资源使用情况
"""

import time
import psutil
import json
import asyncio
import websockets
import threading
from datetime import datetime
from collections import deque
import argparse

class PerformanceMonitor:
    def __init__(self, monitor_duration=300):
        self.monitor_duration = monitor_duration
        self.metrics_history = deque(maxlen=1000)
        self.is_monitoring = False
        
    def get_system_metrics(self):
        """获取系统指标"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # 获取进程信息
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                if 'python' in proc.info['name'].lower() or 'xiaozhi' in proc.info['name'].lower():
                    processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count()
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used
            },
            'disk': {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': (disk.used / disk.total) * 100
            },
            'processes': processes
        }
    
    def start_monitoring(self):
        """开始监控"""
        self.is_monitoring = True
        print(f"🔍 开始性能监控，持续时间: {self.monitor_duration}秒")
        print("=" * 80)
        
        start_time = time.time()
        while self.is_monitoring and (time.time() - start_time) < self.monitor_duration:
            metrics = self.get_system_metrics()
            self.metrics_history.append(metrics)
            
            # 实时显示关键指标
            self.display_realtime_metrics(metrics)
            
            time.sleep(5)  # 每5秒采集一次
        
        self.is_monitoring = False
        print("\n📊 监控完成，生成报告...")
        self.generate_report()
    
    def display_realtime_metrics(self, metrics):
        """实时显示指标"""
        timestamp = metrics['timestamp'].split('T')[1][:8]
        cpu_percent = metrics['cpu']['percent']
        memory_percent = metrics['memory']['percent']
        
        # 获取Python进程的资源使用
        python_cpu = sum(p['cpu_percent'] or 0 for p in metrics['processes'])
        python_memory = sum(p['memory_percent'] or 0 for p in metrics['processes'])
        
        status_cpu = "🔴" if cpu_percent > 80 else "🟡" if cpu_percent > 60 else "🟢"
        status_mem = "🔴" if memory_percent > 80 else "🟡" if memory_percent > 60 else "🟢"
        
        print(f"[{timestamp}] {status_cpu} CPU: {cpu_percent:5.1f}% | "
              f"{status_mem} 内存: {memory_percent:5.1f}% | "
              f"Python进程 CPU: {python_cpu:5.1f}% 内存: {python_memory:5.1f}%")
    
    def generate_report(self):
        """生成性能报告"""
        if not self.metrics_history:
            print("❌ 没有收集到监控数据")
            return
        
        # 计算统计信息
        cpu_values = [m['cpu']['percent'] for m in self.metrics_history]
        memory_values = [m['memory']['percent'] for m in self.metrics_history]
        
        report = {
            'monitoring_duration': len(self.metrics_history) * 5,
            'sample_count': len(self.metrics_history),
            'cpu_stats': {
                'avg': sum(cpu_values) / len(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values)
            },
            'memory_stats': {
                'avg': sum(memory_values) / len(memory_values),
                'max': max(memory_values),
                'min': min(memory_values)
            },
            'peak_usage_times': []
        }
        
        # 找出高负载时间点
        for metrics in self.metrics_history:
            if metrics['cpu']['percent'] > 80 or metrics['memory']['percent'] > 80:
                report['peak_usage_times'].append({
                    'timestamp': metrics['timestamp'],
                    'cpu': metrics['cpu']['percent'],
                    'memory': metrics['memory']['percent']
                })
        
        # 保存详细报告
        report_file = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'report': report,
                'raw_data': list(self.metrics_history)
            }, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print("\n" + "=" * 80)
        print("📈 性能监控报告摘要")
        print("=" * 80)
        print(f"监控时长: {report['monitoring_duration']}秒")
        print(f"采样次数: {report['sample_count']}")
        print(f"CPU使用率 - 平均: {report['cpu_stats']['avg']:.1f}% | "
              f"最高: {report['cpu_stats']['max']:.1f}% | "
              f"最低: {report['cpu_stats']['min']:.1f}%")
        print(f"内存使用率 - 平均: {report['memory_stats']['avg']:.1f}% | "
              f"最高: {report['memory_stats']['max']:.1f}% | "
              f"最低: {report['memory_stats']['min']:.1f}%")
        
        if report['peak_usage_times']:
            print(f"\n⚠️ 检测到 {len(report['peak_usage_times'])} 次高负载时段")
            for peak in report['peak_usage_times'][:5]:  # 显示前5次
                time_str = peak['timestamp'].split('T')[1][:8]
                print(f"  [{time_str}] CPU: {peak['cpu']:.1f}% 内存: {peak['memory']:.1f}%")
        
        print(f"\n📄 详细报告已保存至: {report_file}")
        
        # 给出优化建议
        self.provide_optimization_suggestions(report)
    
    def provide_optimization_suggestions(self, report):
        """提供优化建议"""
        print("\n" + "=" * 80)
        print("💡 优化建议")
        print("=" * 80)
        
        if report['cpu_stats']['avg'] > 70:
            print("🔴 CPU使用率过高建议:")
            print("  1. 减少并发连接数限制 (max_concurrent_sessions: 1-2)")
            print("  2. 增加Docker CPU限制 (cpus: '2.0')")
            print("  3. 优化SenseVoice模型推理频率")
        
        if report['memory_stats']['avg'] > 70:
            print("🔴 内存使用率过高建议:")
            print("  1. 增加Docker内存限制 (memory: 8G)")
            print("  2. 启用模型实例复用")
            print("  3. 优化音频缓存策略")
        
        if len(report['peak_usage_times']) > 10:
            print("🔴 频繁高负载建议:")
            print("  1. 实施连接限流 (max_concurrent_per_ip: 1)")
            print("  2. 增加音频处理队列")
            print("  3. 考虑使用更轻量的ASR模型")
        
        print("\n✅ 应用优化配置:")
        print("  1. 使用 docker-compose_optimized.yml")
        print("  2. 替换为 .config_optimized.yaml")
        print("  3. 部署 sensevoice_optimized.py")

class ConcurrencyTester:
    """并发测试器"""
    
    def __init__(self, server_url="ws://localhost:8000/xiaozhi/v1/"):
        self.server_url = server_url
        self.test_results = []
    
    async def test_single_connection(self, session_id):
        """测试单个连接"""
        try:
            async with websockets.connect(self.server_url) as websocket:
                # 发送初始化消息
                init_msg = {
                    "type": "init",
                    "session_id": session_id,
                    "user_id": f"test_user_{session_id}"
                }
                await websocket.send(json.dumps(init_msg))
                
                # 模拟音频数据发送
                start_time = time.time()
                for i in range(5):  # 发送5次音频数据
                    audio_msg = {
                        "type": "audio_chunk",
                        "data": "fake_audio_data",
                        "chunk_id": i
                    }
                    await websocket.send(json.dumps(audio_msg))
                    await asyncio.sleep(0.5)
                
                # 等待响应
                response_count = 0
                while response_count < 3 and time.time() - start_time < 30:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5)
                        response_count += 1
                    except asyncio.TimeoutError:
                        break
                
                end_time = time.time()
                return {
                    'session_id': session_id,
                    'duration': end_time - start_time,
                    'responses': response_count,
                    'success': response_count > 0
                }
                
        except Exception as e:
            return {
                'session_id': session_id,
                'duration': 0,
                'responses': 0,
                'success': False,
                'error': str(e)
            }
    
    async def run_concurrent_test(self, num_connections=3):
        """运行并发测试"""
        print(f"🧪 开始并发测试，连接数: {num_connections}")
        
        tasks = []
        for i in range(num_connections):
            task = asyncio.create_task(self.test_single_connection(f"test_session_{i}"))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 分析结果
        successful = [r for r in results if isinstance(r, dict) and r.get('success')]
        failed = [r for r in results if not (isinstance(r, dict) and r.get('success'))]
        
        print(f"✅ 成功连接: {len(successful)}/{num_connections}")
        print(f"❌ 失败连接: {len(failed)}")
        
        if successful:
            avg_duration = sum(r['duration'] for r in successful) / len(successful)
            print(f"⏱️ 平均响应时间: {avg_duration:.2f}秒")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='小智ESP32服务器性能监控工具')
    parser.add_argument('--mode', choices=['monitor', 'test', 'both'], default='both',
                       help='运行模式: monitor(监控), test(测试), both(两者)')
    parser.add_argument('--duration', type=int, default=300,
                       help='监控持续时间(秒), 默认300秒')
    parser.add_argument('--connections', type=int, default=3,
                       help='并发测试连接数, 默认3个')
    
    args = parser.parse_args()
    
    if args.mode in ['monitor', 'both']:
        monitor = PerformanceMonitor(args.duration)
        monitor_thread = threading.Thread(target=monitor.start_monitoring)
        monitor_thread.start()
        
        if args.mode == 'both':
            # 等待监控启动
            time.sleep(10)
            
            # 运行并发测试
            async def run_test():
                tester = ConcurrencyTester()
                await tester.run_concurrent_test(args.connections)
            
            asyncio.run(run_test())
        
        monitor_thread.join()
    
    elif args.mode == 'test':
        async def run_test():
            tester = ConcurrencyTester()
            await tester.run_concurrent_test(args.connections)
        
        asyncio.run(run_test())

if __name__ == "__main__":
    main()