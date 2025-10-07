#!/usr/bin/env python3
"""
Xiaozhi ESP32 Server - 性能测试脚本
模拟多设备并发访问，测试系统性能和稳定性
"""

import asyncio
import aiohttp
import time
import json
import random
import argparse
import logging
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import websockets
import base64
import wave
import io

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """测试配置"""
    base_url: str = "http://localhost:8080"
    websocket_url: str = "ws://localhost:8080/ws"
    concurrent_devices: int = 10
    test_duration: int = 300  # 5分钟
    request_interval: float = 2.0  # 每2秒一次请求
    audio_file: str = "test_audio.wav"
    
@dataclass
class TestResult:
    """测试结果"""
    device_id: str
    request_type: str
    start_time: float
    end_time: float
    response_time: float
    status_code: int
    success: bool
    error_message: str = ""
    response_size: int = 0

class AudioGenerator:
    """音频生成器"""
    
    @staticmethod
    def generate_test_audio(duration: float = 3.0, sample_rate: int = 16000) -> bytes:
        """生成测试音频数据"""
        # 生成正弦波音频
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A4音符
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # 转换为16位PCM
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # 创建WAV文件
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        return buffer.getvalue()

class DeviceSimulator:
    """设备模拟器"""
    
    def __init__(self, device_id: str, config: TestConfig):
        self.device_id = device_id
        self.config = config
        self.session = None
        self.results: List[TestResult] = []
        self.is_running = False
        
    async def start(self):
        """启动设备模拟"""
        self.is_running = True
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=10)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        
        logger.info(f"设备 {self.device_id} 开始测试")
        
        # 启动不同类型的测试任务
        tasks = [
            self.run_voice_chat_test(),
            self.run_health_check_test(),
            self.run_websocket_test()
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def stop(self):
        """停止设备模拟"""
        self.is_running = False
        if self.session:
            await self.session.close()
        logger.info(f"设备 {self.device_id} 测试结束")
        
    async def run_voice_chat_test(self):
        """语音对话测试"""
        while self.is_running:
            try:
                # 生成测试音频
                audio_data = AudioGenerator.generate_test_audio()
                
                # VAD检测
                await self.test_vad(audio_data)
                await asyncio.sleep(0.5)
                
                # ASR识别
                text = await self.test_asr(audio_data)
                await asyncio.sleep(0.5)
                
                # LLM对话
                if text:
                    response_text = await self.test_llm(text)
                    await asyncio.sleep(0.5)
                    
                    # TTS合成
                    if response_text:
                        await self.test_tts(response_text)
                
                # 等待下次请求
                await asyncio.sleep(self.config.request_interval)
                
            except Exception as e:
                logger.error(f"设备 {self.device_id} 语音对话测试错误: {e}")
                await asyncio.sleep(1)
                
    async def run_health_check_test(self):
        """健康检查测试"""
        while self.is_running:
            try:
                await self.test_health_check()
                await asyncio.sleep(10)  # 每10秒检查一次
            except Exception as e:
                logger.error(f"设备 {self.device_id} 健康检查错误: {e}")
                await asyncio.sleep(5)
                
    async def run_websocket_test(self):
        """WebSocket连接测试"""
        try:
            uri = f"{self.config.websocket_url}/{self.device_id}"
            async with websockets.connect(uri) as websocket:
                logger.info(f"设备 {self.device_id} WebSocket连接建立")
                
                while self.is_running:
                    # 发送心跳
                    message = {
                        "type": "heartbeat",
                        "device_id": self.device_id,
                        "timestamp": time.time()
                    }
                    await websocket.send(json.dumps(message))
                    
                    # 接收响应
                    try:
                        response = await asyncio.wait_for(
                            websocket.recv(), timeout=5.0
                        )
                        logger.debug(f"设备 {self.device_id} 收到WebSocket响应: {response}")
                    except asyncio.TimeoutError:
                        logger.warning(f"设备 {self.device_id} WebSocket响应超时")
                    
                    await asyncio.sleep(30)  # 每30秒发送心跳
                    
        except Exception as e:
            logger.error(f"设备 {self.device_id} WebSocket连接错误: {e}")
            
    async def test_vad(self, audio_data: bytes) -> bool:
        """测试VAD服务"""
        start_time = time.time()
        
        try:
            # 编码音频数据
            audio_b64 = base64.b64encode(audio_data).decode()
            
            data = {
                "audio_data": audio_b64,
                "sample_rate": 16000,
                "device_id": self.device_id
            }
            
            async with self.session.post(
                f"{self.config.base_url}/api/v1/vad/detect",
                json=data
            ) as response:
                end_time = time.time()
                response_text = await response.text()
                
                result = TestResult(
                    device_id=self.device_id,
                    request_type="VAD",
                    start_time=start_time,
                    end_time=end_time,
                    response_time=end_time - start_time,
                    status_code=response.status,
                    success=response.status == 200,
                    response_size=len(response_text)
                )
                
                if response.status != 200:
                    result.error_message = response_text
                    
                self.results.append(result)
                
                if response.status == 200:
                    response_data = json.loads(response_text)
                    return response_data.get("has_voice", False)
                    
        except Exception as e:
            end_time = time.time()
            result = TestResult(
                device_id=self.device_id,
                request_type="VAD",
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
            self.results.append(result)
            
        return False
        
    async def test_asr(self, audio_data: bytes) -> str:
        """测试ASR服务"""
        start_time = time.time()
        
        try:
            # 编码音频数据
            audio_b64 = base64.b64encode(audio_data).decode()
            
            data = {
                "audio_data": audio_b64,
                "language": "zh",
                "device_id": self.device_id
            }
            
            async with self.session.post(
                f"{self.config.base_url}/api/v1/asr/recognize",
                json=data
            ) as response:
                end_time = time.time()
                response_text = await response.text()
                
                result = TestResult(
                    device_id=self.device_id,
                    request_type="ASR",
                    start_time=start_time,
                    end_time=end_time,
                    response_time=end_time - start_time,
                    status_code=response.status,
                    success=response.status == 200,
                    response_size=len(response_text)
                )
                
                if response.status != 200:
                    result.error_message = response_text
                    
                self.results.append(result)
                
                if response.status == 200:
                    response_data = json.loads(response_text)
                    return response_data.get("text", "")
                    
        except Exception as e:
            end_time = time.time()
            result = TestResult(
                device_id=self.device_id,
                request_type="ASR",
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
            self.results.append(result)
            
        return ""
        
    async def test_llm(self, text: str) -> str:
        """测试LLM服务"""
        start_time = time.time()
        
        try:
            data = {
                "message": text or "你好",
                "device_id": self.device_id,
                "conversation_id": f"conv_{self.device_id}",
                "model": "qwen-7b-chat"
            }
            
            async with self.session.post(
                f"{self.config.base_url}/api/v1/llm/chat",
                json=data
            ) as response:
                end_time = time.time()
                response_text = await response.text()
                
                result = TestResult(
                    device_id=self.device_id,
                    request_type="LLM",
                    start_time=start_time,
                    end_time=end_time,
                    response_time=end_time - start_time,
                    status_code=response.status,
                    success=response.status == 200,
                    response_size=len(response_text)
                )
                
                if response.status != 200:
                    result.error_message = response_text
                    
                self.results.append(result)
                
                if response.status == 200:
                    response_data = json.loads(response_text)
                    return response_data.get("response", "")
                    
        except Exception as e:
            end_time = time.time()
            result = TestResult(
                device_id=self.device_id,
                request_type="LLM",
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
            self.results.append(result)
            
        return ""
        
    async def test_tts(self, text: str) -> bool:
        """测试TTS服务"""
        start_time = time.time()
        
        try:
            data = {
                "text": text or "你好",
                "voice": "zh-CN-XiaoxiaoNeural",
                "device_id": self.device_id,
                "format": "opus"
            }
            
            async with self.session.post(
                f"{self.config.base_url}/api/v1/tts/synthesize",
                json=data
            ) as response:
                end_time = time.time()
                response_data = await response.read()
                
                result = TestResult(
                    device_id=self.device_id,
                    request_type="TTS",
                    start_time=start_time,
                    end_time=end_time,
                    response_time=end_time - start_time,
                    status_code=response.status,
                    success=response.status == 200,
                    response_size=len(response_data)
                )
                
                if response.status != 200:
                    result.error_message = await response.text()
                    
                self.results.append(result)
                return response.status == 200
                
        except Exception as e:
            end_time = time.time()
            result = TestResult(
                device_id=self.device_id,
                request_type="TTS",
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
            self.results.append(result)
            
        return False
        
    async def test_health_check(self) -> bool:
        """测试健康检查"""
        start_time = time.time()
        
        try:
            async with self.session.get(
                f"{self.config.base_url}/health"
            ) as response:
                end_time = time.time()
                response_text = await response.text()
                
                result = TestResult(
                    device_id=self.device_id,
                    request_type="HEALTH",
                    start_time=start_time,
                    end_time=end_time,
                    response_time=end_time - start_time,
                    status_code=response.status,
                    success=response.status == 200,
                    response_size=len(response_text)
                )
                
                if response.status != 200:
                    result.error_message = response_text
                    
                self.results.append(result)
                return response.status == 200
                
        except Exception as e:
            end_time = time.time()
            result = TestResult(
                device_id=self.device_id,
                request_type="HEALTH",
                start_time=start_time,
                end_time=end_time,
                response_time=end_time - start_time,
                status_code=0,
                success=False,
                error_message=str(e)
            )
            self.results.append(result)
            
        return False

class PerformanceAnalyzer:
    """性能分析器"""
    
    @staticmethod
    def analyze_results(all_results: List[TestResult]) -> Dict[str, Any]:
        """分析测试结果"""
        if not all_results:
            return {}
            
        # 按请求类型分组
        results_by_type = {}
        for result in all_results:
            if result.request_type not in results_by_type:
                results_by_type[result.request_type] = []
            results_by_type[result.request_type].append(result)
        
        analysis = {}
        
        for request_type, results in results_by_type.items():
            response_times = [r.response_time for r in results if r.success]
            success_count = sum(1 for r in results if r.success)
            total_count = len(results)
            
            if response_times:
                analysis[request_type] = {
                    "total_requests": total_count,
                    "successful_requests": success_count,
                    "failed_requests": total_count - success_count,
                    "success_rate": success_count / total_count * 100,
                    "avg_response_time": np.mean(response_times),
                    "min_response_time": np.min(response_times),
                    "max_response_time": np.max(response_times),
                    "p50_response_time": np.percentile(response_times, 50),
                    "p95_response_time": np.percentile(response_times, 95),
                    "p99_response_time": np.percentile(response_times, 99),
                    "qps": success_count / (max(r.end_time for r in results) - min(r.start_time for r in results))
                }
            else:
                analysis[request_type] = {
                    "total_requests": total_count,
                    "successful_requests": 0,
                    "failed_requests": total_count,
                    "success_rate": 0,
                    "avg_response_time": 0,
                    "qps": 0
                }
        
        return analysis
    
    @staticmethod
    def generate_report(analysis: Dict[str, Any], config: TestConfig) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 60)
        report.append("Xiaozhi ESP32 Server 性能测试报告")
        report.append("=" * 60)
        report.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"并发设备数: {config.concurrent_devices}")
        report.append(f"测试时长: {config.test_duration}秒")
        report.append(f"请求间隔: {config.request_interval}秒")
        report.append("")
        
        for service_type, metrics in analysis.items():
            report.append(f"{service_type} 服务性能指标:")
            report.append("-" * 40)
            report.append(f"  总请求数: {metrics['total_requests']}")
            report.append(f"  成功请求数: {metrics['successful_requests']}")
            report.append(f"  失败请求数: {metrics['failed_requests']}")
            report.append(f"  成功率: {metrics['success_rate']:.2f}%")
            report.append(f"  平均响应时间: {metrics['avg_response_time']:.3f}s")
            report.append(f"  最小响应时间: {metrics.get('min_response_time', 0):.3f}s")
            report.append(f"  最大响应时间: {metrics.get('max_response_time', 0):.3f}s")
            report.append(f"  P50响应时间: {metrics.get('p50_response_time', 0):.3f}s")
            report.append(f"  P95响应时间: {metrics.get('p95_response_time', 0):.3f}s")
            report.append(f"  P99响应时间: {metrics.get('p99_response_time', 0):.3f}s")
            report.append(f"  QPS: {metrics['qps']:.2f}")
            report.append("")
        
        # 整体性能评估
        total_requests = sum(m['total_requests'] for m in analysis.values())
        total_successful = sum(m['successful_requests'] for m in analysis.values())
        overall_success_rate = total_successful / total_requests * 100 if total_requests > 0 else 0
        
        report.append("整体性能评估:")
        report.append("-" * 40)
        report.append(f"  总请求数: {total_requests}")
        report.append(f"  总成功数: {total_successful}")
        report.append(f"  整体成功率: {overall_success_rate:.2f}%")
        
        # 性能评级
        if overall_success_rate >= 95:
            grade = "优秀"
        elif overall_success_rate >= 90:
            grade = "良好"
        elif overall_success_rate >= 80:
            grade = "一般"
        else:
            grade = "需要优化"
            
        report.append(f"  性能评级: {grade}")
        report.append("")
        
        # 优化建议
        report.append("优化建议:")
        report.append("-" * 40)
        
        for service_type, metrics in analysis.items():
            if metrics['success_rate'] < 95:
                report.append(f"  {service_type}: 成功率偏低({metrics['success_rate']:.1f}%)，建议检查服务稳定性")
            if metrics.get('p95_response_time', 0) > 2.0:
                report.append(f"  {service_type}: P95响应时间过长({metrics.get('p95_response_time', 0):.2f}s)，建议优化性能")
            if metrics['qps'] < 10:
                report.append(f"  {service_type}: QPS偏低({metrics['qps']:.1f})，建议增加并发处理能力")
        
        return "\n".join(report)
    
    @staticmethod
    def plot_performance_charts(all_results: List[TestResult], output_dir: str = "test_results"):
        """绘制性能图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 按服务类型分组
        results_by_type = {}
        for result in all_results:
            if result.request_type not in results_by_type:
                results_by_type[result.request_type] = []
            results_by_type[result.request_type].append(result)
        
        # 响应时间分布图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Xiaozhi ESP32 Server 性能测试结果', fontsize=16)
        
        # 响应时间箱线图
        ax1 = axes[0, 0]
        response_times_data = []
        labels = []
        for service_type, results in results_by_type.items():
            response_times = [r.response_time for r in results if r.success]
            if response_times:
                response_times_data.append(response_times)
                labels.append(service_type)
        
        if response_times_data:
            ax1.boxplot(response_times_data, labels=labels)
            ax1.set_title('响应时间分布')
            ax1.set_ylabel('响应时间 (秒)')
            ax1.tick_params(axis='x', rotation=45)
        
        # 成功率柱状图
        ax2 = axes[0, 1]
        services = []
        success_rates = []
        for service_type, results in results_by_type.items():
            success_count = sum(1 for r in results if r.success)
            total_count = len(results)
            success_rate = success_count / total_count * 100 if total_count > 0 else 0
            services.append(service_type)
            success_rates.append(success_rate)
        
        if services:
            bars = ax2.bar(services, success_rates)
            ax2.set_title('服务成功率')
            ax2.set_ylabel('成功率 (%)')
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{rate:.1f}%', ha='center', va='bottom')
        
        # QPS趋势图
        ax3 = axes[1, 0]
        for service_type, results in results_by_type.items():
            successful_results = [r for r in results if r.success]
            if successful_results:
                # 按时间窗口计算QPS
                start_time = min(r.start_time for r in successful_results)
                end_time = max(r.end_time for r in successful_results)
                window_size = 30  # 30秒窗口
                
                times = []
                qps_values = []
                
                current_time = start_time
                while current_time < end_time:
                    window_end = current_time + window_size
                    window_requests = [
                        r for r in successful_results 
                        if current_time <= r.start_time < window_end
                    ]
                    qps = len(window_requests) / window_size
                    times.append(current_time - start_time)
                    qps_values.append(qps)
                    current_time += window_size
                
                ax3.plot(times, qps_values, label=service_type, marker='o')
        
        ax3.set_title('QPS趋势')
        ax3.set_xlabel('时间 (秒)')
        ax3.set_ylabel('QPS')
        ax3.legend()
        ax3.grid(True)
        
        # 错误分布饼图
        ax4 = axes[1, 1]
        error_counts = {}
        for result in all_results:
            if not result.success:
                error_type = f"{result.request_type}_ERROR"
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        if error_counts:
            ax4.pie(error_counts.values(), labels=error_counts.keys(), autopct='%1.1f%%')
            ax4.set_title('错误分布')
        else:
            ax4.text(0.5, 0.5, '无错误', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('错误分布')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"性能图表已保存到 {output_dir}/performance_charts.png")

class LoadTester:
    """负载测试器"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.devices: List[DeviceSimulator] = []
        self.all_results: List[TestResult] = []
        
    async def run_test(self):
        """运行负载测试"""
        logger.info(f"开始负载测试: {self.config.concurrent_devices}个并发设备")
        
        # 创建设备模拟器
        for i in range(self.config.concurrent_devices):
            device_id = f"device_{i:03d}"
            device = DeviceSimulator(device_id, self.config)
            self.devices.append(device)
        
        # 启动所有设备
        start_time = time.time()
        tasks = [device.start() for device in self.devices]
        
        try:
            # 运行指定时间
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.test_duration
            )
        except asyncio.TimeoutError:
            logger.info("测试时间到达，正在停止所有设备...")
        
        # 停止所有设备
        stop_tasks = [device.stop() for device in self.devices]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        # 收集所有结果
        for device in self.devices:
            self.all_results.extend(device.results)
        
        end_time = time.time()
        actual_duration = end_time - start_time
        
        logger.info(f"负载测试完成，实际运行时间: {actual_duration:.1f}秒")
        logger.info(f"总共收集到 {len(self.all_results)} 个测试结果")
        
        return self.all_results
    
    def save_results(self, output_dir: str = "test_results"):
        """保存测试结果"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始数据
        results_data = [asdict(result) for result in self.all_results]
        with open(f"{output_dir}/raw_results.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # 分析结果
        analysis = PerformanceAnalyzer.analyze_results(self.all_results)
        with open(f"{output_dir}/analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # 生成报告
        report = PerformanceAnalyzer.generate_report(analysis, self.config)
        with open(f"{output_dir}/report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        
        # 生成图表
        PerformanceAnalyzer.plot_performance_charts(self.all_results, output_dir)
        
        logger.info(f"测试结果已保存到 {output_dir} 目录")
        
        return analysis, report

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Xiaozhi ESP32 Server 性能测试")
    parser.add_argument("--url", default="http://localhost:8080", help="服务器URL")
    parser.add_argument("--devices", type=int, default=10, help="并发设备数")
    parser.add_argument("--duration", type=int, default=300, help="测试时长(秒)")
    parser.add_argument("--interval", type=float, default=2.0, help="请求间隔(秒)")
    parser.add_argument("--output", default="test_results", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建测试配置
    config = TestConfig(
        base_url=args.url,
        websocket_url=args.url.replace("http", "ws"),
        concurrent_devices=args.devices,
        test_duration=args.duration,
        request_interval=args.interval
    )
    
    # 创建负载测试器
    tester = LoadTester(config)
    
    try:
        # 运行测试
        results = await tester.run_test()
        
        # 保存结果
        analysis, report = tester.save_results(args.output)
        
        # 打印报告
        print(report)
        
        # 性能评估
        total_requests = sum(m['total_requests'] for m in analysis.values())
        total_successful = sum(m['successful_requests'] for m in analysis.values())
        overall_success_rate = total_successful / total_requests * 100 if total_requests > 0 else 0
        
        if overall_success_rate >= 95:
            logger.info("🎉 性能测试通过！系统表现优秀")
        elif overall_success_rate >= 90:
            logger.info("✅ 性能测试基本通过，系统表现良好")
        elif overall_success_rate >= 80:
            logger.warning("⚠️ 性能测试勉强通过，建议优化")
        else:
            logger.error("❌ 性能测试未通过，需要紧急优化")
        
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())