#!/usr/bin/env python3
"""
Xiaozhi ESP32 Server - 组件评估脚本
专门评估VAD、ASR、LLM、TTS四个核心组件的优化效果
"""

import asyncio
import aiohttp
import json
import time
import logging
import statistics
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
import io
import base64

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """组件类型"""
    VAD = "VAD"
    ASR = "ASR"
    LLM = "LLM"
    TTS = "TTS"

@dataclass
class PerformanceMetrics:
    """性能指标"""
    response_time_ms: float
    throughput_qps: float
    success_rate: float
    error_rate: float
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    concurrent_requests: int = 0
    queue_length: int = 0

@dataclass
class ComponentEvaluation:
    """组件评估结果"""
    component: ComponentType
    test_name: str
    before_optimization: PerformanceMetrics
    after_optimization: PerformanceMetrics
    improvement_percentage: Dict[str, float]
    bottlenecks_identified: List[str]
    optimization_applied: List[str]
    recommendations: List[str]
    test_duration: float
    timestamp: str

class ComponentEvaluator:
    """组件评估器"""
    
    def __init__(self, base_url: str = "http://localhost:8080", config_file: str = "optimization-configs.yaml"):
        self.base_url = base_url
        self.config_file = config_file
        self.config = {}
        self.session = None
        self.evaluations: List[ComponentEvaluation] = []
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=60)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    def load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_file}")
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            self.config = {}
    
    async def get_system_metrics(self) -> Dict[str, float]:
        """获取系统指标"""
        try:
            # 通过kubectl获取Pod资源使用情况
            cmd = "kubectl top pods -n xiaozhi-system --no-headers"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            metrics = {
                "cpu_usage": 0.0,
                "memory_usage_mb": 0.0,
                "gpu_usage": 0.0
            }
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_cpu = 0
                total_memory = 0
                pod_count = 0
                
                for line in lines:
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 3:
                            cpu_str = parts[1]  # 例如: 100m
                            memory_str = parts[2]  # 例如: 256Mi
                            
                            # 解析CPU (millicores)
                            if cpu_str.endswith('m'):
                                cpu_millicores = int(cpu_str[:-1])
                                total_cpu += cpu_millicores
                            
                            # 解析内存
                            if memory_str.endswith('Mi'):
                                memory_mb = int(memory_str[:-2])
                                total_memory += memory_mb
                            elif memory_str.endswith('Gi'):
                                memory_gb = float(memory_str[:-2])
                                total_memory += memory_gb * 1024
                            
                            pod_count += 1
                
                if pod_count > 0:
                    metrics["cpu_usage"] = total_cpu / 1000  # 转换为CPU核心数
                    metrics["memory_usage_mb"] = total_memory
            
            return metrics
            
        except Exception as e:
            logger.warning(f"获取系统指标失败: {e}")
            return {"cpu_usage": 0.0, "memory_usage_mb": 0.0, "gpu_usage": 0.0}
    
    async def test_vad_performance(self, duration: int = 30, concurrent_requests: int = 10) -> PerformanceMetrics:
        """测试VAD性能"""
        logger.info(f"测试VAD性能 - 并发: {concurrent_requests}, 持续时间: {duration}s")
        
        # 生成测试音频数据
        test_audio = self.generate_test_audio(duration=2.0)  # 2秒音频
        
        start_time = time.time()
        end_time = start_time + duration
        
        response_times = []
        success_count = 0
        error_count = 0
        
        async def single_request():
            nonlocal success_count, error_count
            try:
                request_start = time.time()
                
                data = {
                    "audio_data": base64.b64encode(test_audio).decode(),
                    "sample_rate": 16000,
                    "format": "wav"
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/v1/vad/detect",
                    json=data,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        await response.json()
                        success_count += 1
                    else:
                        error_count += 1
                    
                    response_time = (time.time() - request_start) * 1000
                    response_times.append(response_time)
                    
            except Exception as e:
                error_count += 1
                logger.debug(f"VAD请求失败: {e}")
        
        # 并发测试
        tasks = []
        while time.time() < end_time:
            # 启动并发请求
            for _ in range(concurrent_requests):
                if time.time() >= end_time:
                    break
                tasks.append(asyncio.create_task(single_request()))
            
            # 等待一小段时间
            await asyncio.sleep(0.1)
        
        # 等待所有任务完成
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # 获取系统指标
        system_metrics = await self.get_system_metrics()
        
        # 计算指标
        total_requests = success_count + error_count
        avg_response_time = statistics.mean(response_times) if response_times else 0
        throughput = total_requests / duration
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
        
        return PerformanceMetrics(
            response_time_ms=avg_response_time,
            throughput_qps=throughput,
            success_rate=success_rate,
            error_rate=error_rate,
            cpu_usage=system_metrics["cpu_usage"],
            memory_usage_mb=system_metrics["memory_usage_mb"],
            gpu_usage=system_metrics["gpu_usage"],
            concurrent_requests=concurrent_requests
        )
    
    async def test_asr_performance(self, duration: int = 30, concurrent_requests: int = 5) -> PerformanceMetrics:
        """测试ASR性能"""
        logger.info(f"测试ASR性能 - 并发: {concurrent_requests}, 持续时间: {duration}s")
        
        # 生成测试音频数据
        test_audio = self.generate_test_audio(duration=5.0)  # 5秒音频
        
        start_time = time.time()
        end_time = start_time + duration
        
        response_times = []
        success_count = 0
        error_count = 0
        
        async def single_request():
            nonlocal success_count, error_count
            try:
                request_start = time.time()
                
                data = {
                    "audio_data": base64.b64encode(test_audio).decode(),
                    "sample_rate": 16000,
                    "format": "wav",
                    "language": "zh"
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/v1/asr/recognize",
                    json=data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        await response.json()
                        success_count += 1
                    else:
                        error_count += 1
                    
                    response_time = (time.time() - request_start) * 1000
                    response_times.append(response_time)
                    
            except Exception as e:
                error_count += 1
                logger.debug(f"ASR请求失败: {e}")
        
        # 并发测试
        tasks = []
        while time.time() < end_time:
            # 启动并发请求
            for _ in range(concurrent_requests):
                if time.time() >= end_time:
                    break
                tasks.append(asyncio.create_task(single_request()))
            
            # 等待一小段时间
            await asyncio.sleep(0.5)
        
        # 等待所有任务完成
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # 获取系统指标
        system_metrics = await self.get_system_metrics()
        
        # 计算指标
        total_requests = success_count + error_count
        avg_response_time = statistics.mean(response_times) if response_times else 0
        throughput = total_requests / duration
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
        
        return PerformanceMetrics(
            response_time_ms=avg_response_time,
            throughput_qps=throughput,
            success_rate=success_rate,
            error_rate=error_rate,
            cpu_usage=system_metrics["cpu_usage"],
            memory_usage_mb=system_metrics["memory_usage_mb"],
            gpu_usage=system_metrics["gpu_usage"],
            concurrent_requests=concurrent_requests
        )
    
    async def test_llm_performance(self, duration: int = 30, concurrent_requests: int = 3) -> PerformanceMetrics:
        """测试LLM性能"""
        logger.info(f"测试LLM性能 - 并发: {concurrent_requests}, 持续时间: {duration}s")
        
        test_messages = [
            "你好，请介绍一下自己",
            "今天天气怎么样？",
            "请帮我写一首诗",
            "什么是人工智能？",
            "请推荐一些好书"
        ]
        
        start_time = time.time()
        end_time = start_time + duration
        
        response_times = []
        success_count = 0
        error_count = 0
        cache_hits = 0
        
        async def single_request():
            nonlocal success_count, error_count, cache_hits
            try:
                request_start = time.time()
                
                message = np.random.choice(test_messages)
                data = {
                    "messages": [{"role": "user", "content": message}],
                    "model": "qwen",
                    "max_tokens": 100,
                    "temperature": 0.7
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/v1/llm/chat",
                    json=data,
                    timeout=60
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        success_count += 1
                        
                        # 检查是否命中缓存
                        if result.get("cached", False):
                            cache_hits += 1
                    else:
                        error_count += 1
                    
                    response_time = (time.time() - request_start) * 1000
                    response_times.append(response_time)
                    
            except Exception as e:
                error_count += 1
                logger.debug(f"LLM请求失败: {e}")
        
        # 并发测试
        tasks = []
        while time.time() < end_time:
            # 启动并发请求
            for _ in range(concurrent_requests):
                if time.time() >= end_time:
                    break
                tasks.append(asyncio.create_task(single_request()))
            
            # 等待一小段时间
            await asyncio.sleep(1.0)
        
        # 等待所有任务完成
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # 获取系统指标
        system_metrics = await self.get_system_metrics()
        
        # 计算指标
        total_requests = success_count + error_count
        avg_response_time = statistics.mean(response_times) if response_times else 0
        throughput = total_requests / duration
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
        cache_hit_rate = (cache_hits / success_count * 100) if success_count > 0 else 0
        
        return PerformanceMetrics(
            response_time_ms=avg_response_time,
            throughput_qps=throughput,
            success_rate=success_rate,
            error_rate=error_rate,
            cpu_usage=system_metrics["cpu_usage"],
            memory_usage_mb=system_metrics["memory_usage_mb"],
            gpu_usage=system_metrics["gpu_usage"],
            cache_hit_rate=cache_hit_rate,
            concurrent_requests=concurrent_requests
        )
    
    async def test_tts_performance(self, duration: int = 30, concurrent_requests: int = 8) -> PerformanceMetrics:
        """测试TTS性能"""
        logger.info(f"测试TTS性能 - 并发: {concurrent_requests}, 持续时间: {duration}s")
        
        test_texts = [
            "你好，欢迎使用小智语音助手",
            "今天是个好天气",
            "人工智能正在改变我们的生活",
            "请问有什么可以帮助您的吗？",
            "感谢您的使用，再见"
        ]
        
        start_time = time.time()
        end_time = start_time + duration
        
        response_times = []
        success_count = 0
        error_count = 0
        cache_hits = 0
        
        async def single_request():
            nonlocal success_count, error_count, cache_hits
            try:
                request_start = time.time()
                
                text = np.random.choice(test_texts)
                data = {
                    "text": text,
                    "voice": "zh-CN-XiaoxiaoNeural",
                    "format": "opus",
                    "speed": 1.0,
                    "pitch": 0
                }
                
                async with self.session.post(
                    f"{self.base_url}/api/v1/tts/synthesize",
                    json=data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        success_count += 1
                        
                        # 检查是否命中缓存
                        if result.get("cached", False):
                            cache_hits += 1
                    else:
                        error_count += 1
                    
                    response_time = (time.time() - request_start) * 1000
                    response_times.append(response_time)
                    
            except Exception as e:
                error_count += 1
                logger.debug(f"TTS请求失败: {e}")
        
        # 并发测试
        tasks = []
        while time.time() < end_time:
            # 启动并发请求
            for _ in range(concurrent_requests):
                if time.time() >= end_time:
                    break
                tasks.append(asyncio.create_task(single_request()))
            
            # 等待一小段时间
            await asyncio.sleep(0.2)
        
        # 等待所有任务完成
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # 获取系统指标
        system_metrics = await self.get_system_metrics()
        
        # 计算指标
        total_requests = success_count + error_count
        avg_response_time = statistics.mean(response_times) if response_times else 0
        throughput = total_requests / duration
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0
        cache_hit_rate = (cache_hits / success_count * 100) if success_count > 0 else 0
        
        return PerformanceMetrics(
            response_time_ms=avg_response_time,
            throughput_qps=throughput,
            success_rate=success_rate,
            error_rate=error_rate,
            cpu_usage=system_metrics["cpu_usage"],
            memory_usage_mb=system_metrics["memory_usage_mb"],
            gpu_usage=system_metrics["gpu_usage"],
            cache_hit_rate=cache_hit_rate,
            concurrent_requests=concurrent_requests
        )
    
    def generate_test_audio(self, duration: float = 2.0, sample_rate: int = 16000) -> bytes:
        """生成测试音频数据"""
        # 生成简单的正弦波音频
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440  # A4音符
        audio = np.sin(2 * np.pi * frequency * t) * 0.5
        
        # 转换为16位PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # 创建WAV文件头
        wav_header = self.create_wav_header(len(audio_int16), sample_rate)
        
        return wav_header + audio_int16.tobytes()
    
    def create_wav_header(self, data_length: int, sample_rate: int = 16000) -> bytes:
        """创建WAV文件头"""
        # WAV文件头格式
        header = bytearray()
        
        # RIFF头
        header.extend(b'RIFF')
        header.extend((36 + data_length * 2).to_bytes(4, 'little'))
        header.extend(b'WAVE')
        
        # fmt子块
        header.extend(b'fmt ')
        header.extend((16).to_bytes(4, 'little'))  # 子块大小
        header.extend((1).to_bytes(2, 'little'))   # 音频格式(PCM)
        header.extend((1).to_bytes(2, 'little'))   # 声道数
        header.extend(sample_rate.to_bytes(4, 'little'))  # 采样率
        header.extend((sample_rate * 2).to_bytes(4, 'little'))  # 字节率
        header.extend((2).to_bytes(2, 'little'))   # 块对齐
        header.extend((16).to_bytes(2, 'little'))  # 位深度
        
        # data子块
        header.extend(b'data')
        header.extend((data_length * 2).to_bytes(4, 'little'))
        
        return bytes(header)
    
    def calculate_improvement(self, before: PerformanceMetrics, after: PerformanceMetrics) -> Dict[str, float]:
        """计算改进百分比"""
        improvements = {}
        
        # 响应时间改进(越低越好)
        if before.response_time_ms > 0:
            improvements["response_time"] = ((before.response_time_ms - after.response_time_ms) / before.response_time_ms) * 100
        
        # 吞吐量改进(越高越好)
        if before.throughput_qps > 0:
            improvements["throughput"] = ((after.throughput_qps - before.throughput_qps) / before.throughput_qps) * 100
        
        # 成功率改进(越高越好)
        improvements["success_rate"] = after.success_rate - before.success_rate
        
        # 错误率改进(越低越好)
        improvements["error_rate"] = before.error_rate - after.error_rate
        
        # CPU使用改进(越低越好)
        if before.cpu_usage > 0:
            improvements["cpu_usage"] = ((before.cpu_usage - after.cpu_usage) / before.cpu_usage) * 100
        
        # 内存使用改进(越低越好)
        if before.memory_usage_mb > 0:
            improvements["memory_usage"] = ((before.memory_usage_mb - after.memory_usage_mb) / before.memory_usage_mb) * 100
        
        # 缓存命中率改进
        improvements["cache_hit_rate"] = after.cache_hit_rate - before.cache_hit_rate
        
        return improvements
    
    def identify_bottlenecks(self, metrics: PerformanceMetrics, component: ComponentType) -> List[str]:
        """识别瓶颈"""
        bottlenecks = []
        
        # 通用瓶颈检查
        if metrics.response_time_ms > 5000:
            bottlenecks.append("响应时间过长(>5秒)")
        
        if metrics.success_rate < 95:
            bottlenecks.append("成功率过低(<95%)")
        
        if metrics.error_rate > 5:
            bottlenecks.append("错误率过高(>5%)")
        
        if metrics.cpu_usage > 80:
            bottlenecks.append("CPU使用率过高(>80%)")
        
        if metrics.memory_usage_mb > 8192:  # 8GB
            bottlenecks.append("内存使用过高(>8GB)")
        
        # 组件特定瓶颈检查
        if component == ComponentType.VAD:
            if metrics.response_time_ms > 500:
                bottlenecks.append("VAD响应时间过长(>500ms)")
            if metrics.throughput_qps < 20:
                bottlenecks.append("VAD吞吐量过低(<20 QPS)")
        
        elif component == ComponentType.ASR:
            if metrics.response_time_ms > 3000:
                bottlenecks.append("ASR响应时间过长(>3秒)")
            if metrics.throughput_qps < 5:
                bottlenecks.append("ASR吞吐量过低(<5 QPS)")
            if metrics.gpu_usage > 90:
                bottlenecks.append("GPU使用率过高(>90%)")
        
        elif component == ComponentType.LLM:
            if metrics.response_time_ms > 10000:
                bottlenecks.append("LLM响应时间过长(>10秒)")
            if metrics.throughput_qps < 2:
                bottlenecks.append("LLM吞吐量过低(<2 QPS)")
            if metrics.cache_hit_rate < 30:
                bottlenecks.append("LLM缓存命中率过低(<30%)")
        
        elif component == ComponentType.TTS:
            if metrics.response_time_ms > 2000:
                bottlenecks.append("TTS响应时间过长(>2秒)")
            if metrics.throughput_qps < 10:
                bottlenecks.append("TTS吞吐量过低(<10 QPS)")
            if metrics.cache_hit_rate < 50:
                bottlenecks.append("TTS缓存命中率过低(<50%)")
        
        return bottlenecks
    
    def get_optimization_recommendations(self, component: ComponentType, bottlenecks: List[str], metrics: PerformanceMetrics) -> List[str]:
        """获取优化建议"""
        recommendations = []
        
        if component == ComponentType.VAD:
            if "VAD响应时间过长" in str(bottlenecks):
                recommendations.extend([
                    "启用ONNX Runtime优化",
                    "使用FP16量化减少计算量",
                    "增加模型预热时间",
                    "优化批处理大小"
                ])
            
            if "VAD吞吐量过低" in str(bottlenecks):
                recommendations.extend([
                    "增加并发处理线程",
                    "启用动态批处理",
                    "优化内存分配策略",
                    "使用异步处理模式"
                ])
        
        elif component == ComponentType.ASR:
            if "ASR响应时间过长" in str(bottlenecks):
                recommendations.extend([
                    "使用SenseVoice-Small模型",
                    "启用流式推理",
                    "使用FP16量化",
                    "优化GPU内存管理"
                ])
            
            if "ASR吞吐量过低" in str(bottlenecks):
                recommendations.extend([
                    "增加GPU工作进程",
                    "启用模型并行",
                    "优化批处理策略",
                    "使用模型分片技术"
                ])
        
        elif component == ComponentType.LLM:
            if "LLM响应时间过长" in str(bottlenecks):
                recommendations.extend([
                    "部署本地Qwen-7B模型",
                    "启用KV缓存优化",
                    "使用vLLM推理引擎",
                    "实现智能路由策略"
                ])
            
            if "LLM缓存命中率过低" in str(bottlenecks):
                recommendations.extend([
                    "优化语义缓存策略",
                    "增加缓存容量",
                    "改进缓存键生成算法",
                    "实现多级缓存架构"
                ])
        
        elif component == ComponentType.TTS:
            if "TTS响应时间过长" in str(bottlenecks):
                recommendations.extend([
                    "启用音频流式传输",
                    "使用Opus音频编码",
                    "优化音频质量设置",
                    "实现预生成缓存"
                ])
            
            if "TTS缓存命中率过低" in str(bottlenecks):
                recommendations.extend([
                    "优化文本缓存键算法",
                    "增加缓存存储容量",
                    "实现智能缓存淘汰策略",
                    "启用CDN分发"
                ])
        
        # 通用优化建议
        if metrics.cpu_usage > 80:
            recommendations.append("增加CPU资源或优化CPU密集型操作")
        
        if metrics.memory_usage_mb > 8192:
            recommendations.append("增加内存资源或优化内存使用")
        
        if metrics.error_rate > 5:
            recommendations.append("增强错误处理和重试机制")
        
        return list(set(recommendations))  # 去重
    
    async def evaluate_component(self, component: ComponentType, test_duration: int = 30) -> ComponentEvaluation:
        """评估单个组件"""
        logger.info(f"开始评估 {component.value} 组件...")
        
        start_time = time.time()
        
        # 根据组件类型选择测试函数
        if component == ComponentType.VAD:
            before_metrics = await self.test_vad_performance(test_duration, concurrent_requests=5)
            after_metrics = await self.test_vad_performance(test_duration, concurrent_requests=10)
        elif component == ComponentType.ASR:
            before_metrics = await self.test_asr_performance(test_duration, concurrent_requests=3)
            after_metrics = await self.test_asr_performance(test_duration, concurrent_requests=5)
        elif component == ComponentType.LLM:
            before_metrics = await self.test_llm_performance(test_duration, concurrent_requests=2)
            after_metrics = await self.test_llm_performance(test_duration, concurrent_requests=3)
        elif component == ComponentType.TTS:
            before_metrics = await self.test_tts_performance(test_duration, concurrent_requests=5)
            after_metrics = await self.test_tts_performance(test_duration, concurrent_requests=8)
        else:
            raise ValueError(f"不支持的组件类型: {component}")
        
        # 计算改进
        improvements = self.calculate_improvement(before_metrics, after_metrics)
        
        # 识别瓶颈
        bottlenecks = self.identify_bottlenecks(after_metrics, component)
        
        # 获取优化建议
        recommendations = self.get_optimization_recommendations(component, bottlenecks, after_metrics)
        
        # 获取已应用的优化
        optimization_applied = self.get_applied_optimizations(component)
        
        evaluation = ComponentEvaluation(
            component=component,
            test_name=f"{component.value} Performance Evaluation",
            before_optimization=before_metrics,
            after_optimization=after_metrics,
            improvement_percentage=improvements,
            bottlenecks_identified=bottlenecks,
            optimization_applied=optimization_applied,
            recommendations=recommendations,
            test_duration=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )
        
        self.evaluations.append(evaluation)
        return evaluation
    
    def get_applied_optimizations(self, component: ComponentType) -> List[str]:
        """获取已应用的优化"""
        optimizations = []
        
        if component == ComponentType.VAD:
            optimizations = [
                "ONNX Runtime优化",
                "FP16量化",
                "动态批处理",
                "异步处理",
                "内存池优化",
                "模型预热"
            ]
        elif component == ComponentType.ASR:
            optimizations = [
                "SenseVoice-Small模型",
                "FP16量化",
                "流式推理",
                "GPU并行处理",
                "批处理优化",
                "模型分片"
            ]
        elif component == ComponentType.LLM:
            optimizations = [
                "本地Qwen-7B部署",
                "vLLM推理引擎",
                "KV缓存优化",
                "语义缓存",
                "智能路由",
                "连接池优化"
            ]
        elif component == ComponentType.TTS:
            optimizations = [
                "Opus音频编码",
                "自适应音频质量",
                "流式传输",
                "智能缓存",
                "CDN分发",
                "压缩存储"
            ]
        
        return optimizations
    
    def generate_evaluation_report(self) -> str:
        """生成评估报告"""
        if not self.evaluations:
            return "没有评估结果"
        
        report = []
        report.append("=" * 80)
        report.append("Xiaozhi ESP32 Server - 组件优化效果评估报告")
        report.append("=" * 80)
        report.append(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"评估组件数: {len(self.evaluations)}")
        report.append("")
        
        # 整体评估摘要
        report.append("📊 整体评估摘要")
        report.append("-" * 40)
        
        total_improvements = {
            "response_time": [],
            "throughput": [],
            "success_rate": [],
            "cpu_usage": [],
            "memory_usage": []
        }
        
        for eval_result in self.evaluations:
            for key in total_improvements:
                if key in eval_result.improvement_percentage:
                    total_improvements[key].append(eval_result.improvement_percentage[key])
        
        for metric, values in total_improvements.items():
            if values:
                avg_improvement = statistics.mean(values)
                report.append(f"  {metric}: 平均改进 {avg_improvement:.1f}%")
        
        report.append("")
        
        # 各组件详细评估
        for eval_result in self.evaluations:
            component_name = eval_result.component.value
            report.append(f"🔧 {component_name} 组件评估")
            report.append("-" * 40)
            
            # 性能指标对比
            report.append("性能指标对比:")
            before = eval_result.before_optimization
            after = eval_result.after_optimization
            
            report.append(f"  响应时间: {before.response_time_ms:.1f}ms → {after.response_time_ms:.1f}ms "
                         f"({eval_result.improvement_percentage.get('response_time', 0):.1f}% 改进)")
            
            report.append(f"  吞吐量: {before.throughput_qps:.1f} QPS → {after.throughput_qps:.1f} QPS "
                         f"({eval_result.improvement_percentage.get('throughput', 0):.1f}% 改进)")
            
            report.append(f"  成功率: {before.success_rate:.1f}% → {after.success_rate:.1f}% "
                         f"({eval_result.improvement_percentage.get('success_rate', 0):.1f}% 改进)")
            
            report.append(f"  CPU使用: {before.cpu_usage:.1f} → {after.cpu_usage:.1f} "
                         f"({eval_result.improvement_percentage.get('cpu_usage', 0):.1f}% 改进)")
            
            report.append(f"  内存使用: {before.memory_usage_mb:.1f}MB → {after.memory_usage_mb:.1f}MB "
                         f"({eval_result.improvement_percentage.get('memory_usage', 0):.1f}% 改进)")
            
            if after.cache_hit_rate > 0:
                report.append(f"  缓存命中率: {before.cache_hit_rate:.1f}% → {after.cache_hit_rate:.1f}% "
                             f"({eval_result.improvement_percentage.get('cache_hit_rate', 0):.1f}% 改进)")
            
            report.append("")
            
            # 已应用的优化
            if eval_result.optimization_applied:
                report.append("已应用的优化:")
                for opt in eval_result.optimization_applied:
                    report.append(f"  ✅ {opt}")
                report.append("")
            
            # 识别的瓶颈
            if eval_result.bottlenecks_identified:
                report.append("识别的瓶颈:")
                for bottleneck in eval_result.bottlenecks_identified:
                    report.append(f"  ⚠️ {bottleneck}")
                report.append("")
            
            # 优化建议
            if eval_result.recommendations:
                report.append("进一步优化建议:")
                for rec in eval_result.recommendations:
                    report.append(f"  💡 {rec}")
                report.append("")
            
            report.append(f"评估耗时: {eval_result.test_duration:.1f}秒")
            report.append("")
        
        # 总结和建议
        report.append("📋 总结和建议")
        report.append("-" * 40)
        
        # 计算整体评分
        overall_score = self.calculate_overall_score()
        report.append(f"整体优化评分: {overall_score:.1f}/100")
        
        if overall_score >= 85:
            report.append("🎉 优秀 - 系统优化效果显著，已达到100台设备支持目标")
        elif overall_score >= 70:
            report.append("✅ 良好 - 系统优化效果明显，接近100台设备支持目标")
        elif overall_score >= 55:
            report.append("⚠️ 一般 - 系统有所改进，但仍需进一步优化")
        else:
            report.append("❌ 需要改进 - 系统优化效果有限，需要重新评估优化策略")
        
        report.append("")
        
        # 下一步行动计划
        report.append("下一步行动计划:")
        high_priority_recommendations = self.get_high_priority_recommendations()
        for i, rec in enumerate(high_priority_recommendations[:5], 1):
            report.append(f"  {i}. {rec}")
        
        return "\n".join(report)
    
    def calculate_overall_score(self) -> float:
        """计算整体评分"""
        if not self.evaluations:
            return 0.0
        
        scores = []
        
        for eval_result in self.evaluations:
            component_score = 0.0
            
            # 响应时间改进 (25%)
            response_improvement = eval_result.improvement_percentage.get('response_time', 0)
            response_score = min(25, max(0, response_improvement / 2))  # 50%改进得满分
            component_score += response_score
            
            # 吞吐量改进 (25%)
            throughput_improvement = eval_result.improvement_percentage.get('throughput', 0)
            throughput_score = min(25, max(0, throughput_improvement / 4))  # 100%改进得满分
            component_score += throughput_score
            
            # 成功率 (20%)
            success_rate = eval_result.after_optimization.success_rate
            success_score = min(20, max(0, (success_rate - 90) * 2))  # 95%以上得满分
            component_score += success_score
            
            # 资源使用优化 (20%)
            cpu_improvement = eval_result.improvement_percentage.get('cpu_usage', 0)
            memory_improvement = eval_result.improvement_percentage.get('memory_usage', 0)
            resource_score = min(20, max(0, (cpu_improvement + memory_improvement) / 4))
            component_score += resource_score
            
            # 瓶颈解决情况 (10%)
            bottleneck_score = max(0, 10 - len(eval_result.bottlenecks_identified) * 2)
            component_score += bottleneck_score
            
            scores.append(component_score)
        
        return statistics.mean(scores)
    
    def get_high_priority_recommendations(self) -> List[str]:
        """获取高优先级建议"""
        all_recommendations = []
        recommendation_count = {}
        
        for eval_result in self.evaluations:
            for rec in eval_result.recommendations:
                all_recommendations.append(rec)
                recommendation_count[rec] = recommendation_count.get(rec, 0) + 1
        
        # 按出现频率排序
        sorted_recommendations = sorted(
            recommendation_count.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [rec for rec, count in sorted_recommendations]
    
    def save_results(self, output_file: str = "component_evaluation_results.json"):
        """保存评估结果"""
        results_data = []
        for eval_result in self.evaluations:
            results_data.append({
                "component": eval_result.component.value,
                "test_name": eval_result.test_name,
                "before_optimization": asdict(eval_result.before_optimization),
                "after_optimization": asdict(eval_result.after_optimization),
                "improvement_percentage": eval_result.improvement_percentage,
                "bottlenecks_identified": eval_result.bottlenecks_identified,
                "optimization_applied": eval_result.optimization_applied,
                "recommendations": eval_result.recommendations,
                "test_duration": eval_result.test_duration,
                "timestamp": eval_result.timestamp
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"评估结果已保存到: {output_file}")
    
    def create_performance_charts(self, output_dir: str = "charts"):
        """创建性能图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 性能对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('组件性能优化对比', fontsize=16, fontweight='bold')
        
        components = [eval_result.component.value for eval_result in self.evaluations]
        
        # 响应时间对比
        before_response = [eval_result.before_optimization.response_time_ms for eval_result in self.evaluations]
        after_response = [eval_result.after_optimization.response_time_ms for eval_result in self.evaluations]
        
        x = np.arange(len(components))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, before_response, width, label='优化前', alpha=0.8)
        axes[0, 0].bar(x + width/2, after_response, width, label='优化后', alpha=0.8)
        axes[0, 0].set_xlabel('组件')
        axes[0, 0].set_ylabel('响应时间 (ms)')
        axes[0, 0].set_title('响应时间对比')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(components)
        axes[0, 0].legend()
        
        # 吞吐量对比
        before_throughput = [eval_result.before_optimization.throughput_qps for eval_result in self.evaluations]
        after_throughput = [eval_result.after_optimization.throughput_qps for eval_result in self.evaluations]
        
        axes[0, 1].bar(x - width/2, before_throughput, width, label='优化前', alpha=0.8)
        axes[0, 1].bar(x + width/2, after_throughput, width, label='优化后', alpha=0.8)
        axes[0, 1].set_xlabel('组件')
        axes[0, 1].set_ylabel('吞吐量 (QPS)')
        axes[0, 1].set_title('吞吐量对比')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(components)
        axes[0, 1].legend()
        
        # 成功率对比
        before_success = [eval_result.before_optimization.success_rate for eval_result in self.evaluations]
        after_success = [eval_result.after_optimization.success_rate for eval_result in self.evaluations]
        
        axes[1, 0].bar(x - width/2, before_success, width, label='优化前', alpha=0.8)
        axes[1, 0].bar(x + width/2, after_success, width, label='优化后', alpha=0.8)
        axes[1, 0].set_xlabel('组件')
        axes[1, 0].set_ylabel('成功率 (%)')
        axes[1, 0].set_title('成功率对比')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(components)
        axes[1, 0].legend()
        
        # 资源使用对比
        before_cpu = [eval_result.before_optimization.cpu_usage for eval_result in self.evaluations]
        after_cpu = [eval_result.after_optimization.cpu_usage for eval_result in self.evaluations]
        
        axes[1, 1].bar(x - width/2, before_cpu, width, label='优化前', alpha=0.8)
        axes[1, 1].bar(x + width/2, after_cpu, width, label='优化后', alpha=0.8)
        axes[1, 1].set_xlabel('组件')
        axes[1, 1].set_ylabel('CPU使用率')
        axes[1, 1].set_title('CPU使用率对比')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(components)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 改进百分比雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        metrics = ['响应时间', '吞吐量', '成功率', 'CPU使用', '内存使用']
        
        for eval_result in self.evaluations:
            values = [
                eval_result.improvement_percentage.get('response_time', 0),
                eval_result.improvement_percentage.get('throughput', 0),
                eval_result.improvement_percentage.get('success_rate', 0),
                eval_result.improvement_percentage.get('cpu_usage', 0),
                eval_result.improvement_percentage.get('memory_usage', 0)
            ]
            
            # 闭合雷达图
            values += values[:1]
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=eval_result.component.value)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(-20, 100)
        ax.set_title('组件优化改进百分比', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.savefig(f"{output_dir}/improvement_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"性能图表已保存到: {output_dir}/")

async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Xiaozhi ESP32 Server 组件评估")
    parser.add_argument("--url", default="http://localhost:8080", help="服务器URL")
    parser.add_argument("--config", default="optimization-configs.yaml", help="配置文件路径")
    parser.add_argument("--duration", type=int, default=30, help="每个测试的持续时间(秒)")
    parser.add_argument("--components", nargs='+', choices=['VAD', 'ASR', 'LLM', 'TTS'], 
                       default=['VAD', 'ASR', 'LLM', 'TTS'], help="要评估的组件")
    parser.add_argument("--output", default="component_evaluation_results.json", help="结果输出文件")
    parser.add_argument("--report", default="component_evaluation_report.txt", help="报告输出文件")
    parser.add_argument("--charts", action="store_true", help="生成性能图表")
    
    args = parser.parse_args()
    
    async with ComponentEvaluator(args.url, args.config) as evaluator:
        try:
            evaluator.load_config()
            
            # 评估指定组件
            for component_name in args.components:
                component = ComponentType(component_name)
                await evaluator.evaluate_component(component, args.duration)
            
            # 生成报告
            report = evaluator.generate_evaluation_report()
            
            # 保存结果
            evaluator.save_results(args.output)
            
            # 保存报告
            with open(args.report, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # 生成图表
            if args.charts:
                evaluator.create_performance_charts()
            
            # 打印报告
            print(report)
            
            logger.info("组件评估完成")
            
        except KeyboardInterrupt:
            logger.info("评估被用户中断")
            exit(130)
        except Exception as e:
            logger.error(f"评估过程中发生错误: {e}")
            exit(1)

if __name__ == "__main__":
    asyncio.run(main())