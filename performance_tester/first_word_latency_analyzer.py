#!/usr/bin/env python3
"""
首字延迟分析工具
分析VAD、ASR、LLM、TTS各环节的延迟贡献
"""

import asyncio
import time
import json
import statistics
from typing import Dict, List, Tuple
import aiohttp
from tabulate import tabulate
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FirstWordLatencyAnalyzer:
    def __init__(self):
        self.asr_url = "http://localhost:8001"
        self.llm_url = "http://localhost:8002"  # 假设LLM服务端口
        self.tts_url = "http://localhost:8003"  # 假设TTS服务端口
        self.vad_url = "http://localhost:8004"  # 假设VAD服务端口
        
        self.results = {
            'vad': [],
            'asr': [],
            'llm': [],
            'tts': [],
            'total_pipeline': []
        }
    
    async def _test_vad_latency(self, session: aiohttp.ClientSession, audio_data: bytes) -> float:
        """测试VAD延迟"""
        start_time = time.time()
        
        try:
            # 模拟VAD检测
            # 实际应该调用真实的VAD服务
            await asyncio.sleep(0.02)  # 模拟20ms VAD处理时间
            
            return time.time() - start_time
        except Exception as e:
            logger.error(f"VAD测试失败: {e}")
            return 0.05  # 返回默认值
    
    async def _test_asr_latency(self, session: aiohttp.ClientSession, audio_data: bytes) -> Tuple[float, bool]:
        """测试ASR延迟"""
        start_time = time.time()
        
        try:
            import base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            request_data = {
                "session_id": f"latency_test_{int(time.time())}",
                "audio_data": audio_base64,
                "sample_rate": 16000,
                "language": "zh",
                "priority": 1,
                "timestamp": time.time()
            }
            
            async with session.post(
                f"{self.asr_url}/asr/recognize",
                json=request_data,
                headers={'Content-Type': 'application/json'},
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                
                latency = time.time() - start_time
                success = response.status == 200
                
                if success:
                    result = await response.json()
                    logger.debug(f"ASR结果: {result.get('text', '')}")
                
                return latency, success
                
        except Exception as e:
            logger.error(f"ASR测试失败: {e}")
            return time.time() - start_time, False
    
    async def _test_llm_latency(self, session: aiohttp.ClientSession, text: str) -> Tuple[float, bool]:
        """测试LLM首token延迟"""
        start_time = time.time()
        
        try:
            # 模拟LLM调用
            # 实际应该调用真实的LLM服务
            await asyncio.sleep(0.3)  # 模拟300ms LLM首token生成时间
            
            return time.time() - start_time, True
            
        except Exception as e:
            logger.error(f"LLM测试失败: {e}")
            return time.time() - start_time, False
    
    async def _test_tts_latency(self, session: aiohttp.ClientSession, text: str) -> Tuple[float, bool]:
        """测试TTS延迟"""
        start_time = time.time()
        
        try:
            # 模拟TTS调用
            # 实际应该调用真实的TTS服务
            await asyncio.sleep(0.1)  # 模拟100ms TTS处理时间
            
            return time.time() - start_time, True
            
        except Exception as e:
            logger.error(f"TTS测试失败: {e}")
            return time.time() - start_time, False
    
    async def _test_full_pipeline(self, session: aiohttp.ClientSession, audio_data: bytes) -> Dict:
        """测试完整语音交互流水线"""
        pipeline_start = time.time()
        
        # 1. VAD检测
        vad_start = time.time()
        vad_latency = await self._test_vad_latency(session, audio_data)
        vad_end = time.time()
        
        # 2. ASR识别
        asr_start = time.time()
        asr_latency, asr_success = await self._test_asr_latency(session, audio_data)
        asr_end = time.time()
        
        # 3. LLM生成 (使用模拟文本)
        llm_start = time.time()
        llm_latency, llm_success = await self._test_llm_latency(session, "用户说话内容")
        llm_end = time.time()
        
        # 4. TTS合成
        tts_start = time.time()
        tts_latency, tts_success = await self._test_tts_latency(session, "AI回复内容")
        tts_end = time.time()
        
        total_latency = time.time() - pipeline_start
        
        return {
            'vad_latency': vad_latency,
            'asr_latency': asr_latency,
            'llm_latency': llm_latency,
            'tts_latency': tts_latency,
            'total_latency': total_latency,
            'asr_success': asr_success,
            'llm_success': llm_success,
            'tts_success': tts_success,
            'timestamp': pipeline_start
        }
    
    async def run_latency_analysis(self, test_count: int = 20):
        """运行首字延迟分析"""
        logger.info(f"开始首字延迟分析，测试次数: {test_count}")
        
        # 生成测试音频数据
        test_audio = b"fake_audio_data" * 1000  # 模拟音频数据
        
        async with aiohttp.ClientSession() as session:
            for i in range(test_count):
                logger.info(f"执行第 {i+1}/{test_count} 次测试...")
                
                result = await self._test_full_pipeline(session, test_audio)
                
                # 记录各环节结果
                self.results['vad'].append(result['vad_latency'])
                self.results['asr'].append(result['asr_latency'])
                self.results['llm'].append(result['llm_latency'])
                self.results['tts'].append(result['tts_latency'])
                self.results['total_pipeline'].append(result['total_latency'])
                
                # 短暂间隔
                await asyncio.sleep(0.5)
        
        self._analyze_and_report()
    
    def _analyze_and_report(self):
        """分析并生成报告"""
        print("\n" + "=" * 80)
        print("🎯 首字延迟分析报告")
        print("=" * 80)
        
        # 计算各环节统计数据
        stats_data = []
        total_avg_latency = 0
        
        for component, latencies in self.results.items():
            if not latencies:
                continue
                
            avg_latency = statistics.mean(latencies) * 1000  # 转换为毫秒
            min_latency = min(latencies) * 1000
            max_latency = max(latencies) * 1000
            p95_latency = statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) >= 20 else max_latency
            
            # 计算延迟贡献百分比
            if component != 'total_pipeline':
                total_avg_latency += avg_latency
            
            stats_data.append({
                'component': component,
                'avg_ms': avg_latency,
                'min_ms': min_latency,
                'max_ms': max_latency,
                'p95_ms': p95_latency
            })
        
        # 创建结果表格
        headers = ["环节", "平均延迟(ms)", "最小延迟(ms)", "最大延迟(ms)", "P95延迟(ms)", "延迟占比(%)"]
        table_data = []
        
        component_names = {
            'vad': 'VAD检测',
            'asr': 'ASR识别',
            'llm': 'LLM生成',
            'tts': 'TTS合成',
            'total_pipeline': '总流水线'
        }
        
        for stat in stats_data:
            component = stat['component']
            if component == 'total_pipeline':
                percentage = 100.0
            else:
                percentage = (stat['avg_ms'] / total_avg_latency) * 100 if total_avg_latency > 0 else 0
            
            # 添加状态指示器
            if component == 'asr' and stat['avg_ms'] > 500:
                status = "🔴 瓶颈"
            elif component == 'llm' and stat['avg_ms'] > 800:
                status = "🟡 关注"
            elif stat['avg_ms'] > 200:
                status = "🟠 优化"
            else:
                status = "✅ 良好"
            
            table_data.append([
                f"{component_names.get(component, component)} {status}",
                f"{stat['avg_ms']:.0f}",
                f"{stat['min_ms']:.0f}",
                f"{stat['max_ms']:.0f}",
                f"{stat['p95_ms']:.0f}",
                f"{percentage:.1f}%"
            ])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        # 生成优化建议
        self._generate_optimization_suggestions(stats_data)
    
    def _generate_optimization_suggestions(self, stats_data: List[Dict]):
        """生成优化建议"""
        print("\n💡 首字延迟优化建议:")
        print("-" * 50)
        
        # 找出主要瓶颈
        component_stats = {stat['component']: stat['avg_ms'] for stat in stats_data if stat['component'] != 'total_pipeline'}
        sorted_components = sorted(component_stats.items(), key=lambda x: x[1], reverse=True)
        
        print("🎯 优化优先级 (按延迟贡献排序):")
        for i, (component, avg_ms) in enumerate(sorted_components, 1):
            component_names = {
                'vad': 'VAD检测',
                'asr': 'ASR识别', 
                'llm': 'LLM生成',
                'tts': 'TTS合成'
            }
            print(f"  {i}. {component_names.get(component, component)}: {avg_ms:.0f}ms")
        
        print("\n🔧 具体优化方案:")
        
        # ASR优化建议
        asr_latency = component_stats.get('asr', 0)
        if asr_latency > 300:
            print("\n🔴 ASR识别优化 (主要瓶颈):")
            print("  • 启用模型预热: 减少冷启动延迟")
            print("  • 增加ASR_MAX_CONCURRENT: 提高并发处理能力")
            print("  • 优化ASR_BATCH_SIZE: 平衡吞吐量和延迟")
            print("  • 启用ASR缓存: 缓存常见音频模式")
            print("  • 使用更快的模型: 考虑FP16/INT8量化")
            print("  • 音频预处理优化: 减少数据转换时间")
        
        # LLM优化建议
        llm_latency = component_stats.get('llm', 0)
        if llm_latency > 500:
            print("\n🟡 LLM生成优化:")
            print("  • 模型预热: 保持模型在内存中")
            print("  • 首token优化: 使用流式生成")
            print("  • 上下文缓存: 缓存常见对话上下文")
            print("  • 模型量化: 使用FP16或INT8")
            print("  • 并发控制: 优化LLM_MAX_CONCURRENT")
        
        # VAD优化建议
        vad_latency = component_stats.get('vad', 0)
        if vad_latency > 50:
            print("\n🟠 VAD检测优化:")
            print("  • 减少音频缓冲: 降低VAD_BUFFER_SIZE")
            print("  • 算法优化: 使用更快的VAD算法")
            print("  • 硬件加速: 启用GPU加速")
        
        # TTS优化建议
        tts_latency = component_stats.get('tts', 0)
        if tts_latency > 150:
            print("\n🟠 TTS合成优化:")
            print("  • 流式合成: 边生成边播放")
            print("  • 音频缓存: 缓存常见回复")
            print("  • 模型优化: 使用更快的TTS模型")
        
        # 系统级优化
        total_latency = sum(component_stats.values())
        if total_latency > 1000:
            print("\n⚡ 系统级优化:")
            print("  • 流水线并行: VAD+ASR+LLM并行处理")
            print("  • 预测性加载: 提前加载可能需要的模型")
            print("  • 内存优化: 增加系统内存，减少swap")
            print("  • CPU优化: 使用更多CPU核心")
            print("  • 网络优化: 减少服务间通信延迟")

async def main():
    """主函数"""
    analyzer = FirstWordLatencyAnalyzer()
    await analyzer.run_latency_analysis(test_count=10)

if __name__ == "__main__":
    asyncio.run(main())