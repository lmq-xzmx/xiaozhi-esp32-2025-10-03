#!/usr/bin/env python3
"""
ASR (Automatic Speech Recognition) 微服务
基于SenseVoice，支持批处理、队列管理和模型缓存优化
4核8GB服务器优化版本：支持20-25台设备并发
"""

import asyncio
import logging
import time
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool
from config.redis_config import get_redis_client, OptimizedRedisClient
from core.queue_manager import get_queue_manager, QueueRequest, Priority
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import base64
from funasr import AutoModel
import librosa

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ASRRequest:
    session_id: str
    audio_data: bytes
    sample_rate: int = 16000
    language: str = "zh"
    timestamp: float = 0.0
    priority: int = 1  # 1=高优先级, 2=中优先级, 3=低优先级

@dataclass
class ASRResult:
    session_id: str
    text: str
    confidence: float
    language: str
    timestamp: float
    processing_time: float
    cached: bool = False

class SenseVoiceProcessor:
    """4核8GB服务器优化的SenseVoice处理器，支持批处理和量化优化"""
    
    def __init__(self, model_dir: str = "models/SenseVoiceSmall", batch_size: int = 10, enable_fp16: bool = True):
        self.model_dir = model_dir
        # 批处理配置 - 4核8GB优化：提升到10以增加吞吐量
        self.batch_size = batch_size
        self.enable_fp16 = enable_fp16
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 4核8GB优化：实时处理参数
        self.chunk_size = 640  # 40ms@16kHz，降低延迟
        self.chunk_duration_ms = 40
        self.overlap_ms = 8  # 减少重叠以节省计算
        self.max_audio_length_s = 25  # 降低最大音频长度
        self.beam_size = 1  # 贪婪解码，最快速度
        
        # 性能统计
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        
        # 内存管理
        self.max_memory_mb = 1500  # 限制最大内存使用1.5GB
        self.audio_buffer_pool = []
        self.result_cache_max_size = 768  # 优化缓存大小为768MB
        
        # 加载模型
        self.load_model()
        self.warmup_model()
    
    def _model_exists(self, model_path: str) -> bool:
        """检查模型是否存在"""
        import os
        return os.path.exists(model_path)
    
    def load_model(self):
        """加载SenseVoice模型，优先使用量化版本"""
        try:
            # 4核8GB优化：优先加载量化模型
            model_candidates = [
                f"{self.model_dir}_int8",  # 优先INT8量化
                f"{self.model_dir}_fp16" if self.enable_fp16 else None,
                self.model_dir
            ]
            
            model_path = None
            for candidate in model_candidates:
                if candidate and self._model_exists(candidate):
                    model_path = candidate
                    break
            
            if not model_path:
                model_path = self.model_dir
            
            logger.info(f"加载ASR模型: {model_path}")
            
            # 4核8GB优化：使用更保守的模型配置
            self.model = AutoModel(
                model=model_path,
                device=self.device,
                # 内存优化配置
                cache_dir="./cache/asr",
                trust_remote_code=True,
                # 4核8GB优化：降低并发处理
                batch_size=self.batch_size,
                # 启用FP16以节省内存
                torch_dtype=torch.float16 if self.enable_fp16 and self.device.type == 'cuda' else torch.float32
            )
            
            # 设置模型为评估模式
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # 4核8GB优化：启用JIT编译（如果支持）
            if hasattr(torch.jit, 'script') and hasattr(self.model, 'forward'):
                try:
                    self.model = torch.jit.script(self.model)
                    logger.info("启用JIT编译优化")
                except Exception as e:
                    logger.warning(f"JIT编译失败，使用原始模型: {e}")
            
            logger.info(f"ASR模型加载成功，设备: {self.device}")
            
        except Exception as e:
            logger.error(f"ASR模型加载失败: {e}")
            raise
    
    def warmup_model(self):
        """模型预热，减少首次推理延迟"""
        try:
            logger.info("开始ASR模型预热...")
            # 创建虚拟音频数据进行预热
            dummy_audio = np.random.randn(16000).astype(np.float32)  # 1秒音频
            
            # 执行2次预热推理（减少预热次数以节省时间）
            for i in range(2):
                start_time = time.time()
                _ = self.model.generate(input=dummy_audio, cache={}, language="zh", use_itn=True)
                warmup_time = time.time() - start_time
                logger.info(f"预热 {i+1}/2 完成，耗时: {warmup_time:.3f}s")
            
            logger.info("ASR模型预热完成")
        except Exception as e:
            logger.warning(f"ASR模型预热失败: {e}")
    
    def preprocess_audio(self, audio_data: bytes, sample_rate: int = 16000) -> np.ndarray:
        """音频预处理，4核8GB优化版本"""
        try:
            # 将字节数据转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 重采样到16kHz（如果需要）
            if sample_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
            # 4核8GB优化：限制音频长度以节省内存
            max_samples = 16000 * self.max_audio_length_s
            if len(audio_array) > max_samples:
                audio_array = audio_array[:max_samples]
            
            return audio_array
        except Exception as e:
            logger.error(f"音频预处理失败: {e}")
            raise
    
    def generate_audio_hash(self, audio_data: bytes) -> str:
        """生成音频数据的哈希值用于缓存"""
        return hashlib.md5(audio_data).hexdigest()
    
    async def process_batch(self, requests: List[ASRRequest]) -> List[ASRResult]:
        """批处理ASR请求，4核8GB优化版本"""
        if not requests:
            return []
        
        start_time = time.time()
        results = []
        
        try:
            # 4核8GB优化：限制批处理大小
            actual_batch_size = min(len(requests), self.batch_size)
            batch_requests = requests[:actual_batch_size]
            
            # 预处理音频数据
            audio_inputs = []
            for req in batch_requests:
                try:
                    audio_array = self.preprocess_audio(req.audio_data, req.sample_rate)
                    audio_inputs.append(audio_array)
                except Exception as e:
                    logger.error(f"音频预处理失败 {req.session_id}: {e}")
                    # 添加错误结果
                    results.append(ASRResult(
                        session_id=req.session_id,
                        text="",
                        confidence=0.0,
                        language=req.language,
                        timestamp=req.timestamp,
                        processing_time=0.0,
                        cached=False
                    ))
                    continue
            
            if not audio_inputs:
                return results
            
            # 批量推理
            try:
                # 4核8GB优化：使用更保守的推理参数
                batch_results = self.model.generate(
                    input=audio_inputs,
                    cache={},
                    language="zh",
                    use_itn=True,
                    # 4核8GB优化：禁用一些高级功能以节省资源
                    batch_size=len(audio_inputs),
                    # 降低beam_size以提升速度
                    beam_size=self.beam_size
                )
                
                # 处理结果
                processing_time = time.time() - start_time
                
                for i, (req, result) in enumerate(zip(batch_requests, batch_results)):
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('text', '') if isinstance(result[0], dict) else str(result[0])
                        confidence = result[0].get('confidence', 0.8) if isinstance(result[0], dict) else 0.8
                    else:
                        text = str(result) if result else ""
                        confidence = 0.8 if text else 0.0
                    
                    asr_result = ASRResult(
                        session_id=req.session_id,
                        text=text,
                        confidence=confidence,
                        language=req.language,
                        timestamp=req.timestamp,
                        processing_time=processing_time / len(batch_requests),
                        cached=False
                    )
                    results.append(asr_result)
                
            except Exception as e:
                logger.error(f"批量ASR推理失败: {e}")
                # 为所有请求添加错误结果
                for req in batch_requests:
                    results.append(ASRResult(
                        session_id=req.session_id,
                        text="",
                        confidence=0.0,
                        language=req.language,
                        timestamp=req.timestamp,
                        processing_time=time.time() - start_time,
                        cached=False
                    ))
            
            # 更新统计信息
            self.total_requests += len(batch_requests)
            self.total_processing_time += time.time() - start_time
            
            return results
            
        except Exception as e:
            logger.error(f"批处理ASR失败: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """获取处理器统计信息"""
        avg_time = self.total_processing_time / max(self.total_requests, 1)
        cache_hit_rate = self.cache_hits / max(self.total_requests, 1)
        
        return {
            'total_requests': self.total_requests,
            'avg_processing_time': avg_time,
            'cache_hit_rate': cache_hit_rate,
            'model_device': str(self.device),
            'batch_size': self.batch_size,
            'chunk_size': self.chunk_size
        }

class ASRService:
    """4核8GB服务器激进优化的ASR服务，支持批处理、队列管理和缓存优化"""
    
    def __init__(self, batch_size: int = None, max_concurrent: int = None):
        # 从环境变量读取优化参数
        import os
        self.batch_size = batch_size or int(os.getenv('ASR_BATCH_SIZE', '12'))
        self.max_concurrent = max_concurrent or int(os.getenv('ASR_MAX_CONCURRENT', '30'))
        self.enable_fp16 = os.getenv('ASR_ENABLE_FP16', 'true').lower() == 'true'
        self.enable_batch_optimization = os.getenv('ASR_ENABLE_BATCH_OPTIMIZATION', 'true').lower() == 'true'
        self.enable_zero_copy = os.getenv('ASR_ENABLE_ZERO_COPY', 'true').lower() == 'true'
        self.preload_model = os.getenv('ASR_PRELOAD_MODEL', 'true').lower() == 'true'
        self.result_cache_size = int(os.getenv('ASR_RESULT_CACHE_SIZE', '2000'))
        
        logger.info(f"ASR服务激进优化配置 - batch_size: {self.batch_size}, max_concurrent: {self.max_concurrent}")
        self.processor = SenseVoiceProcessor(batch_size=self.batch_size, enable_fp16=self.enable_fp16)
        
        # 4核8GB激进优化：优先级队列，增加队列大小支持更高并发
        queue_size = int(os.getenv('ASR_QUEUE_SIZE', '80'))
        self.high_priority_queue = asyncio.Queue(maxsize=queue_size)
        self.medium_priority_queue = asyncio.Queue(maxsize=queue_size + 20)
        self.low_priority_queue = asyncio.Queue(maxsize=queue_size // 2)
        
        self.redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=4)  # 4核8GB优化：从8降低到4个工作线程
        
        # 4核8GB优化：添加性能监控
        self.performance_stats = {
            'total_requests': 0,
            'avg_latency': 0.0,
            'concurrent_requests': 0,
            'queue_sizes': {'high': 0, 'medium': 0, 'low': 0},
            'memory_usage': 0.0,
            'cpu_usage': 0.0
        }
        
        # 启动批处理任务
        asyncio.create_task(self.batch_processor())
        asyncio.create_task(self.performance_monitor())
    
    async def performance_monitor(self):
        """性能监控任务，4核8GB优化版本"""
        while True:
            await asyncio.sleep(60)  # 每分钟记录一次
            try:
                # 更新队列大小统计
                self.performance_stats['queue_sizes'] = {
                    'high': self.high_priority_queue.qsize(),
                    'medium': self.medium_priority_queue.qsize(),
                    'low': self.low_priority_queue.qsize()
                }
                
                if self.performance_stats['total_requests'] > 0:
                    logger.info(f"ASR性能统计 - 总请求: {self.performance_stats['total_requests']}, "
                              f"平均延迟: {self.performance_stats['avg_latency']:.3f}s, "
                              f"当前并发: {self.performance_stats['concurrent_requests']}, "
                              f"队列: {self.performance_stats['queue_sizes']}")
            except Exception as e:
                logger.warning(f"性能监控错误: {e}")
    
    async def initialize(self):
        """初始化服务"""
        try:
            # 初始化Redis客户端
            self.redis_client = await get_redis_client()
            await self.redis_client.ping()
            logger.info("ASR服务初始化完成")
        except Exception as e:
            logger.error(f"ASR服务初始化失败: {e}")
    
    async def init_redis(self, redis_url: str = "redis://localhost:6379/0"):
        """初始化Redis连接"""
        try:
            self.redis_client = await get_redis_client()
            await self.redis_client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
    
    async def get_next_batch(self) -> List[ASRRequest]:
        """获取下一批请求，优先处理高优先级，4核8GB优化版本"""
        requests = []
        
        # 优先处理高优先级队列
        while len(requests) < self.batch_size and not self.high_priority_queue.empty():
            try:
                request = self.high_priority_queue.get_nowait()
                requests.append(request)
            except asyncio.QueueEmpty:
                break
        
        # 然后处理中优先级队列
        while len(requests) < self.batch_size and not self.medium_priority_queue.empty():
            try:
                request = self.medium_priority_queue.get_nowait()
                requests.append(request)
            except asyncio.QueueEmpty:
                break
        
        # 最后处理低优先级队列
        while len(requests) < self.batch_size and not self.low_priority_queue.empty():
            try:
                request = self.low_priority_queue.get_nowait()
                requests.append(request)
            except asyncio.QueueEmpty:
                break
        
        return requests
    
    async def batch_processor(self):
        """批处理任务，4核8GB优化版本"""
        while True:
            try:
                # 获取批次请求
                requests = await self.get_next_batch()
                
                if not requests:
                    # 如果没有请求，等待一段时间
                    await asyncio.sleep(0.02)  # 4核8GB优化：稍微增加等待时间以减少CPU占用
                    continue
                
                # 更新并发统计
                self.performance_stats['concurrent_requests'] = len(requests)
                
                # 处理批次
                start_time = time.time()
                results = await self.processor.process_batch(requests)
                processing_time = time.time() - start_time
                
                # 缓存结果
                for result in results:
                    await self.cache_result(result)
                
                # 更新统计
                self.performance_stats['total_requests'] += len(requests)
                self.performance_stats['avg_latency'] = (
                    self.performance_stats['avg_latency'] * (self.performance_stats['total_requests'] - len(requests)) + 
                    processing_time
                ) / self.performance_stats['total_requests']
                
                # 4核8GB优化：记录批处理性能
                if len(requests) > 0:
                    logger.debug(f"批处理完成: {len(requests)}个请求, 耗时: {processing_time:.3f}s, "
                               f"平均: {processing_time/len(requests):.3f}s/请求")
                
            except Exception as e:
                logger.error(f"批处理任务错误: {e}")
                await asyncio.sleep(0.1)
    
    async def cache_result(self, result: ASRResult):
        """缓存ASR结果，4核8GB优化版本"""
        if self.redis_client:
            try:
                # 使用音频哈希作为缓存键，减少缓存大小
                cache_key = f"asr:{result.session_id[:8]}:{int(result.timestamp)}"  # 简化缓存键
                cache_data = {
                    'text': result.text,
                    'confidence': result.confidence,
                    'language': result.language,
                    'processing_time': result.processing_time
                }
                # 4核8GB优化：减少缓存时间到15分钟
                await self.redis_client.setex(cache_key, 900, json.dumps(cache_data))
            except Exception as e:
                logger.warning(f"缓存ASR结果失败: {e}")
    
    async def get_cached_result(self, request: ASRRequest) -> Optional[ASRResult]:
        """从缓存获取ASR结果，4核8GB优化版本"""
        if self.redis_client:
            try:
                # 使用音频哈希作为缓存键
                audio_hash = self.processor.generate_audio_hash(request.audio_data)
                cache_key = f"asr_hash:{audio_hash[:16]}"  # 4核8GB优化：缩短哈希长度
                
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    self.processor.cache_hits += 1
                    return ASRResult(
                        session_id=request.session_id,
                        text=data['text'],
                        confidence=data['confidence'],
                        language=data['language'],
                        timestamp=request.timestamp,
                        processing_time=data['processing_time'],
                        cached=True
                    )
            except Exception as e:
                logger.warning(f"获取缓存ASR结果失败: {e}")
        return None
    
    async def add_request(self, request: ASRRequest):
        """添加请求到相应的优先级队列，4核8GB优化版本"""
        try:
            if request.priority == 1:
                await self.high_priority_queue.put(request)
            elif request.priority == 2:
                await self.medium_priority_queue.put(request)
            else:
                await self.low_priority_queue.put(request)
        except asyncio.QueueFull:
            logger.warning(f"队列已满，丢弃请求: {request.session_id}")
            raise HTTPException(status_code=503, detail="服务器繁忙，请稍后重试")

# FastAPI应用
app = FastAPI(title="ASR Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局ASR服务实例 - 4核8GB激进优化配置（从环境变量读取）
asr_service = ASRService()

@app.on_event("startup")
async def startup_event():
    await asr_service.initialize()

@app.post("/asr/recognize")
async def recognize_speech(
    session_id: str,
    audio_data: str,  # base64编码的音频数据
    sample_rate: int = 16000,
    language: str = "zh",
    priority: int = 2,
    timestamp: float = 0.0,
    background_tasks: BackgroundTasks = None
):
    """语音识别API，4核8GB优化版本"""
    try:
        # 解码音频数据
        audio_bytes = base64.b64decode(audio_data)
        
        # 4核8GB优化：限制音频数据大小
        max_audio_size = 1024 * 1024  # 1MB限制
        if len(audio_bytes) > max_audio_size:
            return {
                "session_id": session_id,
                "text": "",
                "confidence": 0.0,
                "error": "音频数据过大，请压缩后重试"
            }
        
        # 创建ASR请求
        request = ASRRequest(
            session_id=session_id,
            audio_data=audio_bytes,
            sample_rate=sample_rate,
            language=language,
            timestamp=timestamp,
            priority=priority
        )
        
        # 检查缓存
        cached_result = await asr_service.get_cached_result(request)
        if cached_result:
            return {
                "session_id": cached_result.session_id,
                "text": cached_result.text,
                "confidence": cached_result.confidence,
                "language": cached_result.language,
                "timestamp": cached_result.timestamp,
                "processing_time": cached_result.processing_time,
                "cached": True
            }
        
        # 添加到处理队列
        await asr_service.add_request(request)
        
        return {
            "session_id": session_id,
            "status": "processing",
            "message": "请求已添加到处理队列"
        }
        
    except Exception as e:
        logger.error(f"ASR识别失败: {e}")
        return {
            "session_id": session_id,
            "text": "",
            "confidence": 0.0,
            "language": language,
            "timestamp": timestamp,
            "processing_time": 0.0,
            "cached": False,
            "error": str(e)
        }

@app.get("/asr/stats")
async def get_stats():
    """获取ASR服务统计信息"""
    processor_stats = asr_service.processor.get_stats()
    service_stats = asr_service.performance_stats
    
    return {
        "processor": processor_stats,
        "service": service_stats,
        "queues": {
            "high": asr_service.high_priority_queue.qsize(),
            "medium": asr_service.medium_priority_queue.qsize(),
            "low": asr_service.low_priority_queue.qsize()
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "asr", "optimization": "4core_8gb"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)