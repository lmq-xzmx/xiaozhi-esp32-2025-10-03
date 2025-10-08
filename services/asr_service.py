#!/usr/bin/env python3
"""
ASR (Automatic Speech Recognition) 微服务
基于SenseVoice，支持批处理、队列管理和模型缓存优化
P0优化版本：支持100台设备并发
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
    """P0优化的SenseVoice处理器，支持批处理和量化优化"""
    
    def __init__(self, model_dir: str = "models/SenseVoiceSmall", batch_size: int = 16, enable_fp16: bool = True):
        self.model_dir = model_dir
        # 批处理配置 - P0优化：从8提升到16
        self.batch_size = batch_size
        self.enable_fp16 = enable_fp16
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # P0优化：实时处理参数
        self.chunk_size = 800  # 50ms@16kHz
        self.chunk_duration_ms = 50
        self.overlap_ms = 10
        self.max_audio_length_s = 30
        self.beam_size = 1  # 贪婪解码，最快速度
        
        # 性能统计
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        
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
            # P0优化：优先加载量化模型
            model_candidates = [
                f"{self.model_dir}_int8",  # P0优化：优先INT8量化
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
            
            # P0优化：模型加载配置
            model_kwargs = {
                "vad_model": "fsmn-vad",
                "vad_kwargs": {
                    "max_single_segment_time": 30000,
                    "max_start_silence_time": 3000,
                    "max_end_silence_time": 800,
                },
                "device": self.device,
                "ncpu": 4,  # 增加CPU线程数
                "batch_size": self.batch_size,
            }
            
            # GPU特定配置
            if torch.cuda.is_available():
                model_kwargs.update({
                    "device_id": 0,
                    "precision": "fp16" if self.enable_fp16 else "fp32",
                })
            
            self.model = AutoModel(
                model=model_path,
                **model_kwargs
            )
            
            # P0优化：FP16转换
            if self.enable_fp16 and torch.cuda.is_available():
                try:
                    # 启用FP16优化
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'half'):
                        self.model.model = self.model.model.half()
                        logger.info("模型已转换为FP16精度")
                    
                    # 启用TensorFloat-32优化
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                        torch.backends.cuda.matmul.allow_tf32 = True
                        
                except Exception as e:
                    logger.warning(f"FP16转换失败，使用FP32: {e}")
            
            logger.info(f"SenseVoice模型加载成功，设备: {self.device}, FP16: {self.enable_fp16}, 批处理: {self.batch_size}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 回退到原始模型
            try:
                self.model = AutoModel(
                    model=self.model_dir,
                    vad_model="fsmn-vad",
                    vad_kwargs={"max_single_segment_time": 30000},
                    device=self.device
                )
                logger.info("回退到原始模型加载成功")
            except Exception as e2:
                logger.error(f"回退模型加载也失败: {e2}")
                raise
    
    def warmup_model(self):
        """模型预热"""
        try:
            # 生成虚拟音频数据进行预热
            dummy_audio = np.random.randn(16000).astype(np.float32)  # 1秒音频
            _ = self.model.generate(input=dummy_audio, cache={}, language="zh")
            logger.info("模型预热完成")
        except Exception as e:
            logger.warning(f"模型预热失败: {e}")
    
    def preprocess_audio(self, audio_data: bytes, sample_rate: int = 16000) -> np.ndarray:
        """预处理音频数据"""
        # 将bytes转换为numpy数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 重采样到16kHz
        if sample_rate != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)
        
        return audio_np
    
    def generate_audio_hash(self, audio_data: bytes) -> str:
        """生成音频数据的哈希值用于缓存"""
        return hashlib.md5(audio_data).hexdigest()
    
    async def process_batch(self, requests: List[ASRRequest]) -> List[ASRResult]:
        """批处理ASR请求"""
        start_time = time.time()
        results = []
        
        try:
            # 预处理所有音频
            audio_inputs = []
            for request in requests:
                audio_np = self.preprocess_audio(request.audio_data, request.sample_rate)
                audio_inputs.append(audio_np)
            
            # 批处理推理
            batch_results = []
            for audio_np in audio_inputs:
                try:
                    # 使用SenseVoice进行推理
                    result = self.model.generate(
                        input=audio_np,
                        cache={},
                        language="zh",
                        use_itn=True,
                        batch_size=1
                    )
                    
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get('text', '')
                        confidence = result[0].get('confidence', 0.8)
                    else:
                        text = str(result) if result else ''
                        confidence = 0.8
                    
                    batch_results.append((text, confidence))
                    
                except Exception as e:
                    logger.warning(f"单个音频推理失败: {e}")
                    batch_results.append(('', 0.0))
            
            # 构建结果
            processing_time = time.time() - start_time
            for i, request in enumerate(requests):
                text, confidence = batch_results[i]
                
                result = ASRResult(
                    session_id=request.session_id,
                    text=text,
                    confidence=confidence,
                    language=request.language,
                    timestamp=request.timestamp,
                    processing_time=processing_time / len(requests),
                    cached=False
                )
                results.append(result)
            
            # 更新统计
            self.total_requests += len(requests)
            self.total_processing_time += processing_time
            
            logger.info(f"批处理完成: {len(requests)}个请求, 耗时: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"批处理ASR失败: {e}")
            # 返回默认结果
            for request in requests:
                results.append(ASRResult(
                    session_id=request.session_id,
                    text='',
                    confidence=0.0,
                    language=request.language,
                    timestamp=request.timestamp,
                    processing_time=time.time() - start_time,
                    cached=False
                ))
        
        return results
    
    def get_stats(self) -> Dict:
        """获取性能统计"""
        avg_time = self.total_processing_time / max(self.total_requests, 1)
        cache_rate = self.cache_hits / max(self.total_requests, 1) * 100
        return {
            'total_requests': self.total_requests,
            'avg_processing_time': avg_time,
            'cache_hit_rate': cache_rate,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'enable_fp16': self.enable_fp16
        }

class ASRService:
    """P0优化的ASR服务，支持批处理、队列管理和缓存优化"""
    
    def __init__(self, batch_size: int = 16, max_concurrent: int = 40):
        self.batch_size = batch_size  # P0优化：从8提升到16
        self.max_concurrent = max_concurrent  # P0优化：从16提升到40
        self.processor = SenseVoiceProcessor(batch_size=batch_size, enable_fp16=True)
        
        # P0优化：优先级队列，添加队列大小限制
        self.high_priority_queue = asyncio.Queue(maxsize=50)
        self.medium_priority_queue = asyncio.Queue(maxsize=75)
        self.low_priority_queue = asyncio.Queue(maxsize=25)
        
        self.redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=8)  # P0优化：从4提升到8个工作线程
        
        # P0优化：添加性能监控
        self.performance_stats = {
            'total_requests': 0,
            'avg_latency': 0.0,
            'concurrent_requests': 0,
            'queue_sizes': {'high': 0, 'medium': 0, 'low': 0}
        }
        
        # 启动批处理任务
        asyncio.create_task(self.batch_processor())
        asyncio.create_task(self.performance_monitor())
    
    async def performance_monitor(self):
        """性能监控任务"""
        while True:
            await asyncio.sleep(60)  # 每分钟记录一次
            if self.performance_stats['total_requests'] > 0:
                logger.info(f"ASR性能统计 - 总请求: {self.performance_stats['total_requests']}, "
                          f"平均延迟: {self.performance_stats['avg_latency']:.3f}s, "
                          f"当前并发: {self.performance_stats['concurrent_requests']}")
    
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
        """获取下一批请求，优先处理高优先级"""
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
        """批处理任务"""
        while True:
            try:
                # 获取批次请求
                requests = await self.get_next_batch()
                
                if not requests:
                    # 如果没有请求，等待一段时间
                    await asyncio.sleep(0.01)
                    continue
                
                # 处理批次
                results = await self.processor.process_batch(requests)
                
                # 缓存结果
                for result in results:
                    await self.cache_result(result)
                
                # 更新统计
                self.performance_stats['total_requests'] += len(requests)
                
            except Exception as e:
                logger.error(f"批处理任务错误: {e}")
                await asyncio.sleep(0.1)
    
    async def cache_result(self, result: ASRResult):
        """缓存ASR结果"""
        if self.redis_client:
            try:
                cache_key = f"asr:{result.session_id}:{result.timestamp}"
                cache_data = {
                    'text': result.text,
                    'confidence': result.confidence,
                    'language': result.language,
                    'processing_time': result.processing_time
                }
                await self.redis_client.setex(cache_key, 1800, json.dumps(cache_data))  # 30分钟缓存
            except Exception as e:
                logger.warning(f"缓存ASR结果失败: {e}")
    
    async def get_cached_result(self, request: ASRRequest) -> Optional[ASRResult]:
        """从缓存获取ASR结果"""
        if self.redis_client:
            try:
                # 使用音频哈希作为缓存键
                audio_hash = self.processor.generate_audio_hash(request.audio_data)
                cache_key = f"asr_hash:{audio_hash}"
                
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
        """添加请求到相应的优先级队列"""
        if request.priority == 1:
            await self.high_priority_queue.put(request)
        elif request.priority == 2:
            await self.medium_priority_queue.put(request)
        else:
            await self.low_priority_queue.put(request)

# FastAPI应用
app = FastAPI(title="ASR Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局ASR服务实例
asr_service = ASRService(batch_size=16, max_concurrent=40)  # P0优化：应用新的批处理和并发参数

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
    """语音识别API"""
    try:
        # 解码音频数据
        audio_bytes = base64.b64decode(audio_data)
        
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
    return asr_service.processor.get_stats()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "asr"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)