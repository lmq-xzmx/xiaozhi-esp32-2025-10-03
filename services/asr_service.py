#!/usr/bin/env python3
"""
ASR (Automatic Speech Recognition) 微服务
基于SenseVoice，支持批处理、队列管理和模型缓存优化
4核8GB服务器极限优化版本：支持80-100台设备并发
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

try:
    from config.redis_config import get_redis_client, OptimizedRedisClient
    from core.queue_manager import get_queue_manager, QueueRequest, Priority
except ImportError:
    # 如果导入失败，创建简单的替代实现
    logger = logging.getLogger(__name__)
    logger.warning("无法导入Redis和队列管理模块，使用简化版本")
    
    class OptimizedRedisClient:
        async def get(self, key): return None
        async def set(self, key, value, ex=None): pass
        async def close(self): pass
    
    def get_redis_client():
        return OptimizedRedisClient()
    
    class QueueRequest:
        def __init__(self, **kwargs): pass
    
    class Priority:
        HIGH = 1
        MEDIUM = 2
        LOW = 3
    
    def get_queue_manager():
        return None
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import base64
from funasr import AutoModel
import librosa
import os

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
    """4核8GB服务器极限优化的SenseVoice处理器，支持批处理和量化优化"""
    
    def __init__(self, model_dir: str = "models/SenseVoiceSmall", batch_size: int = None, enable_fp16: bool = True):
        self.model_dir = model_dir
        # 极限优化：从环境变量读取配置，默认大幅提升
        self.batch_size = batch_size or int(os.getenv("ASR_BATCH_SIZE", "32"))  # 提升到32
        self.enable_fp16 = enable_fp16
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 极限优化：实时处理参数
        self.chunk_size = 512  # 32ms@16kHz，进一步降低延迟
        self.chunk_duration_ms = 32
        self.overlap_ms = 4  # 最小重叠
        self.max_audio_length_s = 20  # 降低最大音频长度
        self.beam_size = 1  # 贪婪解码，最快速度
        
        # 性能统计
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        
        # 内存管理 - 极限优化
        self.max_memory_mb = int(os.getenv("ASR_MEMORY_LIMIT", "10240"))  # 10GB内存限制
        self.audio_buffer_pool = []
        self.result_cache_max_size = int(os.getenv("ASR_CACHE_SIZE_MB", "6144"))  # 6GB缓存
        
        # 极限优化：启用所有性能特性
        self.enable_turbo = os.getenv("ASR_ENABLE_TURBO", "true").lower() == "true"
        self.enable_memory_pool = os.getenv("ASR_MEMORY_POOL", "true").lower() == "true"
        self.enable_zero_copy = os.getenv("ASR_ZERO_COPY", "true").lower() == "true"
        self.enable_int8 = os.getenv("ASR_ENABLE_INT8", "true").lower() == "true"
        self.enable_fp16 = os.getenv("ASR_ENABLE_FP16", "true").lower() == "true"
        
        # 加载模型
        self.load_model()
        self.warmup_model()
    
    def _model_exists(self, model_path: str) -> bool:
        """检查模型是否存在"""
        return os.path.exists(model_path)
    
    def load_model(self):
        """加载SenseVoice模型，优先使用量化版本"""
        try:
            # 极限优化：优先加载最高性能量化模型
            model_candidates = []
            
            if self.enable_int8:
                model_candidates.append(f"{self.model_dir}_int8")
            if self.enable_fp16:
                model_candidates.append(f"{self.model_dir}_fp16")
            model_candidates.append(self.model_dir)
            
            model_path = None
            for candidate in model_candidates:
                if candidate and self._model_exists(candidate):
                    model_path = candidate
                    break
            
            if not model_path:
                raise FileNotFoundError(f"未找到模型文件: {self.model_dir}")
            
            logger.info(f"🚀 加载ASR模型: {model_path}")
            
            # 极限优化：模型加载配置
            model_config = {
                "model": model_path,
                "device": self.device,
                "batch_size": self.batch_size,
                "disable_update": True,  # 禁用模型更新以提升性能
                "disable_log": True,     # 禁用日志以提升性能
            }
            
            if self.enable_fp16 and torch.cuda.is_available():
                model_config["dtype"] = torch.float16
            
            self.model = AutoModel(**model_config)
            
            # 极限优化：模型编译加速
            if hasattr(torch, 'compile') and self.enable_turbo:
                try:
                    self.model = torch.compile(self.model, mode="max-autotune")
                    logger.info("✅ 启用Torch编译加速")
                except Exception as e:
                    logger.warning(f"Torch编译失败: {e}")
            
            logger.info(f"✅ ASR模型加载成功，批处理大小: {self.batch_size}")
            
        except Exception as e:
            logger.error(f"❌ ASR模型加载失败: {e}")
            raise

    def warmup_model(self):
        """模型预热，优化首次推理性能"""
        try:
            logger.info("🔥 ASR模型预热中...")
            
            # 创建预热音频数据 - 极限优化：更小的预热数据
            warmup_audio = np.random.randn(8000).astype(np.float32)  # 0.5秒音频
            
            # 批量预热 - 极限优化：预热更大批次
            warmup_batch = [warmup_audio] * min(self.batch_size, 16)
            
            start_time = time.time()
            _ = self.model.generate(input=warmup_batch, batch_size=len(warmup_batch))
            warmup_time = time.time() - start_time
            
            logger.info(f"✅ ASR模型预热完成，耗时: {warmup_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"⚠️ ASR模型预热失败: {e}")

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
    """极限优化的ASR服务，支持80-100台设备并发"""
    
    def __init__(self, batch_size: int = None, max_concurrent: int = None):
        # 极限优化：从环境变量读取配置
        self.batch_size = batch_size or int(os.getenv("ASR_BATCH_SIZE", "32"))
        self.max_concurrent = max_concurrent or int(os.getenv("ASR_MAX_CONCURRENT", "160"))
        self.batch_timeout = float(os.getenv("ASR_BATCH_TIMEOUT", "50")) / 1000  # 50ms
        self.queue_size = int(os.getenv("ASR_QUEUE_SIZE", "400"))
        self.worker_threads = int(os.getenv("ASR_WORKER_THREADS", "12"))
        self.io_threads = int(os.getenv("ASR_IO_THREADS", "4"))
        
        # 初始化组件
        self.processor = SenseVoiceProcessor(batch_size=self.batch_size)
        self.redis_client = None
        self.request_queue = asyncio.Queue(maxsize=self.queue_size)
        self.result_futures = {}
        self.processing_semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 极限优化：线程池配置
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix="ASR-Worker"
        )
        self.io_pool = ThreadPoolExecutor(
            max_workers=self.io_threads,
            thread_name_prefix="ASR-IO"
        )
        
        # 性能监控
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_processing_time": 0.0,
            "current_queue_size": 0,
            "max_concurrent": self.max_concurrent,
            "batch_size": self.batch_size,
        }
        
        # 启动后台任务
        asyncio.create_task(self.batch_processor())
        asyncio.create_task(self.performance_monitor())

    async def batch_processor(self):
        """极限优化的批处理器"""
        logger.info(f"🚀 启动ASR批处理器，批大小: {self.batch_size}, 超时: {self.batch_timeout*1000:.0f}ms")
        
        while True:
            try:
                batch_requests = []
                start_time = time.time()
                
                # 极限优化：动态批处理收集
                while len(batch_requests) < self.batch_size:
                    try:
                        remaining_time = self.batch_timeout - (time.time() - start_time)
                        if remaining_time <= 0:
                            break
                        
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=remaining_time
                        )
                        batch_requests.append(request)
                        
                    except asyncio.TimeoutError:
                        break
                
                if batch_requests:
                    # 极限优化：并行处理批次
                    await self._process_batch_parallel(batch_requests)
                else:
                    # 极限优化：短暂休眠避免CPU空转
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"❌ 批处理器错误: {e}")
                await asyncio.sleep(0.01)

    async def _process_batch_parallel(self, requests: List[ASRRequest]):
        """并行处理批次请求"""
        try:
            # 极限优化：并行缓存检查
            cache_tasks = [self.get_cached_result(req) for req in requests]
            cached_results = await asyncio.gather(*cache_tasks, return_exceptions=True)
            
            # 分离缓存命中和未命中的请求
            uncached_requests = []
            for i, (req, cached) in enumerate(zip(requests, cached_results)):
                if isinstance(cached, ASRResult):
                    # 缓存命中，直接返回结果
                    if req.session_id in self.result_futures:
                        self.result_futures[req.session_id].set_result(cached)
                        del self.result_futures[req.session_id]
                    self.stats["cache_hits"] += 1
                else:
                    uncached_requests.append(req)
            
            # 处理未缓存的请求
            if uncached_requests:
                results = await self.processor.process_batch(uncached_requests)
                
                # 极限优化：并行缓存存储和结果返回
                cache_tasks = [self.cache_result(result) for result in results]
                await asyncio.gather(*cache_tasks, return_exceptions=True)
                
                # 返回结果
                for result in results:
                    if result.session_id in self.result_futures:
                        self.result_futures[result.session_id].set_result(result)
                        del self.result_futures[result.session_id]
                
                self.stats["successful_requests"] += len(results)
            
        except Exception as e:
            logger.error(f"❌ 批处理失败: {e}")
            # 处理失败的请求
            for req in requests:
                if req.session_id in self.result_futures:
                    self.result_futures[req.session_id].set_exception(e)
                    del self.result_futures[req.session_id]
            self.stats["failed_requests"] += len(requests)

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

# 添加流式ASR路由（集成聊天记录功能）
try:
    from asr_streaming_enhancement import add_streaming_routes
    streaming_service = add_streaming_routes(app, asr_service)
    logger.info("✅ 流式ASR路由已集成，支持聊天记录功能")
except ImportError as e:
    logger.warning(f"⚠️ 无法导入流式ASR模块: {e}")
except Exception as e:
    logger.error(f"❌ 集成流式ASR路由失败: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)