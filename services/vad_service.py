#!/usr/bin/env python3
"""
VAD (Voice Activity Detection) 微服务
基于SileroVAD，支持批处理、队列管理和ONNX Runtime优化
P0优化版本：支持100台设备并发
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import redis.asyncio as redis
from config.redis_config import get_redis_client, OptimizedRedisClient
from core.queue_manager import get_queue_manager, QueueRequest, Priority
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import base64

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VADRequest:
    session_id: str
    audio_data: bytes
    sample_rate: int = 16000
    timestamp: float = 0.0

@dataclass
class VADResult:
    session_id: str
    is_speech: bool
    confidence: float
    timestamp: float
    processing_time: float

class SileroVADProcessor:
    """P0优化的SileroVAD处理器，支持批处理和ONNX Runtime优化"""
    
    def __init__(self, model_path: str = "silero_vad.onnx", batch_size: int = 32):
        # 批处理配置 - P0优化：从16提升到32
        self.batch_size = batch_size
        self.model = None
        self.onnx_session = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_onnx = model_path.endswith('.onnx')
        
        # P0优化：实时处理参数
        self.chunk_size = 512  # 32ms@16kHz，极低延迟
        self.chunk_overlap = 64  # 4ms重叠
        self.speech_threshold = 0.6  # 语音检测阈值
        
        self.load_model(model_path)
        
        # 性能统计
        self.total_requests = 0
        self.total_processing_time = 0.0
        
    def load_model(self, model_path: str):
        """加载SileroVAD模型，优先使用ONNX Runtime"""
        if self.use_onnx:
            try:
                import onnxruntime as ort
                
                # P0优化：ONNX Runtime配置
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                options.intra_op_num_threads = 4  # P0优化：增加到4个线程
                options.inter_op_num_threads = 4  # P0优化：增加到4个线程
                options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                options.enable_cpu_mem_arena = True
                options.enable_mem_pattern = True  # P0优化：启用内存模式优化
                options.enable_mem_reuse = True    # P0优化：启用内存重用
                
                # 选择执行提供者
                providers = []
                if torch.cuda.is_available():
                    providers.append(('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }))
                providers.append(('CPUExecutionProvider', {
                    'arena_extend_strategy': 'kSameAsRequested',
                    'enable_cpu_mem_arena': True,
                }))

                self.onnx_session = ort.InferenceSession(
                    model_path, 
                    options, 
                    providers=providers
                )
                logger.info(f"ONNX SileroVAD模型加载成功，提供者: {self.onnx_session.get_providers()}")
                return
            except Exception as e:
                logger.warning(f"ONNX模型加载失败: {e}，回退到PyTorch模型")
        
        try:
            # 尝试加载PyTorch JIT模型
            self.model = torch.jit.load(model_path.replace('.onnx', '.pt'), map_location=self.device)
            self.model.eval()
            # JIT优化
            if hasattr(torch.jit, 'optimize_for_inference'):
                self.model = torch.jit.optimize_for_inference(self.model)
            logger.info(f"PyTorch SileroVAD模型加载成功，设备: {self.device}")
        except Exception as e:
            logger.error(f"PyTorch模型加载失败: {e}")
            # 最后回退到预训练模型
            self.model = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=True,
                                      onnx=False)
            self.model.to(self.device)
            logger.info("使用预训练SileroVAD模型")
    
    def preprocess_audio(self, audio_data: bytes, sample_rate: int = 16000) -> torch.Tensor:
        """预处理音频数据"""
        # 将bytes转换为numpy数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 重采样到16kHz
        if sample_rate != 16000:
            # 简单线性插值重采样
            ratio = 16000 / sample_rate
            new_length = int(len(audio_np) * ratio)
            audio_np = np.interp(np.linspace(0, len(audio_np), new_length), 
                               np.arange(len(audio_np)), audio_np)
        
        # 转换为tensor
        audio_tensor = torch.from_numpy(audio_np).to(self.device)
        return audio_tensor
    
    async def process_batch(self, requests: List[VADRequest]) -> List[VADResult]:
        """批处理VAD请求"""
        start_time = time.time()
        results = []
        
        try:
            # 预处理所有音频
            audio_tensors = []
            for request in requests:
                audio_tensor = self.preprocess_audio(request.audio_data, request.sample_rate)
                audio_tensors.append(audio_tensor)
            
            # 批处理推理
            if self.onnx_session:
                # ONNX推理
                for i, (request, audio_tensor) in enumerate(zip(requests, audio_tensors)):
                    audio_np = audio_tensor.cpu().numpy()
                    speech_prob = self.onnx_session.run(None, {'input': audio_np})[0]
                    
                    is_speech = float(speech_prob) > self.speech_threshold
                    confidence = float(speech_prob)
                    
                    results.append(VADResult(
                        session_id=request.session_id,
                        is_speech=is_speech,
                        confidence=confidence,
                        timestamp=request.timestamp,
                        processing_time=time.time() - start_time
                    ))
            else:
                # PyTorch推理
                with torch.no_grad():
                    for i, (request, audio_tensor) in enumerate(zip(requests, audio_tensors)):
                        speech_prob = self.model(audio_tensor, 16000)
                        
                        is_speech = float(speech_prob) > self.speech_threshold
                        confidence = float(speech_prob)
                        
                        results.append(VADResult(
                            session_id=request.session_id,
                            is_speech=is_speech,
                            confidence=confidence,
                            timestamp=request.timestamp,
                            processing_time=time.time() - start_time
                        ))
            
            # 更新统计
            self.total_requests += len(requests)
            self.total_processing_time += time.time() - start_time
            
        except Exception as e:
            logger.error(f"批处理VAD失败: {e}")
            # 返回默认结果
            for request in requests:
                results.append(VADResult(
                    session_id=request.session_id,
                    is_speech=False,
                    confidence=0.0,
                    timestamp=request.timestamp,
                    processing_time=time.time() - start_time
                ))
        
        return results
    
    def get_stats(self) -> Dict:
        """获取性能统计"""
        avg_time = self.total_processing_time / max(self.total_requests, 1)
        return {
            'total_requests': self.total_requests,
            'avg_processing_time': avg_time,
            'batch_size': self.batch_size,
            'device': str(self.device),
            'use_onnx': self.use_onnx
        }

class VADService:
    """P0优化的VAD微服务主类"""
    
    def __init__(self, batch_size: int = 32, max_concurrent: int = 48):
        self.batch_size = batch_size  # P0优化：从16提升到32
        self.max_concurrent = max_concurrent  # P0优化：从24提升到48
        self.processor = SileroVADProcessor(batch_size=batch_size)
        self.request_queue = asyncio.Queue(maxsize=200)  # P0优化：添加队列大小限制
        self.redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=6)  # P0优化：从4提升到6个工作线程
        
        # P0优化：添加性能监控
        self.performance_stats = {
            'total_requests': 0,
            'avg_latency': 0.0,
            'concurrent_requests': 0,
            'queue_size': 0
        }
        
        # 启动批处理任务
        asyncio.create_task(self.batch_processor())
    
    async def initialize(self):
        """初始化服务"""
        try:
            # 初始化优化的Redis客户端
            self.redis_client = await get_redis_client()
            await self.redis_client.health_check()
            logger.info("VAD服务初始化完成")
        except Exception as e:
            logger.error(f"VAD服务初始化失败: {e}")
    
    async def init_redis(self, redis_url: str = "redis://localhost:6379"):
        """初始化Redis连接"""
        try:
            self.redis_client = await get_redis_client()
            await self.redis_client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
    
    async def batch_processor(self):
        """批处理任务"""
        while True:
            try:
                requests = []
                # 收集批次
                try:
                    # 等待第一个请求
                    first_request = await asyncio.wait_for(self.request_queue.get(), timeout=0.1)
                    requests.append(first_request)
                    
                    # 收集更多请求直到批次满或超时
                    start_time = time.time()
                    while len(requests) < self.batch_size and (time.time() - start_time) < 0.05:
                        try:
                            request = await asyncio.wait_for(self.request_queue.get(), timeout=0.01)
                            requests.append(request)
                        except asyncio.TimeoutError:
                            break
                
                except asyncio.TimeoutError:
                    continue
                
                if requests:
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
    
    async def cache_result(self, result: VADResult):
        """缓存VAD结果"""
        if self.redis_client:
            try:
                cache_key = f"vad:{result.session_id}:{result.timestamp}"
                cache_data = {
                    'is_speech': result.is_speech,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time
                }
                await self.redis_client.setex(cache_key, 300, json.dumps(cache_data))  # 5分钟缓存
            except Exception as e:
                logger.warning(f"缓存VAD结果失败: {e}")
    
    async def process_vad_request(self, request: VADRequest) -> VADResult:
        """处理VAD请求"""
        # 检查缓存
        cached_result = await self.get_cached_result(request.session_id, request.timestamp)
        if cached_result:
            return cached_result
        
        # 添加到队列
        await self.request_queue.put(request)
        
        # 等待处理结果（简化版本，实际应该使用更复杂的结果匹配机制）
        await asyncio.sleep(0.1)  # 等待批处理
        
        # 从缓存获取结果
        return await self.get_cached_result(request.session_id, request.timestamp)
    
    async def get_cached_result(self, session_id: str, timestamp: float) -> Optional[VADResult]:
        """从缓存获取VAD结果"""
        if self.redis_client:
            try:
                cache_key = f"vad:{session_id}:{timestamp}"
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    data = json.loads(cached_data)
                    return VADResult(
                        session_id=session_id,
                        is_speech=data['is_speech'],
                        confidence=data['confidence'],
                        timestamp=timestamp,
                        processing_time=data['processing_time']
                    )
            except Exception as e:
                logger.warning(f"获取缓存VAD结果失败: {e}")
        return None

# FastAPI应用
app = FastAPI(title="VAD Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局VAD服务实例
vad_service = VADService(batch_size=32, max_concurrent=48)  # P0优化：应用新的批处理和并发参数

@app.on_event("startup")
async def startup_event():
    await vad_service.initialize()

@app.post("/vad/detect")
async def detect_voice_activity(
    session_id: str,
    audio_data: str,  # base64编码的音频数据
    sample_rate: int = 16000,
    timestamp: float = 0.0
):
    """检测语音活动"""
    try:
        # 解码音频数据
        audio_bytes = base64.b64decode(audio_data)
        
        # 创建VAD请求
        request = VADRequest(
            session_id=session_id,
            audio_data=audio_bytes,
            sample_rate=sample_rate,
            timestamp=timestamp
        )
        
        # 处理请求
        result = await vad_service.process_vad_request(request)
        
        return {
            "session_id": result.session_id,
            "is_speech": result.is_speech,
            "confidence": result.confidence,
            "timestamp": result.timestamp,
            "processing_time": result.processing_time
        }
    except Exception as e:
        logger.error(f"VAD检测失败: {e}")
        return {
            "session_id": session_id,
            "is_speech": False,
            "confidence": 0.0,
            "timestamp": timestamp,
            "processing_time": 0.0,
            "error": str(e)
        }

@app.get("/vad/stats")
async def get_stats():
    """获取VAD服务统计信息"""
    return vad_service.processor.get_stats()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "vad"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004, workers=1)