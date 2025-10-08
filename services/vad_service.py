#!/usr/bin/env python3
"""
VAD (Voice Activity Detection) 微服务
支持批处理和并发优化，适用于100台设备并发
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
<<<<<<< HEAD
        self.batch_size = batch_size  # 从16提升到32
=======
        self.batch_size = batch_size  # P0优化：从16提升到32
>>>>>>> 2aa4dbc2af6d2f1b4c89b11ac5f75eb495cde788
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
<<<<<<< HEAD
                options.intra_op_num_threads = 4  # 从2提升到4个线程
                options.inter_op_num_threads = 4  # 从2提升到4个线程
                options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                options.enable_cpu_mem_arena = True
                options.enable_mem_pattern = True
                options.enable_mem_reuse = True
=======
                options.intra_op_num_threads = 4  # P0优化：增加到4个线程
                options.inter_op_num_threads = 4  # P0优化：增加到4个线程
                options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                options.enable_mem_pattern = True  # P0优化：启用内存模式优化
                options.enable_mem_reuse = True    # P0优化：启用内存重用
>>>>>>> 2aa4dbc2af6d2f1b4c89b11ac5f75eb495cde788
                
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
            # 使用torch.jit.load加载模型
            self.model = torch.jit.load(model_path.replace('.onnx', '.pt'), map_location=self.device)
            self.model.eval()
            # 启用JIT优化
            if hasattr(torch.jit, 'optimize_for_inference'):
                self.model = torch.jit.optimize_for_inference(self.model)
            logger.info(f"PyTorch SileroVAD模型加载成功，设备: {self.device}")
        except Exception as e:
            logger.error(f"PyTorch模型加载失败: {e}")
            # 使用预训练模型
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
        
        # 重采样到16kHz（如果需要）
        if sample_rate != 16000:
            # 简单的重采样（生产环境建议使用librosa）
            ratio = 16000 / sample_rate
            new_length = int(len(audio_np) * ratio)
            audio_np = np.interp(np.linspace(0, len(audio_np), new_length), 
                               np.arange(len(audio_np)), audio_np)
        
        # 转换为torch tensor
        audio_tensor = torch.from_numpy(audio_np).to(self.device)
        return audio_tensor
    
    async def process_batch(self, requests: List[VADRequest]) -> List[VADResult]:
        """批处理VAD请求，支持ONNX Runtime优化"""
        start_time = time.time()
        results = []
        
        try:
            # 预处理所有音频
            audio_tensors = []
            audio_arrays = []
            for req in requests:
                if self.onnx_session:
                    # ONNX需要numpy数组
                    audio_np = np.frombuffer(req.audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if req.sample_rate != 16000:
                        ratio = 16000 / req.sample_rate
                        new_length = int(len(audio_np) * ratio)
                        audio_np = np.interp(np.linspace(0, len(audio_np), new_length), 
                                           np.arange(len(audio_np)), audio_np)
                    audio_arrays.append(audio_np)
                else:
                    # PyTorch需要tensor
                    audio_tensor = self.preprocess_audio(req.audio_data, req.sample_rate)
                    audio_tensors.append(audio_tensor)
            
            # 批处理推理
            batch_results = []
            if self.onnx_session:
                # 使用ONNX Runtime进行批处理推理
                for audio_np in audio_arrays:
                    # ONNX模型输入
                    input_data = audio_np.reshape(1, -1).astype(np.float32)
                    outputs = self.onnx_session.run(None, {'input': input_data})
                    speech_prob = float(outputs[0][0])
                    is_speech = speech_prob > self.speech_threshold  # P0优化：使用可配置阈值
                    batch_results.append((is_speech, speech_prob))
            else:
                # 使用PyTorch进行推理
                with torch.no_grad():
                    for audio_tensor in audio_tensors:
                        speech_prob = self.model(audio_tensor, 16000).item()
                        is_speech = speech_prob > 0.5
                        batch_results.append((is_speech, speech_prob))
            
            # 构建结果
            processing_time = time.time() - start_time
            for i, req in enumerate(requests):
                is_speech, confidence = batch_results[i]
                result = VADResult(
                    session_id=req.session_id,
                    is_speech=is_speech,
                    confidence=confidence,
                    timestamp=req.timestamp,
                    processing_time=processing_time / len(requests)
                )
                results.append(result)
            
            # 更新统计
            self.total_requests += len(requests)
            self.total_processing_time += processing_time
            
            logger.info(f"批处理完成: {len(requests)}个请求, 耗时: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            # 返回错误结果
            for req in requests:
                results.append(VADResult(
                    session_id=req.session_id,
                    is_speech=False,
                    confidence=0.0,
                    timestamp=req.timestamp,
                    processing_time=0.0
                ))
        
        return results
    
    def get_stats(self) -> Dict:
        """获取性能统计"""
        avg_time = self.total_processing_time / max(self.total_requests, 1)
        return {
            "total_requests": self.total_requests,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_time,
            "requests_per_second": self.total_requests / max(self.total_processing_time, 1)
        }

class VADService:
<<<<<<< HEAD
    """VAD微服务主类，支持高并发和批处理优化"""
    
    def __init__(self, batch_size: int = 32, max_concurrent: int = 48):
        self.batch_size = batch_size  # 从16提升到32
        self.max_concurrent = max_concurrent  # 从24提升到48
        self.processor = SileroVADProcessor(batch_size=batch_size)
        # 使用优化的Redis客户端
        self.redis_client = None
        
        # 智能队列管理器
        self.queue_manager = get_queue_manager(
            "vad_service",
            max_queue_size=5000,
            batch_size=batch_size,
            batch_timeout=0.05,  # 50ms批处理超时
            max_concurrent=max_concurrent
        )
        
        # 优先级队列
        self.high_priority_queue = asyncio.Queue(maxsize=200)  # 增加队列容量
        self.medium_priority_queue = asyncio.Queue(maxsize=500)
        self.low_priority_queue = asyncio.Queue(maxsize=1000)
        
        # 请求队列用于批处理
        self.request_queue = asyncio.Queue(maxsize=1000)
        
        # 性能统计
        self.total_requests = 0
        self.cache_hits = 0
        self.total_processing_time = 0.0
        self.current_concurrent = 0
=======
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
>>>>>>> 2aa4dbc2af6d2f1b4c89b11ac5f75eb495cde788
        
        # 启动批处理任务
        asyncio.create_task(self.batch_processor())
    
    async def initialize(self):
        """初始化服务"""
        try:
            await self.processor.initialize()
            # 初始化优化的Redis客户端
            self.redis_client = await get_redis_client()
            await self.redis_client.health_check()
            # 启动智能队列管理器
            await self.queue_manager.start()
            logger.info("VAD服务初始化成功")
            return True
        except Exception as e:
            logger.error(f"VAD服务初始化失败: {e}")
            return False
    
    async def init_redis(self, redis_url: str = "redis://localhost:6379"):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
    
    async def batch_processor(self):
        """P0优化的批处理任务"""
        while True:
            try:
                requests = []
                
                # P0优化：收集批处理请求（减少延迟）
                timeout = 0.015  # P0优化：15ms超时，减少延迟
                try:
                    # 获取第一个请求
                    req = await asyncio.wait_for(self.request_queue.get(), timeout=timeout)
                    requests.append(req)
                    
                    # P0优化：尝试获取更多请求组成批次（减少等待时间）
                    for _ in range(self.batch_size - 1):
                        try:
                            req = await asyncio.wait_for(self.request_queue.get(), timeout=0.005)  # P0优化：减少到5ms
                            requests.append(req)
                        except asyncio.TimeoutError:
                            break
                
                except asyncio.TimeoutError:
                    continue
                
                if requests:
                    # 处理批次
                    results = await self.processor.process_batch(requests)
                    
                    # 缓存结果到Redis
                    if self.redis_client:
                        for result in results:
                            await self.cache_result(result)
                
            except Exception as e:
                logger.error(f"批处理任务错误: {e}")
                await asyncio.sleep(0.1)
    
    async def cache_result(self, result: VADResult):
        """缓存VAD结果到Redis"""
        try:
            key = f"vad_result:{result.session_id}:{int(result.timestamp)}"
            value = {
                "is_speech": result.is_speech,
                "confidence": result.confidence,
                "timestamp": result.timestamp,
                "processing_time": result.processing_time
            }
            await self.redis_client.setex(key, 300, json.dumps(value))  # 5分钟过期
        except Exception as e:
            logger.error(f"缓存结果失败: {e}")
    
    async def process_vad_request(self, request: VADRequest) -> VADResult:
        """处理单个VAD请求"""
        # 检查缓存
        if self.redis_client:
            cached_result = await self.get_cached_result(request.session_id, request.timestamp)
            if cached_result:
                return cached_result
        
        # 添加到队列等待批处理
        await self.request_queue.put(request)
        
        # 等待结果（简化实现，实际应该使用回调机制）
        # 这里为了演示，直接进行单独处理
        results = await self.processor.process_batch([request])
        return results[0]
    
    async def get_cached_result(self, session_id: str, timestamp: float) -> Optional[VADResult]:
        """从Redis获取缓存结果"""
        try:
            key = f"vad_result:{session_id}:{int(timestamp)}"
            cached = await self.redis_client.get(key)
            if cached:
                data = json.loads(cached)
                return VADResult(
                    session_id=session_id,
                    is_speech=data["is_speech"],
                    confidence=data["confidence"],
                    timestamp=data["timestamp"],
                    processing_time=data["processing_time"]
                )
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
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
<<<<<<< HEAD
vad_service = VADService(batch_size=32, max_concurrent=48)  # 提升配置
=======
vad_service = VADService(batch_size=32, max_concurrent=48)  # P0优化：应用新的批处理和并发参数
>>>>>>> 2aa4dbc2af6d2f1b4c89b11ac5f75eb495cde788

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    await vad_service.init_redis()

@app.post("/vad/detect")
async def detect_voice_activity(
    session_id: str,
    audio_data: str,  # base64编码的音频数据
    sample_rate: int = 16000,
    timestamp: float = 0.0
):
    """VAD检测API"""
    try:
        # 解码音频数据
        audio_bytes = base64.b64decode(audio_data)
        
        # 创建请求
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vad/stats")
async def get_stats():
    """获取服务统计信息"""
    return vad_service.processor.get_stats()

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "vad"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004, workers=1)