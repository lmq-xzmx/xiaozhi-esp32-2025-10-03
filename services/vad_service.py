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
    """优化的SileroVAD处理器，支持批处理和ONNX Runtime优化"""
    
    def __init__(self, model_path: str = "silero_vad.onnx", batch_size: int = 16):
        self.batch_size = batch_size  # 从8提升到16
        self.model = None
        self.onnx_session = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_onnx = model_path.endswith('.onnx')
        self.load_model(model_path)
        
        # 性能统计
        self.total_requests = 0
        self.total_processing_time = 0.0
        
    def load_model(self, model_path: str):
        """加载SileroVAD模型，优先使用ONNX Runtime"""
        if self.use_onnx:
            try:
                import onnxruntime as ort
                
                # ONNX Runtime优化配置
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                options.intra_op_num_threads = 2  # 使用2个线程
                options.inter_op_num_threads = 2  # 使用2个线程
                options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                
                # 选择执行提供者
                providers = ['CPUExecutionProvider']
                if torch.cuda.is_available():
                    providers.insert(0, 'CUDAExecutionProvider')
                
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
                    is_speech = speech_prob > 0.5
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
    """VAD微服务主类"""
    
    def __init__(self, batch_size: int = 16, max_concurrent: int = 24):
        self.batch_size = batch_size  # 从8提升到16
        self.max_concurrent = max_concurrent  # 从16提升到24
        self.processor = SileroVADProcessor(batch_size=batch_size)
        self.request_queue = asyncio.Queue()
        self.redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 启动批处理任务
        asyncio.create_task(self.batch_processor())
    
    async def init_redis(self, redis_url: str = "redis://localhost:6379"):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
    
    async def batch_processor(self):
        """批处理任务"""
        while True:
            try:
                requests = []
                
                # 收集批处理请求
                timeout = 0.05  # 50ms超时
                try:
                    # 获取第一个请求
                    req = await asyncio.wait_for(self.request_queue.get(), timeout=timeout)
                    requests.append(req)
                    
                    # 尝试获取更多请求组成批次
                    for _ in range(self.batch_size - 1):
                        try:
                            req = await asyncio.wait_for(self.request_queue.get(), timeout=0.01)
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
vad_service = VADService(batch_size=16, max_concurrent=24)

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