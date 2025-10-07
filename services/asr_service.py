#!/usr/bin/env python3
"""
ASR (Automatic Speech Recognition) 微服务
基于SenseVoice，支持批处理、队列管理和模型缓存优化
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
    """优化的SenseVoice处理器，支持FP16量化和更大批处理"""
    
    def __init__(self, model_dir: str = "models/SenseVoiceSmall", batch_size: int = 8, enable_fp16: bool = True):
        self.model_dir = model_dir
        self.batch_size = batch_size  # 从4提升到8
        self.enable_fp16 = enable_fp16
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 性能统计
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        
        # 模型预热
        self.load_model()
        self.warmup_model()
    
    def load_model(self):
        """加载SenseVoice模型，支持FP16量化"""
        try:
            # 检查是否有FP16量化版本
            fp16_model_dir = f"{self.model_dir}_fp16" if self.enable_fp16 else self.model_dir
            
            self.model = AutoModel(
                model=fp16_model_dir if self.enable_fp16 else self.model_dir,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device
            )
            
            # 如果启用FP16且在GPU上，转换模型精度
            if self.enable_fp16 and torch.cuda.is_available():
                try:
                    # 尝试将模型转换为FP16
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'half'):
                        self.model.model = self.model.model.half()
                        logger.info("模型已转换为FP16精度")
                except Exception as e:
                    logger.warning(f"FP16转换失败，使用FP32: {e}")
            
            logger.info(f"SenseVoice模型加载成功，设备: {self.device}, FP16: {self.enable_fp16}")
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
            audio_list = []
            for req in requests:
                audio_np = self.preprocess_audio(req.audio_data, req.sample_rate)
                audio_list.append(audio_np)
            
            # 批处理推理
            batch_results = self.model.generate(
                input=audio_list,
                cache={},
                language="zh",
                use_itn=True,
                batch_size=len(requests)
            )
            
            # 处理结果
            processing_time = time.time() - start_time
            
            for i, req in enumerate(requests):
                if isinstance(batch_results, list):
                    result_text = batch_results[i].get("text", "")
                else:
                    result_text = batch_results.get("text", "")
                
                # 计算置信度（简化实现）
                confidence = min(0.95, len(result_text) / 50.0) if result_text else 0.0
                
                result = ASRResult(
                    session_id=req.session_id,
                    text=result_text,
                    confidence=confidence,
                    language=req.language,
                    timestamp=req.timestamp,
                    processing_time=processing_time / len(requests)
                )
                results.append(result)
            
            # 更新统计
            self.total_requests += len(requests)
            self.total_processing_time += processing_time
            
            logger.info(f"ASR批处理完成: {len(requests)}个请求, 耗时: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"ASR批处理失败: {e}")
            # 返回错误结果
            for req in requests:
                results.append(ASRResult(
                    session_id=req.session_id,
                    text="",
                    confidence=0.0,
                    language=req.language,
                    timestamp=req.timestamp,
                    processing_time=0.0
                ))
        
        return results
    
    def get_stats(self) -> Dict:
        """获取性能统计"""
        avg_time = self.total_processing_time / max(self.total_requests, 1)
        cache_hit_rate = self.cache_hits / max(self.total_requests, 1)
        return {
            "total_requests": self.total_requests,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_time,
            "requests_per_second": self.total_requests / max(self.total_processing_time, 1),
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate
        }

class ASRService:
    """ASR微服务主类"""
    
    def __init__(self, batch_size: int = 8, max_concurrent: int = 16):
        self.batch_size = batch_size  # 从4提升到8
        self.max_concurrent = max_concurrent  # 从8提升到16
        self.processor = SenseVoiceProcessor(batch_size=batch_size, enable_fp16=True)
        
        # 优先级队列
        self.high_priority_queue = asyncio.Queue()
        self.medium_priority_queue = asyncio.Queue()
        self.low_priority_queue = asyncio.Queue()
        
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
    
    async def get_next_batch(self) -> List[ASRRequest]:
        """从优先级队列获取下一批请求"""
        requests = []
        timeout = 0.1  # 100ms超时
        
        # 优先处理高优先级队列
        for queue in [self.high_priority_queue, self.medium_priority_queue, self.low_priority_queue]:
            if requests:
                break
                
            try:
                # 获取第一个请求
                req = await asyncio.wait_for(queue.get(), timeout=timeout)
                requests.append(req)
                
                # 尝试获取更多请求组成批次
                for _ in range(self.batch_size - 1):
                    try:
                        req = await asyncio.wait_for(queue.get(), timeout=0.01)
                        requests.append(req)
                    except asyncio.TimeoutError:
                        break
                        
            except asyncio.TimeoutError:
                continue
        
        return requests
    
    async def batch_processor(self):
        """批处理任务"""
        while True:
            try:
                requests = await self.get_next_batch()
                
                if requests:
                    # 检查缓存
                    cached_results = []
                    uncached_requests = []
                    
                    for req in requests:
                        cached_result = await self.get_cached_result(req)
                        if cached_result:
                            cached_results.append(cached_result)
                            self.processor.cache_hits += 1
                        else:
                            uncached_requests.append(req)
                    
                    # 处理未缓存的请求
                    if uncached_requests:
                        new_results = await self.processor.process_batch(uncached_requests)
                        
                        # 缓存新结果
                        for result in new_results:
                            await self.cache_result(result)
                        
                        cached_results.extend(new_results)
                    
                    # 这里应该将结果发送给客户端（简化实现）
                    logger.info(f"处理完成: {len(cached_results)}个结果")
                
                else:
                    await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"批处理任务错误: {e}")
                await asyncio.sleep(0.1)
    
    async def cache_result(self, result: ASRResult):
        """缓存ASR结果到Redis"""
        try:
            # 使用音频哈希作为缓存键
            key = f"asr_result:{result.session_id}:{int(result.timestamp)}"
            value = {
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "timestamp": result.timestamp,
                "processing_time": result.processing_time
            }
            await self.redis_client.setex(key, 1800, json.dumps(value))  # 30分钟过期
        except Exception as e:
            logger.error(f"缓存结果失败: {e}")
    
    async def get_cached_result(self, request: ASRRequest) -> Optional[ASRResult]:
        """从Redis获取缓存结果"""
        try:
            key = f"asr_result:{request.session_id}:{int(request.timestamp)}"
            cached = await self.redis_client.get(key)
            if cached:
                data = json.loads(cached)
                result = ASRResult(
                    session_id=request.session_id,
                    text=data["text"],
                    confidence=data["confidence"],
                    language=data["language"],
                    timestamp=data["timestamp"],
                    processing_time=data["processing_time"],
                    cached=True
                )
                return result
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
        return None
    
    async def add_request(self, request: ASRRequest):
        """添加ASR请求到优先级队列"""
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
asr_service = ASRService(batch_size=8, max_concurrent=16)

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    await asr_service.init_redis()

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
    """ASR识别API"""
    try:
        # 解码音频数据
        audio_bytes = base64.b64decode(audio_data)
        
        # 创建请求
        request = ASRRequest(
            session_id=session_id,
            audio_data=audio_bytes,
            sample_rate=sample_rate,
            language=language,
            priority=priority,
            timestamp=timestamp
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
        
        # 添加到队列
        await asr_service.add_request(request)
        
        # 简化实现：直接处理单个请求
        results = await asr_service.processor.process_batch([request])
        result = results[0]
        
        # 缓存结果
        if background_tasks:
            background_tasks.add_task(asr_service.cache_result, result)
        
        return {
            "session_id": result.session_id,
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language,
            "timestamp": result.timestamp,
            "processing_time": result.processing_time,
            "cached": False
        }
        
    except Exception as e:
        logger.error(f"ASR识别失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/asr/stats")
async def get_stats():
    """获取服务统计信息"""
    stats = asr_service.processor.get_stats()
    stats.update({
        "high_priority_queue_size": asr_service.high_priority_queue.qsize(),
        "medium_priority_queue_size": asr_service.medium_priority_queue.qsize(),
        "low_priority_queue_size": asr_service.low_priority_queue.qsize()
    })
    return stats

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "asr"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)