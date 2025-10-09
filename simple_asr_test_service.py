#!/usr/bin/env python3
"""
简化的ASR测试服务
用于验证ASR优化配置，不依赖funasr模块
"""

import asyncio
import logging
import time
import json
import base64
import os
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASRRequest(BaseModel):
    session_id: str
    audio_data: str  # base64编码的音频数据
    sample_rate: int = 16000
    language: str = "zh"
    priority: int = 2
    timestamp: float = 0.0

class SimpleASRService:
    """简化的ASR服务，用于测试优化配置"""
    
    def __init__(self, batch_size: int = 8, max_concurrent: int = 20):
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.current_concurrent = 0
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        
        # 模拟队列
        self.high_priority_queue = asyncio.Queue(maxsize=30)
        self.medium_priority_queue = asyncio.Queue(maxsize=40)
        self.low_priority_queue = asyncio.Queue(maxsize=20)
        
        # 简单的内存缓存
        self.cache = {}
        
        logger.info(f"简化ASR服务初始化完成 - batch_size: {batch_size}, max_concurrent: {max_concurrent}")
    
    async def recognize_audio(self, request: ASRRequest) -> Dict[str, Any]:
        """模拟ASR识别"""
        start_time = time.time()
        
        try:
            # 检查并发限制
            if self.current_concurrent >= self.max_concurrent:
                raise HTTPException(status_code=503, detail="服务繁忙，请稍后重试")
            
            self.current_concurrent += 1
            self.total_requests += 1
            
            # 解码音频数据
            try:
                audio_bytes = base64.b64decode(request.audio_data)
                audio_size = len(audio_bytes)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"音频数据解码失败: {str(e)}")
            
            # 检查音频大小限制
            max_audio_size = 1024 * 1024  # 1MB限制
            if audio_size > max_audio_size:
                raise HTTPException(status_code=400, detail="音频数据过大，请压缩后重试")
            
            # 生成缓存键
            cache_key = f"{request.session_id}_{hash(request.audio_data)}"
            
            # 检查缓存
            if cache_key in self.cache:
                self.cache_hits += 1
                cached_result = self.cache[cache_key]
                logger.info(f"缓存命中: {request.session_id}")
                return {
                    "session_id": request.session_id,
                    "text": cached_result["text"],
                    "confidence": cached_result["confidence"],
                    "language": request.language,
                    "timestamp": request.timestamp,
                    "processing_time": time.time() - start_time,
                    "cached": True,
                    "audio_size": audio_size
                }
            
            # 模拟ASR处理延迟（根据音频大小）
            processing_delay = min(0.1 + (audio_size / 100000), 2.0)  # 0.1-2秒
            await asyncio.sleep(processing_delay)
            
            # 模拟识别结果
            mock_texts = [
                "你好，这是一个测试音频",
                "ASR服务运行正常",
                "语音识别功能测试",
                "优化配置验证成功",
                "系统性能良好"
            ]
            
            # 根据session_id选择结果
            text_index = hash(request.session_id) % len(mock_texts)
            recognized_text = mock_texts[text_index]
            confidence = 0.85 + (hash(request.session_id) % 15) / 100  # 0.85-0.99
            
            # 缓存结果
            result = {
                "text": recognized_text,
                "confidence": confidence
            }
            self.cache[cache_key] = result
            
            # 限制缓存大小
            if len(self.cache) > 1000:
                # 删除最旧的缓存项
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            logger.info(f"ASR识别完成: {request.session_id}, 耗时: {processing_time:.3f}s")
            
            return {
                "session_id": request.session_id,
                "text": recognized_text,
                "confidence": confidence,
                "language": request.language,
                "timestamp": request.timestamp,
                "processing_time": processing_time,
                "cached": False,
                "audio_size": audio_size
            }
            
        finally:
            self.current_concurrent -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        avg_processing_time = self.total_processing_time / max(self.total_requests, 1)
        cache_hit_rate = self.cache_hits / max(self.total_requests, 1)
        
        return {
            "processor": {
                "total_requests": self.total_requests,
                "avg_processing_time": avg_processing_time,
                "cache_hit_rate": cache_hit_rate,
                "batch_size": self.batch_size,
                "cache_size": len(self.cache)
            },
            "service": {
                "current_concurrent": self.current_concurrent,
                "max_concurrent": self.max_concurrent,
                "total_requests": self.total_requests,
                "cache_hits": self.cache_hits
            },
            "queues": {
                "high": self.high_priority_queue.qsize(),
                "medium": self.medium_priority_queue.qsize(),
                "low": self.low_priority_queue.qsize()
            }
        }

# FastAPI应用
app = FastAPI(title="Simple ASR Test Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 从环境变量读取优化配置
batch_size = int(os.getenv("ASR_BATCH_SIZE", "10"))
max_concurrent = int(os.getenv("ASR_MAX_CONCURRENT", "25"))
cache_size_mb = int(os.getenv("ASR_CACHE_SIZE_MB", "768"))
worker_threads = int(os.getenv("ASR_WORKER_THREADS", "4"))

logger.info(f"ASR优化配置: batch_size={batch_size}, max_concurrent={max_concurrent}, cache_size_mb={cache_size_mb}, worker_threads={worker_threads}")

# 全局ASR服务实例 - 4核8GB优化配置
asr_service = SimpleASRService(batch_size=batch_size, max_concurrent=max_concurrent)

@app.post("/asr/recognize")
async def recognize_speech(request: ASRRequest):
    """语音识别API"""
    return await asr_service.recognize_audio(request)

@app.get("/asr/stats")
async def get_stats():
    """获取ASR服务统计信息"""
    return asr_service.get_stats()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "simple_asr_test", 
        "optimization": "4core_8gb",
        "current_concurrent": asr_service.current_concurrent,
        "max_concurrent": asr_service.max_concurrent
    }

if __name__ == "__main__":
    logger.info("启动简化ASR测试服务...")
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)