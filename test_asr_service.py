#!/usr/bin/env python3
"""
简化的ASR服务测试版本 - 专注于测试聊天记录功能
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import asyncio
import logging
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import base64

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
    priority: int = 1

@dataclass
class ASRResult:
    session_id: str
    text: str
    confidence: float
    language: str
    timestamp: float
    processing_time: float
    cached: bool = False

class MockASRProcessor:
    """模拟ASR处理器，用于测试"""
    
    def __init__(self):
        self.mock_responses = [
            "你好，我是小智助手",
            "今天天气怎么样？",
            "请帮我查询一下明天的日程",
            "谢谢你的帮助",
            "再见"
        ]
        self.response_index = 0
    
    async def process_batch(self, requests: List[ASRRequest]) -> List[ASRResult]:
        """模拟批处理ASR请求"""
        results = []
        for request in requests:
            # 模拟处理时间
            await asyncio.sleep(0.1)
            
            # 循环使用模拟响应
            text = self.mock_responses[self.response_index % len(self.mock_responses)]
            self.response_index += 1
            
            result = ASRResult(
                session_id=request.session_id,
                text=text,
                confidence=0.95,
                language=request.language,
                timestamp=request.timestamp,
                processing_time=0.1,
                cached=False
            )
            results.append(result)
            logger.info(f"🎤 模拟ASR结果: {text}")
        
        return results

class MockASRService:
    """模拟ASR服务"""
    
    def __init__(self):
        self.processor = MockASRProcessor()
        self.request_queue = asyncio.Queue()
        self.running = True
        
    async def add_request(self, request: ASRRequest):
        """添加ASR请求到队列"""
        await self.request_queue.put(request)
        logger.info(f"📝 添加ASR请求: {request.session_id}")
    
    async def batch_processor(self):
        """批处理器"""
        while self.running:
            try:
                # 等待请求
                request = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                
                # 处理请求
                results = await self.processor.process_batch([request])
                
                # 这里可以添加结果处理逻辑
                for result in results:
                    logger.info(f"✅ ASR处理完成: {result.text}")
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"❌ ASR处理错误: {e}")

# 创建FastAPI应用
app = FastAPI(title="Test ASR Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建模拟ASR服务实例
asr_service = MockASRService()

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    logger.info("🚀 启动测试ASR服务...")
    # 启动批处理器
    asyncio.create_task(asr_service.batch_processor())

@app.post("/asr/recognize")
async def recognize_speech(
    session_id: str,
    audio_data: str,  # base64编码的音频数据
    sample_rate: int = 16000,
    language: str = "zh",
    priority: int = 2,
    timestamp: float = 0.0
):
    """模拟语音识别接口"""
    try:
        # 解码音频数据
        audio_bytes = base64.b64decode(audio_data)
        
        # 创建ASR请求
        request = ASRRequest(
            session_id=session_id,
            audio_data=audio_bytes,
            sample_rate=sample_rate,
            language=language,
            timestamp=timestamp or time.time(),
            priority=priority
        )
        
        # 添加到处理队列
        await asr_service.add_request(request)
        
        # 模拟处理结果
        mock_text = asr_service.processor.mock_responses[
            asr_service.processor.response_index % len(asr_service.processor.mock_responses)
        ]
        asr_service.processor.response_index += 1
        
        return {
            "success": True,
            "session_id": session_id,
            "text": mock_text,
            "confidence": 0.95,
            "language": language,
            "timestamp": request.timestamp,
            "processing_time": 0.1
        }
        
    except Exception as e:
        logger.error(f"❌ ASR识别失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

@app.get("/asr/stats")
async def get_stats():
    """获取服务统计信息"""
    return {
        "service": "test_asr",
        "status": "running",
        "queue_size": asr_service.request_queue.qsize(),
        "mock_responses": len(asr_service.processor.mock_responses)
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "service": "test_asr"}

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
    logger.info("🎯 启动测试ASR服务 (端口: 8001)")
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)