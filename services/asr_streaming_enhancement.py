#!/usr/bin/env python3
"""
ASR流式处理增强模块
解决实时性问题，支持真正的流式语音识别
"""

import asyncio
import logging
import time
import json
from typing import AsyncGenerator, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import numpy as np
import torch
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StreamingASRRequest:
    """流式ASR请求"""
    session_id: str
    chunk_data: bytes
    sample_rate: int = 16000
    language: str = "zh"
    is_final: bool = False
    chunk_index: int = 0

@dataclass
class StreamingASRResult:
    """流式ASR结果"""
    session_id: str
    text: str
    confidence: float
    is_partial: bool = True
    is_final: bool = False
    chunk_index: int = 0
    processing_time: float = 0.0

class StreamingASRProcessor:
    """流式ASR处理器 - 16GB服务器优化版本"""
    
    def __init__(self, model, chunk_size: int = 320, overlap_size: int = 64):
        self.model = model
        self.chunk_size = chunk_size  # 20ms@16kHz，极低延迟
        self.overlap_size = overlap_size  # 4ms重叠
        self.sample_rate = 16000
        
        # 16GB优化：增大缓冲区
        self.audio_buffer_size = 8192  # 8KB缓冲区
        self.max_context_length = 3200  # 200ms上下文
        
        # 会话管理
        self.active_sessions = {}
        self.session_buffers = {}
        self.session_contexts = {}
        
    async def process_audio_chunk(self, request: StreamingASRRequest) -> StreamingASRResult:
        """处理单个音频块"""
        start_time = time.time()
        session_id = request.session_id
        
        # 初始化会话
        if session_id not in self.session_buffers:
            self.session_buffers[session_id] = bytearray()
            self.session_contexts[session_id] = []
            
        # 添加音频数据到缓冲区
        self.session_buffers[session_id].extend(request.chunk_data)
        
        # 检查是否有足够数据进行处理
        buffer = self.session_buffers[session_id]
        if len(buffer) < self.chunk_size * 2:  # 16位音频
            return StreamingASRResult(
                session_id=session_id,
                text="",
                confidence=0.0,
                is_partial=True,
                chunk_index=request.chunk_index,
                processing_time=time.time() - start_time
            )
        
        try:
            # 提取处理块
            chunk_bytes = buffer[:self.chunk_size * 2]
            # 保留重叠部分
            self.session_buffers[session_id] = buffer[self.chunk_size * 2 - self.overlap_size * 2:]
            
            # 转换为numpy数组
            audio_array = np.frombuffer(chunk_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # 添加上下文
            context = self.session_contexts[session_id]
            if context:
                # 合并上下文和当前块
                full_audio = np.concatenate([context[-self.max_context_length:], audio_array])
            else:
                full_audio = audio_array
            
            # 更新上下文
            self.session_contexts[session_id] = full_audio[-self.max_context_length:]
            
            # 流式推理
            result = await self._streaming_inference(full_audio, request.is_final)
            
            processing_time = time.time() - start_time
            
            return StreamingASRResult(
                session_id=session_id,
                text=result.get('text', ''),
                confidence=result.get('confidence', 0.8),
                is_partial=not request.is_final,
                is_final=request.is_final,
                chunk_index=request.chunk_index,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"流式ASR处理失败 {session_id}: {e}")
            return StreamingASRResult(
                session_id=session_id,
                text="",
                confidence=0.0,
                is_partial=True,
                chunk_index=request.chunk_index,
                processing_time=time.time() - start_time
            )
    
    async def _streaming_inference(self, audio_data: np.ndarray, is_final: bool = False) -> dict:
        """执行流式推理"""
        try:
            # 使用SenseVoice进行流式推理
            # 16GB优化：使用更大的beam_size提高准确性
            beam_size = 3 if is_final else 1  # 最终结果使用更大beam_size
            
            result = self.model.generate(
                input=audio_data,
                cache={},
                language="zh",
                use_itn=True,
                beam_size=beam_size,
                # 流式优化参数
                streaming=True,
                chunk_size=self.chunk_size
            )
            
            if isinstance(result, list) and len(result) > 0:
                return {
                    'text': result[0].get('text', '') if isinstance(result[0], dict) else str(result[0]),
                    'confidence': result[0].get('confidence', 0.8) if isinstance(result[0], dict) else 0.8
                }
            else:
                return {'text': '', 'confidence': 0.0}
                
        except Exception as e:
            logger.error(f"流式推理失败: {e}")
            return {'text': '', 'confidence': 0.0}
    
    def cleanup_session(self, session_id: str):
        """清理会话数据"""
        self.session_buffers.pop(session_id, None)
        self.session_contexts.pop(session_id, None)
        self.active_sessions.pop(session_id, None)

class StreamingASRService:
    """流式ASR服务 - 16GB服务器优化版本"""
    
    def __init__(self, base_asr_service):
        self.base_service = base_asr_service
        self.streaming_processor = StreamingASRProcessor(
            model=base_asr_service.processor.model,
            chunk_size=320,  # 20ms极低延迟
            overlap_size=64   # 4ms重叠
        )
        
        # 16GB优化：增加并发连接数
        self.max_concurrent_streams = 100
        self.active_connections = {}
        
    async def handle_streaming_recognition(self, websocket: WebSocket, session_id: str):
        """处理WebSocket流式识别"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        try:
            chunk_index = 0
            while True:
                # 接收音频数据
                data = await websocket.receive_bytes()
                
                # 创建流式请求
                request = StreamingASRRequest(
                    session_id=session_id,
                    chunk_data=data,
                    chunk_index=chunk_index,
                    is_final=False
                )
                
                # 处理音频块
                result = await self.streaming_processor.process_audio_chunk(request)
                
                # 发送结果
                if result.text:  # 只发送有文本的结果
                    await websocket.send_json({
                        "session_id": result.session_id,
                        "text": result.text,
                        "confidence": result.confidence,
                        "is_partial": result.is_partial,
                        "chunk_index": result.chunk_index,
                        "processing_time": result.processing_time
                    })
                
                chunk_index += 1
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket连接断开: {session_id}")
        except Exception as e:
            logger.error(f"流式识别错误 {session_id}: {e}")
        finally:
            # 清理资源
            self.streaming_processor.cleanup_session(session_id)
            self.active_connections.pop(session_id, None)
    
    async def handle_streaming_http(self, session_id: str, audio_stream) -> AsyncGenerator[str, None]:
        """处理HTTP流式识别"""
        chunk_index = 0
        
        try:
            async for audio_chunk in audio_stream:
                request = StreamingASRRequest(
                    session_id=session_id,
                    chunk_data=audio_chunk,
                    chunk_index=chunk_index,
                    is_final=False
                )
                
                result = await self.streaming_processor.process_audio_chunk(request)
                
                if result.text:
                    yield json.dumps({
                        "session_id": result.session_id,
                        "text": result.text,
                        "confidence": result.confidence,
                        "is_partial": result.is_partial,
                        "chunk_index": result.chunk_index,
                        "processing_time": result.processing_time
                    }) + "\n"
                
                chunk_index += 1
                
        except Exception as e:
            logger.error(f"HTTP流式识别错误 {session_id}: {e}")
        finally:
            self.streaming_processor.cleanup_session(session_id)

# FastAPI路由扩展
def add_streaming_routes(app: FastAPI, asr_service):
    """添加流式ASR路由"""
    streaming_service = StreamingASRService(asr_service)
    
    @app.websocket("/asr/stream/{session_id}")
    async def websocket_streaming_asr(websocket: WebSocket, session_id: str):
        """WebSocket流式ASR端点"""
        await streaming_service.handle_streaming_recognition(websocket, session_id)
    
    @app.post("/asr/stream_http/{session_id}")
    async def http_streaming_asr(session_id: str, audio_data: bytes):
        """HTTP流式ASR端点"""
        async def audio_generator():
            # 将音频数据分块
            chunk_size = 640  # 40ms@16kHz
            for i in range(0, len(audio_data), chunk_size):
                yield audio_data[i:i + chunk_size]
                await asyncio.sleep(0.02)  # 模拟实时流
        
        return StreamingResponse(
            streaming_service.handle_streaming_http(session_id, audio_generator()),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    return streaming_service