#!/usr/bin/env python3
"""
ASR流式处理增强模块
解决实时性问题，支持真正的流式语音识别
集成聊天记录功能
"""

import asyncio
import logging
import time
import json
import sys
import os
from typing import AsyncGenerator, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import numpy as np
import torch
from dataclasses import dataclass

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入聊天记录服务
from core.chat_history_service import ChatHistoryService

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
    """流式ASR服务 - 16GB服务器优化版本，集成聊天记录功能"""
    
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
        
        # 聊天记录服务
        self.chat_service = ChatHistoryService()
        
        # 会话管理：存储设备ID和学生ID的映射
        self.session_device_mapping = {}  # session_id -> device_id
        self.session_student_mapping = {}  # session_id -> student_id
        
        # 累积文本缓存，用于完整句子的记录
        self.session_text_buffer = {}  # session_id -> accumulated_text
        
    async def handle_streaming_recognition(self, websocket: WebSocket, session_id: str, device_id: str = None, student_id: int = None):
        """处理WebSocket流式识别，集成聊天记录功能"""
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        # 存储设备和学生映射信息
        if device_id:
            self.session_device_mapping[session_id] = device_id
        if student_id:
            self.session_student_mapping[session_id] = student_id
        
        # 初始化文本缓存
        self.session_text_buffer[session_id] = ""
        
        try:
            chunk_index = 0
            while True:
                # 接收数据（可能是音频数据或JSON消息）
                try:
                    # 尝试接收JSON消息（包含设备信息）
                    message = await websocket.receive_json()
                    if message.get("type") == "device_info":
                        # 更新设备和学生信息
                        self.session_device_mapping[session_id] = message.get("device_id", device_id)
                        self.session_student_mapping[session_id] = message.get("student_id", student_id)
                        logger.info(f"更新会话 {session_id} 设备信息: device_id={message.get('device_id')}, student_id={message.get('student_id')}")
                        continue
                except:
                    # 如果不是JSON，则接收音频数据
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
                
                # 发送结果并处理聊天记录
                if result.text:  # 只发送有文本的结果
                    # 累积文本
                    self.session_text_buffer[session_id] += result.text
                    
                    # 发送ASR结果
                    response = {
                        "session_id": result.session_id,
                        "text": result.text,
                        "confidence": result.confidence,
                        "is_partial": result.is_partial,
                        "chunk_index": result.chunk_index,
                        "processing_time": result.processing_time
                    }
                    await websocket.send_json(response)
                    
                    # 如果是最终结果，记录到聊天历史
                    if result.is_final or result.confidence > 0.8:
                        await self._record_user_input(session_id, self.session_text_buffer[session_id])
                        # 清空缓存
                        self.session_text_buffer[session_id] = ""
                
                chunk_index += 1
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket连接断开: {session_id}")
        except Exception as e:
            logger.error(f"流式识别错误 {session_id}: {e}")
        finally:
            # 记录最后的累积文本（如果有）
            if session_id in self.session_text_buffer and self.session_text_buffer[session_id].strip():
                await self._record_user_input(session_id, self.session_text_buffer[session_id])
            
            # 清理资源
            self.streaming_processor.cleanup_session(session_id)
            self.active_connections.pop(session_id, None)
            self.session_device_mapping.pop(session_id, None)
            self.session_student_mapping.pop(session_id, None)
            self.session_text_buffer.pop(session_id, None)
    
    async def _record_user_input(self, session_id: str, user_text: str):
        """记录用户输入到聊天历史"""
        try:
            device_id = self.session_device_mapping.get(session_id)
            student_id = self.session_student_mapping.get(session_id)
            
            if device_id and student_id and user_text.strip():
                await self.chat_service.record_chat(
                    device_id=device_id,
                    student_id=student_id,
                    user_input=user_text.strip(),
                    ai_response="",  # ASR阶段还没有AI响应
                    session_id=session_id
                )
                logger.info(f"记录用户输入: device_id={device_id}, student_id={student_id}, text='{user_text.strip()}'")
        except Exception as e:
            logger.error(f"记录用户输入失败 {session_id}: {e}")
    
    async def record_ai_response(self, session_id: str, ai_response: str):
        """记录AI响应到聊天历史"""
        try:
            device_id = self.session_device_mapping.get(session_id)
            student_id = self.session_student_mapping.get(session_id)
            
            if device_id and student_id and ai_response.strip():
                await self.chat_service.record_chat(
                    device_id=device_id,
                    student_id=student_id,
                    user_input="",  # 这里只记录AI响应
                    ai_response=ai_response.strip(),
                    session_id=session_id
                )
                logger.info(f"记录AI响应: device_id={device_id}, student_id={student_id}, response='{ai_response.strip()}'")
        except Exception as e:
            logger.error(f"记录AI响应失败 {session_id}: {e}")
    
    async def get_session_info(self, session_id: str) -> dict:
        """获取会话信息"""
        return {
            "session_id": session_id,
            "device_id": self.session_device_mapping.get(session_id),
            "student_id": self.session_student_mapping.get(session_id),
            "text_buffer": self.session_text_buffer.get(session_id, ""),
            "is_active": session_id in self.active_connections
        }
    
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
    """添加流式ASR路由，集成聊天记录功能"""
    streaming_service = StreamingASRService(asr_service)
    
    @app.websocket("/asr/stream/{session_id}")
    async def websocket_streaming_asr(websocket: WebSocket, session_id: str, device_id: str = None, student_id: int = None):
        """WebSocket流式ASR端点，支持聊天记录"""
        await streaming_service.handle_streaming_recognition(websocket, session_id, device_id, student_id)
    
    @app.websocket("/asr/stream/{device_id}/{student_id}/{session_id}")
    async def websocket_streaming_asr_with_ids(websocket: WebSocket, device_id: str, student_id: int, session_id: str):
        """WebSocket流式ASR端点，明确指定设备ID和学生ID"""
        await streaming_service.handle_streaming_recognition(websocket, session_id, device_id, student_id)
    
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
    
    @app.post("/asr/record_ai_response/{session_id}")
    async def record_ai_response(session_id: str, response_data: dict):
        """记录AI响应到聊天历史"""
        ai_response = response_data.get("ai_response", "")
        await streaming_service.record_ai_response(session_id, ai_response)
        return {"status": "success", "message": "AI响应已记录"}
    
    @app.get("/asr/session/{session_id}")
    async def get_session_info(session_id: str):
        """获取会话信息"""
        return await streaming_service.get_session_info(session_id)
    
    @app.get("/asr/sessions")
    async def get_active_sessions():
        """获取所有活跃会话"""
        sessions = []
        for session_id in streaming_service.active_connections.keys():
            session_info = await streaming_service.get_session_info(session_id)
            sessions.append(session_info)
        return {"active_sessions": sessions, "total": len(sessions)}
    
    @app.get("/asr/chat_health")
    async def chat_service_health():
        """聊天记录服务健康检查"""
        try:
            health = await streaming_service.chat_service.health_check()
            return health
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    return streaming_service