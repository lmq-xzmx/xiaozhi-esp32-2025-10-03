#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESP32服务器API客户端
用于与xiaozhi-esp32-server进行通信，实现完整的音频处理流程
"""

import asyncio
import aiohttp
import json
import logging
import websockets
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from dataclasses import dataclass
import base64

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VADResult:
    """VAD检测结果"""
    has_speech: bool
    confidence: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str
    confidence: float
    language: Optional[str] = None
    segments: Optional[List[Dict]] = None


@dataclass
class LLMResponse:
    """LLM响应结果"""
    text: str
    model: str
    usage: Optional[Dict] = None
    finish_reason: Optional[str] = None


@dataclass
class TTSResult:
    """TTS合成结果"""
    audio_data: bytes
    format: str
    sample_rate: int
    duration: Optional[float] = None


class ESP32ServerAPIClient:
    """ESP32服务器API客户端"""
    
    def __init__(self, 
                 base_url: str = "http://xiaozhi-esp32-server:8003",
                 websocket_url: str = "ws://xiaozhi-esp32-server:8000",
                 timeout: int = 30):
        """
        初始化ESP32服务器API客户端
        
        Args:
            base_url: HTTP API基础URL
            websocket_url: WebSocket URL
            timeout: 请求超时时间
        """
        self.base_url = base_url
        self.websocket_url = websocket_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self._websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        if not self.session or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': 'xiaozhi-server/1.0'
                }
            )
        return self.session
    
    async def close(self):
        """关闭客户端连接"""
        if self.session and not self.session.closed:
            await self.session.close()
        
        # 关闭WebSocket连接
        for ws in self._websocket_connections.values():
            if not ws.closed:
                await ws.close()
        self._websocket_connections.clear()
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "status": "healthy",
                        "esp32_server": data,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"HTTP {resp.status}",
                        "timestamp": datetime.now().isoformat()
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def vad_detect(self, audio_data: bytes, format: str = "opus") -> VADResult:
        """
        VAD语音活动检测
        
        Args:
            audio_data: 音频数据
            format: 音频格式
            
        Returns:
            VAD检测结果
        """
        try:
            session = await self._get_session()
            
            # 准备请求数据
            payload = {
                "audio": base64.b64encode(audio_data).decode('utf-8'),
                "format": format,
                "sample_rate": 16000
            }
            
            async with session.post(f"{self.base_url}/api/vad", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return VADResult(
                        has_speech=data.get('has_speech', False),
                        confidence=data.get('confidence', 0.0),
                        start_time=data.get('start_time'),
                        end_time=data.get('end_time')
                    )
                else:
                    logger.error(f"VAD检测失败: HTTP {resp.status}")
                    return VADResult(has_speech=False, confidence=0.0)
                    
        except Exception as e:
            logger.error(f"VAD检测异常: {e}")
            return VADResult(has_speech=False, confidence=0.0)
    
    async def asr_recognize(self, audio_data: bytes, format: str = "opus") -> ASRResult:
        """
        ASR语音识别
        
        Args:
            audio_data: 音频数据
            format: 音频格式
            
        Returns:
            ASR识别结果
        """
        try:
            session = await self._get_session()
            
            # 准备请求数据
            payload = {
                "audio": base64.b64encode(audio_data).decode('utf-8'),
                "format": format,
                "sample_rate": 16000,
                "language": "zh"
            }
            
            async with session.post(f"{self.base_url}/api/asr", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return ASRResult(
                        text=data.get('text', ''),
                        confidence=data.get('confidence', 0.0),
                        language=data.get('language'),
                        segments=data.get('segments')
                    )
                else:
                    logger.error(f"ASR识别失败: HTTP {resp.status}")
                    return ASRResult(text='', confidence=0.0)
                    
        except Exception as e:
            logger.error(f"ASR识别异常: {e}")
            return ASRResult(text='', confidence=0.0)
    
    async def asr_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[ASRResult, None]:
        """
        流式ASR语音识别
        
        Args:
            audio_stream: 音频数据流
            
        Yields:
            ASR识别结果流
        """
        try:
            # 建立WebSocket连接进行流式识别
            uri = f"{self.websocket_url}/asr/stream"
            async with websockets.connect(uri) as websocket:
                # 发送配置
                config = {
                    "type": "config",
                    "format": "opus",
                    "sample_rate": 16000,
                    "language": "zh"
                }
                await websocket.send(json.dumps(config))
                
                # 启动音频发送任务
                async def send_audio():
                    async for chunk in audio_stream:
                        audio_msg = {
                            "type": "audio",
                            "data": base64.b64encode(chunk).decode('utf-8')
                        }
                        await websocket.send(json.dumps(audio_msg))
                    
                    # 发送结束信号
                    await websocket.send(json.dumps({"type": "end"}))
                
                # 启动接收任务
                async def receive_results():
                    async for message in websocket:
                        data = json.loads(message)
                        if data.get('type') == 'result':
                            yield ASRResult(
                                text=data.get('text', ''),
                                confidence=data.get('confidence', 0.0),
                                language=data.get('language'),
                                segments=data.get('segments')
                            )
                
                # 并发执行发送和接收
                send_task = asyncio.create_task(send_audio())
                async for result in receive_results():
                    yield result
                
                await send_task
                
        except Exception as e:
            logger.error(f"流式ASR识别异常: {e}")
    
    async def llm_chat(self, 
                      message: str, 
                      model: str = "qwen2:7b",
                      context: Optional[List[Dict]] = None) -> LLMResponse:
        """
        LLM对话（非流式）
        
        Args:
            message: 用户消息
            model: 模型名称
            context: 对话上下文
            
        Returns:
            LLM响应结果
        """
        try:
            session = await self._get_session()
            
            # 准备请求数据
            payload = {
                "message": message,
                "model": model,
                "stream": False,
                "context": context or []
            }
            
            # 非流式响应
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return LLMResponse(
                        text=data.get('text', ''),
                        model=data.get('model', model),
                        usage=data.get('usage'),
                        finish_reason=data.get('finish_reason')
                    )
                else:
                    logger.error(f"LLM对话失败: HTTP {resp.status}")
                    return LLMResponse(text='', model=model)
                        
        except Exception as e:
            logger.error(f"LLM对话异常: {e}")
            return LLMResponse(text='', model=model)
    
    async def llm_chat_stream(self, 
                             message: str, 
                             model: str = "qwen2:7b",
                             context: Optional[List[Dict]] = None) -> AsyncGenerator[str, None]:
        """
        LLM对话（流式）
        
        Args:
            message: 用户消息
            model: 模型名称
            context: 对话上下文
            
        Yields:
            流式响应文本
        """
        try:
            session = await self._get_session()
            
            # 准备请求数据
            payload = {
                "message": message,
                "model": model,
                "stream": True,
                "context": context or []
            }
            
            # 流式响应
            async with session.post(f"{self.base_url}/api/chat/stream", json=payload) as resp:
                if resp.status == 200:
                    async for line in resp.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'text' in data:
                                    yield data['text']
                            except json.JSONDecodeError:
                                continue
                else:
                    logger.error(f"LLM流式对话失败: HTTP {resp.status}")
                        
        except Exception as e:
            logger.error(f"LLM流式对话异常: {e}")
    
    async def tts_synthesize(self, 
                           text: str, 
                           voice: str = "zh-CN-XiaoxiaoNeural",
                           format: str = "opus",
                           speed: float = 1.0) -> TTSResult:
        """
        TTS语音合成
        
        Args:
            text: 要合成的文本
            voice: 语音模型
            format: 音频格式
            speed: 语速
            
        Returns:
            TTS合成结果
        """
        try:
            session = await self._get_session()
            
            # 准备请求数据
            payload = {
                "text": text,
                "voice": voice,
                "format": format,
                "speed": speed
            }
            
            async with session.post(f"{self.base_url}/api/tts", json=payload) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    
                    # 获取响应头信息
                    content_type = resp.headers.get('content-type', '')
                    sample_rate = int(resp.headers.get('x-sample-rate', '16000'))
                    duration = float(resp.headers.get('x-duration', '0'))
                    
                    return TTSResult(
                        audio_data=audio_data,
                        format=format,
                        sample_rate=sample_rate,
                        duration=duration if duration > 0 else None
                    )
                else:
                    logger.error(f"TTS合成失败: HTTP {resp.status}")
                    return TTSResult(audio_data=b'', format=format, sample_rate=16000)
                    
        except Exception as e:
            logger.error(f"TTS合成异常: {e}")
            return TTSResult(audio_data=b'', format=format, sample_rate=16000)
    
    async def get_recent_chat_history(self, 
                                    device_id: str,
                                    limit: int = 50,
                                    since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        获取最近的聊天记录
        
        Args:
            device_id: 设备ID
            limit: 记录数量限制
            since: 起始时间
            
        Returns:
            聊天记录列表
        """
        try:
            session = await self._get_session()
            
            params = {
                "device_id": device_id,
                "limit": limit
            }
            if since:
                params["since"] = since.isoformat()
            
            async with session.get(f"{self.base_url}/api/chat/history", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('records', [])
                else:
                    logger.error(f"获取聊天记录失败: HTTP {resp.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"获取聊天记录异常: {e}")
            return []
    
    async def get_chat_history_since(self, since_time: datetime) -> List[Dict[str, Any]]:
        """
        获取指定时间之后的聊天记录
        
        Args:
            since_time: 起始时间
            
        Returns:
            聊天记录列表
        """
        try:
            session = await self._get_session()
            
            params = {
                "since": since_time.isoformat(),
                "limit": 1000  # 获取更多记录用于同步
            }
            
            async with session.get(f"{self.base_url}/api/chat/history/since", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get('records', [])
                else:
                    logger.error(f"获取聊天记录失败: HTTP {resp.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"获取聊天记录异常: {e}")
            return []
    
    async def sync_chat_records(self, records: List[Dict[str, Any]]) -> bool:
        """
        同步聊天记录到ESP32服务器
        
        Args:
            records: 要同步的聊天记录列表
            
        Returns:
            是否同步成功
        """
        try:
            session = await self._get_session()
            
            payload = {
                "records": records,
                "sync_time": datetime.now().isoformat()
            }
            
            async with session.post(f"{self.base_url}/api/chat/sync", json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.info(f"成功同步 {len(records)} 条记录到ESP32服务器")
                    return data.get('success', False)
                else:
                    logger.error(f"同步聊天记录失败: HTTP {resp.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"同步聊天记录异常: {e}")
            return False
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """
        获取ESP32服务器的同步状态
        
        Returns:
            同步状态信息
        """
        try:
            session = await self._get_session()
            
            async with session.get(f"{self.base_url}/api/sync/status") as resp:
                if resp.status == 200:
                    return await resp.json()
                else:
                    logger.error(f"获取同步状态失败: HTTP {resp.status}")
                    return {"error": f"HTTP {resp.status}"}
                    
        except Exception as e:
            logger.error(f"获取同步状态异常: {e}")
            return {"error": str(e)}
    
    async def complete_audio_processing(self, 
                                      audio_data: bytes,
                                      device_id: str,
                                      session_id: str) -> Dict[str, Any]:
        """
        完整的音频处理流程：VAD → ASR → LLM → TTS
        
        Args:
            audio_data: 音频数据
            device_id: 设备ID
            session_id: 会话ID
            
        Returns:
            处理结果
        """
        result = {
            "success": False,
            "session_id": session_id,
            "device_id": device_id,
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }
        
        try:
            # 1. VAD检测
            logger.info(f"开始VAD检测 - 设备: {device_id}")
            vad_result = await self.vad_detect(audio_data)
            result["steps"]["vad"] = {
                "has_speech": vad_result.has_speech,
                "confidence": vad_result.confidence
            }
            
            if not vad_result.has_speech:
                result["message"] = "未检测到语音活动"
                return result
            
            # 2. ASR识别
            logger.info(f"开始ASR识别 - 设备: {device_id}")
            asr_result = await self.asr_recognize(audio_data)
            result["steps"]["asr"] = {
                "text": asr_result.text,
                "confidence": asr_result.confidence
            }
            
            if not asr_result.text.strip():
                result["message"] = "语音识别无结果"
                return result
            
            # 3. LLM处理
            logger.info(f"开始LLM处理 - 设备: {device_id}, 文本: {asr_result.text}")
            llm_response = await self.llm_chat(asr_result.text)
            result["steps"]["llm"] = {
                "response": llm_response.text,
                "model": llm_response.model
            }
            
            if not llm_response.text.strip():
                result["message"] = "LLM无响应"
                return result
            
            # 4. TTS合成
            logger.info(f"开始TTS合成 - 设备: {device_id}, 文本: {llm_response.text}")
            tts_result = await self.tts_synthesize(llm_response.text)
            result["steps"]["tts"] = {
                "audio_length": len(tts_result.audio_data),
                "format": tts_result.format,
                "sample_rate": tts_result.sample_rate
            }
            
            # 5. 完成
            result["success"] = True
            result["user_text"] = asr_result.text
            result["ai_response"] = llm_response.text
            result["audio_data"] = tts_result.audio_data
            result["message"] = "音频处理完成"
            
            logger.info(f"音频处理完成 - 设备: {device_id}, 会话: {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"音频处理异常 - 设备: {device_id}: {e}")
            result["error"] = str(e)
            result["message"] = "音频处理失败"
            return result


# 全局实例
_esp32_api_client = None

def get_esp32_api_client() -> ESP32ServerAPIClient:
    """获取ESP32 API客户端的全局实例"""
    global _esp32_api_client
    if _esp32_api_client is None:
        _esp32_api_client = ESP32ServerAPIClient()
    return _esp32_api_client


if __name__ == "__main__":
    # 测试ESP32 API客户端
    async def test_client():
        client = ESP32ServerAPIClient()
        
        # 健康检查
        health = await client.health_check()
        print("ESP32服务器健康检查结果:")
        print(json.dumps(health, indent=2, ensure_ascii=False))
        
        await client.close()
    
    asyncio.run(test_client())