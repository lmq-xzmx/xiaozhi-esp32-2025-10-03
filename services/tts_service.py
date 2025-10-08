#!/usr/bin/env python3
"""
TTS (Text-to-Speech) 微服务
支持并发处理、音频缓存、多引擎负载均衡和流式传输
"""

import asyncio
import logging
import time
import hashlib
import json
import io
import base64
from typing import List, Dict, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum
import aiofiles
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel
import edge_tts
import azure.cognitiveservices.speech as speechsdk
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSEngine(Enum):
    """TTS引擎枚举"""
    EDGE_TTS = "edge_tts"
    AZURE_TTS = "azure_tts"
    XUNFEI_TTS = "xunfei_tts"
    LOCAL_TTS = "local_tts"
    HUOSHAN_TTS = "huoshan_tts"  # 新增火山引擎TTS

@dataclass
class TTSVoice:
    """TTS语音配置"""
    engine: TTSEngine
    voice_id: str
    language: str
    gender: str
    name: str
    sample_rate: int = 24000
    quality: str = "high"
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0

class TTSRequest(BaseModel):
    """TTS请求模型"""
    session_id: str
    text: str
    voice_id: Optional[str] = None
    language: str = "zh-CN"
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    format: str = "opus"  # P0优化：默认使用opus格式提高压缩效率
    sample_rate: int = 24000
    stream: bool = False
    cache_enabled: bool = True
    priority: int = 2  # 1=高优先级, 2=中优先级, 3=低优先级

class TTSResponse(BaseModel):
    """TTS响应模型"""
    session_id: str
    audio_data: Optional[str] = None  # base64编码的音频数据
    audio_url: Optional[str] = None
    duration: float
    format: str
    sample_rate: int
    file_size: int
    processing_time: float
    cached: bool = False
    voice_id: str
    engine: str

class TTSCache:
    """TTS音频缓存管理器 - P0优化版本"""
    
    def __init__(self, redis_client, cache_dir: str = "/tmp/tts_cache"):
        self.redis_client = redis_client
        self.cache_dir = cache_dir
        self.cache_ttl = 3600  # P0优化：1小时TTL（优化配置建议）
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.compression_enabled = True  # P0优化：启用压缩
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # P0优化：预缓存常用短语
        self.common_phrases = {
            "greetings": ["你好", "早上好", "下午好", "晚上好", "欢迎使用", "很高兴为您服务"],
            "responses": ["好的", "明白了", "没问题", "请稍等", "正在处理", "已经完成", "收到", "了解"],
            "errors": ["抱歉，我没听清", "请重新说一遍", "网络连接异常", "系统繁忙，请稍后再试", "语音识别失败"],
            "confirmations": ["是的", "不是", "确认", "取消", "继续", "停止"]
        }
    
    def generate_cache_key(self, text: str, voice_id: str, speed: float, pitch: float, volume: float) -> str:
        """生成缓存键"""
        content = f"{text}:{voice_id}:{speed}:{pitch}:{volume}"
        return f"tts_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get_cached_audio(self, cache_key: str) -> Optional[Dict]:
        """获取缓存音频"""
        try:
            cached_info = await self.redis_client.get(cache_key)
            if cached_info:
                info = json.loads(cached_info)
                file_path = info["file_path"]
                
                # 检查文件是否存在
                if os.path.exists(file_path):
                    async with aiofiles.open(file_path, 'rb') as f:
                        audio_data = await f.read()
                    
                    info["audio_data"] = audio_data
                    return info
                else:
                    # 文件不存在，删除缓存记录
                    await self.redis_client.delete(cache_key)
        except Exception as e:
            logger.error(f"获取缓存音频失败: {e}")
        return None
    
    async def cache_audio(self, cache_key: str, audio_data: bytes, metadata: Dict):
        """缓存音频数据"""
        try:
            # 检查文件大小
            if len(audio_data) > self.max_file_size:
                logger.warning(f"音频文件过大，跳过缓存: {len(audio_data)} bytes")
                return
            
            # 保存到文件
            file_name = f"{cache_key.split(':')[-1]}.{metadata['format']}"
            file_path = os.path.join(self.cache_dir, file_name)
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(audio_data)
            
            # 保存元数据到Redis
            cache_info = {
                "file_path": file_path,
                "duration": metadata["duration"],
                "format": metadata["format"],
                "sample_rate": metadata["sample_rate"],
                "file_size": len(audio_data),
                "voice_id": metadata["voice_id"],
                "engine": metadata["engine"],
                "timestamp": time.time()
            }
            
            await self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(cache_info))
            logger.info(f"音频缓存成功: {file_path}")
            
        except Exception as e:
            logger.error(f"缓存音频失败: {e}")
    
    async def cleanup_expired_cache(self):
        """清理过期缓存"""
        try:
            # 获取所有缓存键
            keys = await self.redis_client.keys("tts_cache:*")
            
            for key in keys:
                cached_info = await self.redis_client.get(key)
                if cached_info:
                    info = json.loads(cached_info)
                    file_path = info["file_path"]
                    
                    # 检查文件是否存在，不存在则删除Redis记录
                    if not os.path.exists(file_path):
                        await self.redis_client.delete(key)
                        logger.info(f"清理无效缓存: {key}")
        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
    
    async def preload_common_cache(self, engine):
        """P0优化：预加载常用短语缓存"""
        try:
            logger.info("开始预加载TTS常用短语缓存...")
            preload_count = 0
            
            # 默认语音配置
            default_voices = ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"]
            
            for category, phrases in self.common_phrases.items():
                for phrase in phrases:
                    for voice_id in default_voices:
                        try:
                            # 检查是否已缓存
                            cache_key = self.generate_cache_key(phrase, voice_id, 1.0, 0.0, 1.0)
                            cached = await self.get_cached_audio(cache_key)
                            
                            if not cached:
                                # 生成音频
                                audio_data = await engine.synthesize(phrase, voice_id)
                                
                                # 缓存音频
                                metadata = {
                                    "duration": len(audio_data) / 48000,  # 估算时长
                                    "format": "mp3",
                                    "sample_rate": 24000,
                                    "voice_id": voice_id,
                                    "engine": "edge_tts",
                                    "category": category,
                                    "preloaded": True
                                }
                                await self.cache_audio(cache_key, audio_data, metadata)
                                preload_count += 1
                                
                                # 避免过快请求
                                await asyncio.sleep(0.1)
                                
                        except Exception as e:
                            logger.warning(f"预缓存失败 '{phrase}' ({voice_id}): {e}")
                            continue
            
            logger.info(f"TTS预缓存完成，共预生成 {preload_count} 个音频文件")
            
        except Exception as e:
            logger.error(f"TTS预缓存失败: {e}")
    
    def is_common_phrase(self, text: str) -> bool:
        """检查是否为常用短语"""
        text = text.strip()
        for phrases in self.common_phrases.values():
            if text in phrases:
                return True
        return False

class EdgeTTSEngine:
    """Edge TTS引擎"""
    
    def __init__(self):
        self.voices = {}
        self.load_voices()
    
    def load_voices(self):
        """加载可用语音"""
        # 常用中文语音
        self.voices = {
            "zh-CN-XiaoxiaoNeural": TTSVoice(TTSEngine.EDGE_TTS, "zh-CN-XiaoxiaoNeural", "zh-CN", "female", "晓晓"),
            "zh-CN-YunxiNeural": TTSVoice(TTSEngine.EDGE_TTS, "zh-CN-YunxiNeural", "zh-CN", "male", "云希"),
            "zh-CN-YunyangNeural": TTSVoice(TTSEngine.EDGE_TTS, "zh-CN-YunyangNeural", "zh-CN", "male", "云扬"),
            "zh-CN-XiaoyiNeural": TTSVoice(TTSEngine.EDGE_TTS, "zh-CN-XiaoyiNeural", "zh-CN", "female", "晓伊"),
        }
    
    async def synthesize(self, text: str, voice_id: str, speed: float = 1.0, pitch: float = 0.0, volume: float = 1.0) -> bytes:
        """合成语音"""
        try:
            # 构建SSML
            rate = f"{int((speed - 1) * 100):+d}%"
            pitch_str = f"{int(pitch * 50):+d}Hz"
            volume_str = f"{int(volume * 100)}%"
            
            communicate = edge_tts.Communicate(text, voice_id, rate=rate, pitch=pitch_str, volume=volume_str)
            
            # 收集音频数据
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Edge TTS合成失败: {e}")
            raise
    
    async def synthesize_stream(self, text: str, voice_id: str, speed: float = 1.0, pitch: float = 0.0, volume: float = 1.0) -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        try:
            # 构建SSML
            rate = f"{int((speed - 1) * 100):+d}%"
            pitch_str = f"{int(pitch * 50):+d}Hz"
            volume_str = f"{int(volume * 100)}%"
            
            communicate = edge_tts.Communicate(text, voice_id, rate=rate, pitch=pitch_str, volume=volume_str)
            
            # 流式返回音频数据
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
            
        except Exception as e:
            logger.error(f"Edge TTS流式合成失败: {e}")
            raise

class HuoshanTTSEngine:
    """火山引擎双流TTS引擎"""
    
    def __init__(self):
        self.voices = {}
        self.load_voices()
        self.api_url = "http://182.44.78.40:8002/api/v1/tts"  # 火山引擎API地址
    
    def load_voices(self):
        """加载可用语音"""
        # 火山引擎支持的语音
        self.voices = {
            "zh-CN-HuoshanNeural": TTSVoice(TTSEngine.HUOSHAN_TTS, "zh-CN-HuoshanNeural", "zh-CN", "female", "火山"),
            "zh-CN-HuoshanMaleNeural": TTSVoice(TTSEngine.HUOSHAN_TTS, "zh-CN-HuoshanMaleNeural", "zh-CN", "male", "火山男声"),
        }
    
    async def synthesize(self, text: str, voice_id: str, speed: float = 1.0, pitch: float = 0.0, volume: float = 1.0) -> bytes:
        """合成语音"""
        try:
            import aiohttp
            
            data = {
                "text": text,
                "voice": voice_id,
                "speed": speed,
                "pitch": pitch,
                "volume": volume,
                "format": "mp3"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        error_text = await response.text()
                        raise Exception(f"火山TTS API错误: {response.status} - {error_text}")
            
        except Exception as e:
            logger.error(f"火山TTS合成失败: {e}")
            raise
    
    async def synthesize_stream(self, text: str, voice_id: str, speed: float = 1.0, pitch: float = 0.0, volume: float = 1.0) -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        try:
            import aiohttp
            
            data = {
                "text": text,
                "voice": voice_id,
                "speed": speed,
                "pitch": pitch,
                "volume": volume,
                "format": "mp3",
                "stream": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        async for chunk in response.content.iter_chunked(8192):
                            yield chunk
                    else:
                        error_text = await response.text()
                        raise Exception(f"火山TTS流式API错误: {response.status} - {error_text}")
            
        except Exception as e:
            logger.error(f"火山TTS流式合成失败: {e}")
            raise

class TTSLoadBalancer:
    """TTS负载均衡器 - P0优化版本"""
    
    def __init__(self):
        self.engines = {
            TTSEngine.EDGE_TTS: EdgeTTSEngine(),
            TTSEngine.HUOSHAN_TTS: HuoshanTTSEngine()  # 新增火山引擎
        }
        
        # P0优化：权重配置调整
        # 注意：TTS引擎配置现在通过 http://182.44.78.40:8002/#/model-config 统一管理
        self.engine_weights = {
            TTSEngine.HUOSHAN_TTS: 1.0,   # 100%使用火山引擎TTS
            TTSEngine.EDGE_TTS: 0.0,      # EdgeTTS作为备用，权重设为0
        }
        
        # P0优化：引擎优先级配置
        self.engine_priority = {
            TTSEngine.HUOSHAN_TTS: 1,     # 最高优先级
            TTSEngine.EDGE_TTS: 2,        # 备用优先级
        }
        
        # P0优化：引擎超时配置
        self.engine_timeouts = {
            TTSEngine.HUOSHAN_TTS: 5,     # 5秒超时
            TTSEngine.EDGE_TTS: 2,        # 2秒超时
        }
        
        self.engine_stats = {
            TTSEngine.HUOSHAN_TTS: {
                "total_requests": 0,
                "total_errors": 0,
                "total_time": 0.0,
                "current_load": 0,
                "max_concurrent": 50,      # 火山引擎支持更高并发
                "timeout_count": 0,
                "error_count": 0
            },
            TTSEngine.EDGE_TTS: {
                "total_requests": 0,
                "total_errors": 0,
                "total_time": 0.0,
                "current_load": 0,
                "max_concurrent": 10,
                "timeout_count": 0,
                "error_count": 0
            }
        }
    
    def select_engine(self, voice_id: str, text_length: int = 0) -> TTSEngine:
        """选择TTS引擎 - 优先使用火山引擎，失败时回退到EdgeTTS"""
        # 首先尝试火山引擎
        huoshan_stats = self.engine_stats[TTSEngine.HUOSHAN_TTS]
        if (huoshan_stats["current_load"] < huoshan_stats["max_concurrent"] and 
            huoshan_stats["error_count"] < 5):  # 错误次数少于5次
            return TTSEngine.HUOSHAN_TTS
        
        # 回退到EdgeTTS
        logger.warning("火山TTS不可用，回退到EdgeTTS")
        return TTSEngine.EDGE_TTS


    
    def select_engine(self, voice_id: str, text_length: int = 0) -> TTSEngine:
        """P0优化：智能选择最佳TTS引擎"""
        # 1. 优先使用本地Edge TTS（80%权重）
        if text_length <= 500:  # 短文本优先本地
            edge_stats = self.engine_stats[TTSEngine.EDGE_TTS]
            if edge_stats.get("success_rate", 0.9) > 0.9:  # 成功率>90%
                return TTSEngine.EDGE_TTS
        
        # 2. 根据引擎健康状态选择
        available_engines = []
        for engine, stats in self.engine_stats.items():
            success_rate = 1.0 - (stats["total_errors"] / max(stats["total_requests"], 1))
            if success_rate > 0.8:  # 成功率>80%
                available_engines.append((engine, self.engine_priority[engine]))
        
        # 3. 按优先级排序，选择最高优先级的可用引擎
        if available_engines:
            available_engines.sort(key=lambda x: x[1])  # 按优先级排序
            return available_engines[0][0]
        
        # 4. 兜底：返回Edge TTS
        return TTSEngine.EDGE_TTS
    
    def get_engine(self, engine_type: TTSEngine):
        """获取引擎实例"""
        return self.engines.get(engine_type)
    
    def update_stats(self, engine_type: TTSEngine, processing_time: float, success: bool):
        """更新引擎统计"""
        stats = self.engine_stats[engine_type]
        stats["total_requests"] += 1
        stats["total_time"] += processing_time
        
        if not success:
            stats["total_errors"] += 1

class TTSService:
    """TTS微服务主类 - P0优化版本"""
    
    def __init__(self, max_concurrent: int = 40):  # P0优化：从20提升到40
        self.max_concurrent = max_concurrent
        self.load_balancer = TTSLoadBalancer()
        self.cache = None
        self.redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=12)  # P0优化：从8提升到12
        
        # 优先级队列
        self.high_priority_queue = asyncio.Queue()
        self.medium_priority_queue = asyncio.Queue()
        self.low_priority_queue = asyncio.Queue()
        
        # 性能统计
        self.total_requests = 0
        self.cache_hits = 0
        self.total_processing_time = 0.0
        self.current_concurrent = 0
        
        # 启动处理任务
        asyncio.create_task(self.process_queue())
        asyncio.create_task(self.cleanup_task())
    
    async def init_redis(self, redis_url: str = "redis://localhost:6379"):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            self.cache = TTSCache(self.redis_client)
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
    
    async def process_queue(self):
        """处理队列任务"""
        while True:
            try:
                # 检查并发限制
                if self.current_concurrent >= self.max_concurrent:
                    await asyncio.sleep(0.1)
                    continue
                
                # 从优先级队列获取请求
                request = await self.get_next_request()
                if request:
                    asyncio.create_task(self.process_single_request(request))
                else:
                    await asyncio.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"队列处理错误: {e}")
                await asyncio.sleep(0.1)
    
    async def get_next_request(self) -> Optional[TTSRequest]:
        """从优先级队列获取下一个请求"""
        timeout = 0.1
        
        for queue in [self.high_priority_queue, self.medium_priority_queue, self.low_priority_queue]:
            try:
                request = await asyncio.wait_for(queue.get(), timeout=timeout)
                return request
            except asyncio.TimeoutError:
                continue
        
        return None
    
    async def process_single_request(self, request: TTSRequest):
        """处理单个TTS请求"""
        self.current_concurrent += 1
        start_time = time.time()
        
        try:
            result = await self.synthesize_speech(request)
            processing_time = time.time() - start_time
            
            # 更新统计
            self.total_requests += 1
            self.total_processing_time += processing_time
            
            # 这里应该将结果发送给客户端（简化实现）
            logger.info(f"TTS处理完成: {request.session_id}, 耗时: {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"TTS处理失败: {e}")
        finally:
            self.current_concurrent -= 1
    
    async def synthesize_speech(self, request: TTSRequest) -> TTSResponse:
        """P0优化：合成语音 - 本地优先策略"""
        start_time = time.time()
        
        try:
            # 生成缓存键
            voice_id = request.voice_id or "zh-CN-XiaoxiaoNeural"
            cache_key = None
            
            if request.cache_enabled and self.cache:
                cache_key = self.cache.generate_cache_key(
                    request.text,
                    voice_id,
                    request.speed,
                    request.pitch,
                    request.volume
                )
                
                # P0优化：优先检查常用短语缓存
                if self.cache.is_common_phrase(request.text):
                    cached_audio = await self.cache.get_cached_audio(cache_key)
                    if cached_audio:
                        self.cache_hits += 1
                        logger.info(f"常用短语缓存命中: {request.text}")
                        return TTSResponse(
                            session_id=request.session_id,
                            audio_data=base64.b64encode(cached_audio["audio_data"]).decode(),
                            duration=cached_audio["duration"],
                            format=cached_audio["format"],
                            sample_rate=cached_audio["sample_rate"],
                            file_size=cached_audio["file_size"],
                            processing_time=time.time() - start_time,
                            cached=True,
                            voice_id=voice_id,
                            engine=cached_audio["engine"]
                        )
                
                # 检查普通缓存
                cached_audio = await self.cache.get_cached_audio(cache_key)
                if cached_audio:
                    self.cache_hits += 1
                    logger.info(f"TTS缓存命中: {request.text[:20]}...")
                    return TTSResponse(
                        session_id=request.session_id,
                        audio_data=base64.b64encode(cached_audio["audio_data"]).decode(),
                        duration=cached_audio["duration"],
                        format=cached_audio["format"],
                        sample_rate=cached_audio["sample_rate"],
                        file_size=cached_audio["file_size"],
                        processing_time=time.time() - start_time,
                        cached=True,
                        voice_id=voice_id,
                        engine=cached_audio["engine"]
                    )
            
            # P0优化：智能选择TTS引擎（本地优先）
            text_length = len(request.text)
            engine_type = self.load_balancer.select_engine(voice_id, text_length)
            engine = self.load_balancer.get_engine(engine_type)
            
            if not engine:
                raise Exception(f"不支持的TTS引擎: {engine_type}")
            
            # 合成语音
            audio_data = await engine.synthesize(
                request.text,
                voice_id,
                request.speed,
                request.pitch,
                request.volume
            )
            
            processing_time = time.time() - start_time
            
            # 计算音频时长（简化实现）
            duration = len(audio_data) / (request.sample_rate * 2)  # 假设16位音频
            
            # 更新引擎统计
            self.load_balancer.update_stats(engine_type, processing_time, True)
            
            # 缓存音频
            if request.cache_enabled and self.cache:
                metadata = {
                    "duration": duration,
                    "format": request.format,
                    "sample_rate": request.sample_rate,
                    "voice_id": voice_id,
                    "engine": engine_type.value
                }
                await self.cache.cache_audio(cache_key, audio_data, metadata)
            
            return TTSResponse(
                session_id=request.session_id,
                audio_data=base64.b64encode(audio_data).decode(),
                duration=duration,
                format=request.format,
                sample_rate=request.sample_rate,
                file_size=len(audio_data),
                processing_time=processing_time,
                cached=False,
                voice_id=voice_id,
                engine=engine_type.value
            )
            
        except Exception as e:
            # 更新引擎统计
            if 'engine_type' in locals():
                self.load_balancer.update_stats(engine_type, 0, False)
            
            logger.error(f"语音合成失败: {e}")
            raise
    
    async def add_request(self, request: TTSRequest):
        """添加TTS请求到优先级队列"""
        if request.priority == 1:
            await self.high_priority_queue.put(request)
        elif request.priority == 2:
            await self.medium_priority_queue.put(request)
        else:
            await self.low_priority_queue.put(request)
    
    async def cleanup_task(self):
        """定期清理任务"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时执行一次
                if self.cache:
                    await self.cache.cleanup_expired_cache()
            except Exception as e:
                logger.error(f"清理任务失败: {e}")
    
    def get_stats(self) -> Dict:
        """获取服务统计"""
        avg_processing_time = self.total_processing_time / max(self.total_requests, 1)
        cache_hit_rate = self.cache_hits / max(self.total_requests, 1)
        
        engine_stats = []
        for engine_type, stats in self.load_balancer.engine_stats.items():
            avg_time = stats["total_time"] / max(stats["total_requests"], 1)
            error_rate = stats["total_errors"] / max(stats["total_requests"], 1)
            
            engine_stats.append({
                "engine": engine_type.value,
                "total_requests": stats["total_requests"],
                "total_errors": stats["total_errors"],
                "error_rate": error_rate,
                "avg_processing_time": avg_time,
                "current_load": stats["current_load"],
                "max_concurrent": stats["max_concurrent"]
            })
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_processing_time": avg_processing_time,
            "current_concurrent": self.current_concurrent,
            "max_concurrent": self.max_concurrent,
            "queue_sizes": {
                "high_priority": self.high_priority_queue.qsize(),
                "medium_priority": self.medium_priority_queue.qsize(),
                "low_priority": self.low_priority_queue.qsize()
            },
            "engines": engine_stats
        }

# FastAPI应用
app = FastAPI(title="TTS Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局TTS服务实例 - P0优化配置
tts_service = TTSService(max_concurrent=40)  # P0优化：从30提升到40

@app.on_event("startup")
async def startup_event():
    """启动事件 - P0优化版本"""
    await tts_service.init_redis()
    
    # P0优化：启动时预加载常用短语缓存
    if tts_service.cache and tts_service.load_balancer.engines:
        edge_engine = tts_service.load_balancer.get_engine(TTSEngine.EDGE_TTS)
        if edge_engine:
            await tts_service.cache.preload_common_cache(edge_engine)
            logger.info("TTS服务启动完成，预缓存已加载")

@app.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """TTS语音合成API"""
    try:
        response = await tts_service.synthesize_speech(request)
        return response
    except Exception as e:
        logger.error(f"TTS合成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/synthesize_stream")
async def synthesize_speech_stream(request: TTSRequest):
    """TTS流式语音合成API - 真正的流式传输"""
    try:
        async def audio_stream_generator():
            """音频流生成器"""
            voice_id = request.voice_id or "zh-CN-XiaoxiaoNeural"
            engine_type = tts_service.load_balancer.select_engine(voice_id)
            engine = tts_service.load_balancer.get_engine(engine_type)
            
            if not engine:
                raise Exception(f"不支持的TTS引擎: {engine_type}")
            
            # 使用流式合成
            async for audio_chunk in engine.synthesize_stream(
                request.text,
                voice_id,
                request.speed,
                request.pitch,
                request.volume
            ):
                yield audio_chunk
        
        return StreamingResponse(
            audio_stream_generator(),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=tts_{request.session_id}.mp3",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
    except Exception as e:
        logger.error(f"TTS流式合成失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts/voices")
async def get_available_voices():
    """获取可用语音列表"""
    edge_engine = tts_service.load_balancer.get_engine(TTSEngine.EDGE_TTS)
    voices = []
    
    for voice_id, voice in edge_engine.voices.items():
        voices.append({
            "voice_id": voice_id,
            "name": voice.name,
            "language": voice.language,
            "gender": voice.gender,
            "engine": voice.engine.value
        })
    
    return {"voices": voices}

@app.get("/tts/stats")
async def get_stats():
    """获取服务统计信息"""
    return tts_service.get_stats()

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "tts",
        "current_concurrent": tts_service.current_concurrent,
        "max_concurrent": tts_service.max_concurrent
    }

@app.get("/xiaozhi/ota/")
async def ota_version_check():
    """OTA版本检查端点 - ESP32设备用于检查固件更新"""
    return {
        "firmware": {
            "version": "1.0.0",
            "url": "https://api.tenclass.net/xiaozhi/ota/firmware.bin"
        },
        "server_time": {
            "timestamp": int(time.time() * 1000),  # 毫秒时间戳
            "timezone_offset": 480  # UTC+8 (中国时区)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, workers=1)