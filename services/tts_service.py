#!/usr/bin/env python3
"""
TTS (Text-to-Speech) 微服务
支持并发处理、音频缓存、多引擎负载均衡和流式传输
极限优化版本：支持80-100台设备并发
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
    HUOSHAN_TTS = "huoshan_tts"  # 火山引擎TTS (双流式)

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
    format: str = "opus"  # 极限优化：默认使用opus格式提高压缩效率
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
    """TTS音频缓存管理器 - 极限优化版本"""
    
    def __init__(self, redis_client, cache_dir: str = "/tmp/tts_cache"):
        self.redis_client = redis_client
        self.cache_dir = cache_dir
        # 极限优化：从环境变量读取配置
        self.cache_ttl = int(os.getenv("TTS_CACHE_TTL", "7200"))  # 2小时TTL
        self.max_file_size = int(os.getenv("TTS_MAX_FILE_SIZE", "20")) * 1024 * 1024  # 20MB
        self.compression_enabled = os.getenv("TTS_ENABLE_COMPRESSION", "true").lower() == "true"
        self.preload_enabled = os.getenv("TTS_ENABLE_PRELOAD", "true").lower() == "true"
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 极限优化：扩展预缓存常用短语
        self.common_phrases = {
            "greetings": ["你好", "早上好", "下午好", "晚上好", "欢迎使用", "很高兴为您服务"],
            "confirmations": ["好的", "明白了", "收到", "没问题", "可以", "当然"],
            "questions": ["有什么可以帮您的吗", "还有其他问题吗", "需要我做什么", "请问您需要什么"],
            "responses": ["正在处理", "请稍等", "马上为您处理", "正在为您查询", "处理完成"],
            "errors": ["抱歉", "出现了问题", "请重试", "系统繁忙", "连接失败"],
            "numbers": [str(i) for i in range(100)],  # 0-99数字
            "time": ["点", "分", "秒", "上午", "下午", "今天", "明天", "昨天"],
        }
        
        # 极限优化：缓存统计
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "preload_hits": 0,
            "total_size": 0,
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
    """火山引擎TTS - 双流式语音合成"""
    
    def __init__(self):
        # 极限优化：从环境变量读取火山引擎配置
        self.api_key = os.getenv("HUOSHAN_TTS_API_KEY", "")
        self.app_id = os.getenv("HUOSHAN_TTS_APP_ID", "")
        self.cluster = os.getenv("HUOSHAN_TTS_CLUSTER", "volcano_tts")
        self.voice_type = os.getenv("HUOSHAN_TTS_VOICE_TYPE", "BV700_streaming")  # 双流式
        self.enabled = bool(self.api_key and self.app_id)
        
        # 极限优化：双流式配置
        self.stream_enabled = True
        self.chunk_size = int(os.getenv("HUOSHAN_TTS_CHUNK_SIZE", "1024"))
        self.sample_rate = int(os.getenv("HUOSHAN_TTS_SAMPLE_RATE", "24000"))
        
        if self.enabled:
            logger.info("🔥 火山引擎双流式TTS已启用")
        else:
            logger.warning("⚠️ 火山引擎TTS配置缺失，使用Edge TTS作为备用")

    def load_voices(self):
        """加载火山引擎语音列表"""
        if not self.enabled:
            return {}
        
        return {
            "BV700_streaming": TTSVoice(
                engine=TTSEngine.HUOSHAN_TTS,
                voice_id="BV700_streaming",
                language="zh-CN",
                gender="female",
                name="火山双流式女声",
                sample_rate=self.sample_rate,
                quality="high"
            ),
            "BV701_streaming": TTSVoice(
                engine=TTSEngine.HUOSHAN_TTS,
                voice_id="BV701_streaming", 
                language="zh-CN",
                gender="male",
                name="火山双流式男声",
                sample_rate=self.sample_rate,
                quality="high"
            )
        }

    async def synthesize(self, text: str, voice_id: str, speed: float = 1.0, pitch: float = 0.0, volume: float = 1.0) -> bytes:
        """火山引擎TTS合成 - 双流式"""
        if not self.enabled:
            raise Exception("火山引擎TTS未配置")
        
        try:
            # 极限优化：使用双流式API
            import requests
            
            url = f"https://openspeech.bytedance.com/api/v1/tts"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "app": {
                    "appid": self.app_id,
                    "token": self.api_key,
                    "cluster": self.cluster
                },
                "user": {
                    "uid": "xiaozhi_user"
                },
                "audio": {
                    "voice_type": voice_id,
                    "encoding": "opus",  # 极限优化：使用opus编码
                    "speed_ratio": speed,
                    "volume_ratio": volume,
                    "pitch_ratio": pitch,
                    "sample_rate": self.sample_rate
                },
                "request": {
                    "reqid": f"xiaozhi_{int(time.time())}",
                    "text": text,
                    "text_type": "plain",
                    "operation": "submit"
                }
            }
            
            # 极限优化：异步请求
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: requests.post(url, json=payload, headers=headers, timeout=10)
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 3000:
                    # 获取音频数据
                    audio_data = base64.b64decode(result["data"])
                    return audio_data
                else:
                    raise Exception(f"火山引擎TTS错误: {result.get('message', '未知错误')}")
            else:
                raise Exception(f"火山引擎TTS请求失败: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ 火山引擎TTS合成失败: {e}")
            raise

    async def synthesize_stream(self, text: str, voice_id: str, speed: float = 1.0, pitch: float = 0.0, volume: float = 1.0) -> AsyncGenerator[bytes, None]:
        """火山引擎双流式TTS合成"""
        if not self.enabled:
            raise Exception("火山引擎TTS未配置")
        
        try:
            # 极限优化：双流式实现
            audio_data = await self.synthesize(text, voice_id, speed, pitch, volume)
            
            # 分块流式返回
            chunk_size = self.chunk_size
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.001)  # 极限优化：最小延迟
                
        except Exception as e:
            logger.error(f"❌ 火山引擎双流式TTS失败: {e}")
            raise

class TTSLoadBalancer:
    """TTS负载均衡器 - 极限优化版本"""
    
    def __init__(self):
        # 极限优化：动态引擎权重配置
        self.engines = {
            TTSEngine.HUOSHAN_TTS: {
                "weight": float(os.getenv("HUOSHAN_TTS_WEIGHT", "0.8")),  # 优先火山引擎
                "max_concurrent": int(os.getenv("HUOSHAN_TTS_MAX_CONCURRENT", "60")),
                "current_load": 0,
                "total_requests": 0,
                "success_rate": 1.0,
                "avg_latency": 0.0,
                "enabled": True
            },
            TTSEngine.EDGE_TTS: {
                "weight": float(os.getenv("EDGE_TTS_WEIGHT", "0.2")),  # 备用引擎
                "max_concurrent": int(os.getenv("EDGE_TTS_MAX_CONCURRENT", "40")),
                "current_load": 0,
                "total_requests": 0,
                "success_rate": 1.0,
                "avg_latency": 0.0,
                "enabled": True
            }
        }
        
        # 初始化引擎实例
        self.engine_instances = {
            TTSEngine.HUOSHAN_TTS: HuoshanTTSEngine(),
            TTSEngine.EDGE_TTS: EdgeTTSEngine(),
        }
        
        # 极限优化：智能路由配置
        self.enable_smart_routing = os.getenv("TTS_SMART_ROUTING", "true").lower() == "true"
        self.failover_enabled = os.getenv("TTS_FAILOVER_ENABLED", "true").lower() == "true"

    def select_engine(self, voice_id: str, text_length: int = 0) -> TTSEngine:
        """智能选择TTS引擎 - 极限优化"""
        try:
            # 极限优化：优先使用火山引擎（如果配置了）
            huoshan_engine = self.engine_instances[TTSEngine.HUOSHAN_TTS]
            if huoshan_engine.enabled and self.engines[TTSEngine.HUOSHAN_TTS]["enabled"]:
                huoshan_load = self.engines[TTSEngine.HUOSHAN_TTS]["current_load"]
                huoshan_max = self.engines[TTSEngine.HUOSHAN_TTS]["max_concurrent"]
                
                if huoshan_load < huoshan_max:
                    return TTSEngine.HUOSHAN_TTS
            
            # 备用：使用Edge TTS
            edge_load = self.engines[TTSEngine.EDGE_TTS]["current_load"]
            edge_max = self.engines[TTSEngine.EDGE_TTS]["max_concurrent"]
            
            if edge_load < edge_max:
                return TTSEngine.EDGE_TTS
            
            # 极限优化：如果都满载，选择负载较低的
            if self.enable_smart_routing:
                huoshan_ratio = huoshan_load / huoshan_max if huoshan_max > 0 else 1.0
                edge_ratio = edge_load / edge_max if edge_max > 0 else 1.0
                
                return TTSEngine.HUOSHAN_TTS if huoshan_ratio <= edge_ratio else TTSEngine.EDGE_TTS
            
            # 默认返回火山引擎
            return TTSEngine.HUOSHAN_TTS
            
        except Exception as e:
            logger.warning(f"⚠️ 引擎选择失败，使用默认: {e}")
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
    """极限优化的TTS服务，支持80-100台设备并发"""
    
    def __init__(self, max_concurrent: int = None):
        # 极限优化：从环境变量读取配置
        self.max_concurrent = max_concurrent or int(os.getenv("TTS_MAX_CONCURRENT", "80"))  # 提升到80
        self.queue_size = int(os.getenv("TTS_QUEUE_SIZE", "200"))  # 提升队列大小
        self.worker_threads = int(os.getenv("TTS_WORKER_THREADS", "8"))  # 增加工作线程
        self.batch_size = int(os.getenv("TTS_BATCH_SIZE", "16"))  # 批处理大小
        self.batch_timeout = float(os.getenv("TTS_BATCH_TIMEOUT", "100")) / 1000  # 100ms
        
        # 初始化组件
        self.load_balancer = TTSLoadBalancer()
        self.redis_client = None
        self.cache = None
        
        # 极限优化：多优先级队列
        self.high_priority_queue = asyncio.Queue(maxsize=self.queue_size // 3)
        self.medium_priority_queue = asyncio.Queue(maxsize=self.queue_size // 2)
        self.low_priority_queue = asyncio.Queue(maxsize=self.queue_size)
        
        # 极限优化：线程池配置
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.worker_threads,
            thread_name_prefix="TTS-Worker"
        )
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "average_processing_time": 0.0,
            "current_concurrent": 0,
            "max_concurrent": self.max_concurrent,
            "queue_sizes": {"high": 0, "medium": 0, "low": 0},
            "engine_stats": {}
        }
        
        # 启动后台任务
        asyncio.create_task(self.process_queue())
        asyncio.create_task(self.performance_monitor())
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