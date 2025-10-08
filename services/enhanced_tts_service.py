"""
增强版TTS服务
集成远程API调用、智能路由、故障转移和缓存
"""

import asyncio
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.remote_api_config import RemoteAPIConfig, APIProvider
from core.intelligent_router import IntelligentRouter, RoutingStrategy, RequestPriority, RoutingContext
from core.queue_manager import get_queue_manager, QueueRequest, Priority
from core.redis_client import RedisClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFormat(str, Enum):
    """音频格式"""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    AAC = "aac"

class VoiceGender(str, Enum):
    """声音性别"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"

class TTSRequest(BaseModel):
    """TTS请求模型"""
    text: str = Field(..., description="要合成的文本")
    voice: Optional[str] = Field(None, description="声音ID")
    language: str = Field("zh-CN", description="语言代码")
    speed: float = Field(1.0, ge=0.1, le=3.0, description="语速")
    pitch: float = Field(1.0, ge=0.1, le=2.0, description="音调")
    volume: float = Field(1.0, ge=0.1, le=2.0, description="音量")
    format: AudioFormat = Field(AudioFormat.MP3, description="音频格式")
    quality: str = Field("standard", description="音质等级")
    priority: RequestPriority = Field(RequestPriority.MEDIUM, description="请求优先级")
    cache_enabled: bool = Field(True, description="是否启用缓存")
    streaming: bool = Field(False, description="是否流式输出")

class TTSResponse(BaseModel):
    """TTS响应模型"""
    success: bool
    audio_data: Optional[bytes] = None
    audio_url: Optional[str] = None
    format: AudioFormat
    duration: Optional[float] = None
    provider: Optional[str] = None
    voice_used: Optional[str] = None
    cost: Optional[float] = None
    cached: bool = False
    processing_time: float = 0.0
    error: Optional[str] = None

@dataclass
class TTSMetrics:
    """TTS服务指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    total_response_time: float = 0.0
    total_cost: float = 0.0
    provider_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

class EnhancedTTSService:
    """增强版TTS服务"""
    
    def __init__(self, max_concurrent: int = 50):
        self.max_concurrent = max_concurrent
        self.current_requests = 0
        self.api_config: Optional[RemoteAPIConfig] = None
        self.router: Optional[IntelligentRouter] = None
        self.redis_client: Optional[RedisClient] = None
        self.queue_manager = None
        self.metrics = TTSMetrics()
        
        # 熔断器配置
        self.circuit_breaker = {
            "state": "CLOSED",  # CLOSED, OPEN, HALF_OPEN
            "failure_count": 0,
            "failure_threshold": 5,
            "recovery_timeout": 60,
            "last_failure_time": 0
        }
        
        # 本地TTS引擎配置
        self.local_engines = {
            "edge_tts": {
                "enabled": True,
                "voices": {
                    "zh-CN": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
                    "en-US": ["en-US-AriaNeural", "en-US-JennyNeural"]
                }
            }
        }
    
    async def initialize(self):
        """初始化服务"""
        try:
            # 初始化API配置
            self.api_config = RemoteAPIConfig()
            await self.api_config.initialize()
            
            # 初始化智能路由器
            self.router = IntelligentRouter()
            
            # 初始化Redis客户端
            self.redis_client = RedisClient()
            await self.redis_client.initialize()
            
            # 初始化队列管理器
            self.queue_manager = get_queue_manager(
                max_queue_size=2000,
                batch_timeout=0.1,  # 100ms
                batch_size=10,
                max_concurrent=self.max_concurrent
            )
            await self.queue_manager.start()
            
            # 启动后台任务
            asyncio.create_task(self._health_check_task())
            asyncio.create_task(self._metrics_collection_task())
            asyncio.create_task(self._cache_cleanup_task())
            
            logger.info("增强版TTS服务初始化完成")
            
        except Exception as e:
            logger.error(f"TTS服务初始化失败: {e}")
            raise
    
    async def process_request(self, request: TTSRequest) -> TTSResponse:
        """处理TTS请求"""
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            # 检查熔断器状态
            if not self._check_circuit_breaker():
                raise HTTPException(status_code=503, detail="服务暂时不可用")
            
            # 检查并发限制
            if self.current_requests >= self.max_concurrent:
                # 使用队列管理器
                queue_request = QueueRequest(
                    request_id=f"tts_{int(time.time() * 1000)}",
                    priority=Priority.HIGH if request.priority == RequestPriority.HIGH else Priority.MEDIUM,
                    data=request.dict(),
                    created_at=time.time()
                )
                
                result = await self.queue_manager.add_request(queue_request)
                if not result:
                    raise HTTPException(status_code=429, detail="请求队列已满")
                
                # 等待处理
                await asyncio.sleep(0.1)
            
            self.current_requests += 1
            
            try:
                # 检查缓存
                if request.cache_enabled:
                    cached_response = await self._get_cached_response(request)
                    if cached_response:
                        self.metrics.cache_hits += 1
                        cached_response.cached = True
                        cached_response.processing_time = time.time() - start_time
                        return cached_response
                
                # 选择最佳端点
                routing_context = RoutingContext(
                    request_type="tts",
                    priority=request.priority,
                    text_length=len(request.text),
                    language=request.language,
                    quality=request.quality
                )
                
                endpoint = await self.router.select_tts_endpoint(
                    self.api_config.tts_endpoints,
                    routing_context
                )
                
                if not endpoint:
                    # 回退到本地引擎
                    return await self._process_with_local_engine(request, start_time)
                
                # 调用远程API
                response = await self._call_tts_api(endpoint, request)
                
                # 更新负载
                await self.router.update_load(endpoint, 1)
                
                # 缓存响应
                if request.cache_enabled and response.success:
                    await self._cache_response(request, response)
                
                # 更新统计
                processing_time = time.time() - start_time
                cost = self._calculate_cost(endpoint, len(request.text))
                self._update_provider_stats(endpoint.provider, processing_time, response.success, cost)
                
                response.processing_time = processing_time
                response.cost = cost
                response.provider = endpoint.provider.value
                
                self.metrics.successful_requests += 1
                self._reset_circuit_breaker()
                
                return response
                
            except Exception as e:
                logger.error(f"TTS请求处理失败: {e}")
                self.metrics.failed_requests += 1
                self._record_failure()
                
                # 尝试回退到本地引擎
                try:
                    return await self._process_with_local_engine(request, start_time)
                except Exception as fallback_error:
                    logger.error(f"本地引擎回退失败: {fallback_error}")
                    return TTSResponse(
                        success=False,
                        format=request.format,
                        processing_time=time.time() - start_time,
                        error=str(e)
                    )
            
            finally:
                self.current_requests -= 1
                
        except Exception as e:
            logger.error(f"TTS请求处理异常: {e}")
            return TTSResponse(
                success=False,
                format=request.format,
                processing_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _process_with_local_engine(self, request: TTSRequest, start_time: float) -> TTSResponse:
        """使用本地引擎处理"""
        try:
            # 这里实现本地Edge TTS引擎调用
            import edge_tts
            
            # 选择合适的声音
            voice = request.voice
            if not voice:
                voices = self.local_engines["edge_tts"]["voices"].get(request.language, [])
                voice = voices[0] if voices else "zh-CN-XiaoxiaoNeural"
            
            # 生成语音
            communicate = edge_tts.Communicate(request.text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return TTSResponse(
                success=True,
                audio_data=audio_data,
                format=request.format,
                provider="edge_tts",
                voice_used=voice,
                processing_time=time.time() - start_time,
                cost=0.0  # 本地引擎免费
            )
            
        except Exception as e:
            logger.error(f"本地引擎处理失败: {e}")
            raise
    
    async def _call_tts_api(self, endpoint, request: TTSRequest) -> TTSResponse:
        """调用TTS API"""
        try:
            session = await self.api_config.get_session(endpoint.provider)
            
            if endpoint.provider == APIProvider.AZURE_TTS:
                return await self._call_azure_tts(session, endpoint, request)
            elif endpoint.provider == APIProvider.GOOGLE_TTS:
                return await self._call_google_tts(session, endpoint, request)
            elif endpoint.provider == APIProvider.AWS_POLLY:
                return await self._call_aws_polly(session, endpoint, request)
            elif endpoint.provider == APIProvider.BAIDU_TTS:
                return await self._call_baidu_tts(session, endpoint, request)
            elif endpoint.provider == APIProvider.XUNFEI_TTS:
                return await self._call_xunfei_tts(session, endpoint, request)
            else:
                raise ValueError(f"不支持的TTS提供商: {endpoint.provider}")
                
        except Exception as e:
            logger.error(f"调用TTS API失败: {e}")
            raise
    
    async def _call_azure_tts(self, session: aiohttp.ClientSession, endpoint, request: TTSRequest) -> TTSResponse:
        """调用Azure TTS"""
        headers = {
            "Ocp-Apim-Subscription-Key": endpoint.api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-16khz-128kbitrate-mono-mp3"
        }
        
        # 构建SSML
        ssml = f"""
        <speak version='1.0' xml:lang='{request.language}'>
            <voice xml:lang='{request.language}' name='{request.voice or "zh-CN-XiaoxiaoNeural"}'>
                <prosody rate='{request.speed}' pitch='{request.pitch}' volume='{request.volume}'>
                    {request.text}
                </prosody>
            </voice>
        </speak>
        """
        
        async with session.post(
            f"{endpoint.base_url}/cognitiveservices/v1",
            headers=headers,
            data=ssml.encode('utf-8'),
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                audio_data = await response.read()
                return TTSResponse(
                    success=True,
                    audio_data=audio_data,
                    format=request.format,
                    voice_used=request.voice or "zh-CN-XiaoxiaoNeural"
                )
            else:
                error_text = await response.text()
                raise Exception(f"Azure TTS API错误: {response.status} - {error_text}")
    
    async def _call_google_tts(self, session: aiohttp.ClientSession, endpoint, request: TTSRequest) -> TTSResponse:
        """调用Google TTS"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": {"text": request.text},
            "voice": {
                "languageCode": request.language,
                "name": request.voice or "zh-CN-Wavenet-A"
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": request.speed,
                "pitch": request.pitch,
                "volumeGainDb": request.volume
            }
        }
        
        async with session.post(
            f"{endpoint.base_url}/v1/text:synthesize",
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                result = await response.json()
                import base64
                audio_data = base64.b64decode(result["audioContent"])
                return TTSResponse(
                    success=True,
                    audio_data=audio_data,
                    format=request.format,
                    voice_used=request.voice or "zh-CN-Wavenet-A"
                )
            else:
                error_text = await response.text()
                raise Exception(f"Google TTS API错误: {response.status} - {error_text}")
    
    async def _call_aws_polly(self, session: aiohttp.ClientSession, endpoint, request: TTSRequest) -> TTSResponse:
        """调用AWS Polly"""
        # AWS Polly需要特殊的签名认证，这里简化实现
        headers = {
            "Authorization": f"AWS4-HMAC-SHA256 {endpoint.api_key}",
            "Content-Type": "application/x-amz-json-1.0"
        }
        
        data = {
            "Text": request.text,
            "OutputFormat": "mp3",
            "VoiceId": request.voice or "Zhiyu",
            "LanguageCode": request.language
        }
        
        async with session.post(
            f"{endpoint.base_url}/v1/speech",
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                audio_data = await response.read()
                return TTSResponse(
                    success=True,
                    audio_data=audio_data,
                    format=request.format,
                    voice_used=request.voice or "Zhiyu"
                )
            else:
                error_text = await response.text()
                raise Exception(f"AWS Polly API错误: {response.status} - {error_text}")
    
    async def _call_baidu_tts(self, session: aiohttp.ClientSession, endpoint, request: TTSRequest) -> TTSResponse:
        """调用百度TTS"""
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "tex": request.text,
            "tok": endpoint.api_key,
            "cuid": "xiaozhi_tts",
            "ctp": "1",
            "lan": "zh",
            "spd": int(request.speed * 5),  # 百度TTS速度范围0-15
            "pit": int(request.pitch * 5),  # 百度TTS音调范围0-15
            "vol": int(request.volume * 15), # 百度TTS音量范围0-15
            "per": 4  # 声音选择
        }
        
        async with session.post(
            f"{endpoint.base_url}/text2audio",
            headers=headers,
            data=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                content_type = response.headers.get('Content-Type', '')
                if 'audio' in content_type:
                    audio_data = await response.read()
                    return TTSResponse(
                        success=True,
                        audio_data=audio_data,
                        format=request.format,
                        voice_used="baidu_voice"
                    )
                else:
                    error_text = await response.text()
                    raise Exception(f"百度TTS API错误: {error_text}")
            else:
                error_text = await response.text()
                raise Exception(f"百度TTS API错误: {response.status} - {error_text}")
    
    async def _call_xunfei_tts(self, session: aiohttp.ClientSession, endpoint, request: TTSRequest) -> TTSResponse:
        """调用讯飞TTS"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text": request.text,
            "voice": request.voice or "xiaoyan",
            "speed": int(request.speed * 50),
            "pitch": int(request.pitch * 50),
            "volume": int(request.volume * 100),
            "format": "mp3"
        }
        
        async with session.post(
            f"{endpoint.base_url}/v1/tts",
            headers=headers,
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                result = await response.json()
                if result.get("code") == 0:
                    import base64
                    audio_data = base64.b64decode(result["data"]["audio"])
                    return TTSResponse(
                        success=True,
                        audio_data=audio_data,
                        format=request.format,
                        voice_used=request.voice or "xiaoyan"
                    )
                else:
                    raise Exception(f"讯飞TTS API错误: {result.get('message', '未知错误')}")
            else:
                error_text = await response.text()
                raise Exception(f"讯飞TTS API错误: {response.status} - {error_text}")
    
    def _calculate_cost(self, endpoint, text_length: int) -> float:
        """计算成本"""
        # 按字符数计算成本
        return (text_length / 1000) * endpoint.cost_per_1k_tokens
    
    async def _get_cached_response(self, request: TTSRequest) -> Optional[TTSResponse]:
        """获取缓存的响应"""
        try:
            cache_key = self._generate_cache_key(request)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                data = json.loads(cached_data)
                return TTSResponse(**data)
            
            return None
            
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
            return None
    
    async def _cache_response(self, request: TTSRequest, response: TTSResponse):
        """缓存响应"""
        try:
            cache_key = self._generate_cache_key(request)
            
            # 不缓存音频数据，只缓存元数据
            cache_data = response.dict()
            cache_data["audio_data"] = None  # 移除音频数据
            
            await self.redis_client.set_with_ttl(
                cache_key,
                json.dumps(cache_data),
                3600  # 1小时
            )
            
        except Exception as e:
            logger.error(f"缓存响应失败: {e}")
    
    def _generate_cache_key(self, request: TTSRequest) -> str:
        """生成缓存键"""
        key_data = f"{request.text}_{request.voice}_{request.language}_{request.speed}_{request.pitch}_{request.volume}_{request.format}_{request.quality}"
        return f"tts_cache_{hashlib.md5(key_data.encode()).hexdigest()}"
    
    def _check_circuit_breaker(self) -> bool:
        """检查熔断器状态"""
        current_time = time.time()
        
        if self.circuit_breaker["state"] == "OPEN":
            if current_time - self.circuit_breaker["last_failure_time"] > self.circuit_breaker["recovery_timeout"]:
                self.circuit_breaker["state"] = "HALF_OPEN"
                return True
            return False
        
        return True
    
    def _record_failure(self):
        """记录失败"""
        self.circuit_breaker["failure_count"] += 1
        self.circuit_breaker["last_failure_time"] = time.time()
        
        if self.circuit_breaker["failure_count"] >= self.circuit_breaker["failure_threshold"]:
            self.circuit_breaker["state"] = "OPEN"
    
    def _reset_circuit_breaker(self):
        """重置熔断器"""
        if self.circuit_breaker["state"] == "HALF_OPEN":
            self.circuit_breaker["state"] = "CLOSED"
            self.circuit_breaker["failure_count"] = 0
    
    def _update_provider_stats(self, provider: APIProvider, response_time: float, success: bool, cost: float):
        """更新提供商统计"""
        provider_name = provider.value
        if provider_name not in self.metrics.provider_stats:
            self.metrics.provider_stats[provider_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_response_time": 0.0,
                "total_cost": 0.0,
                "avg_response_time": 0.0
            }
        
        stats = self.metrics.provider_stats[provider_name]
        stats["total_requests"] += 1
        stats["total_response_time"] += response_time
        stats["total_cost"] += cost
        
        if success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1
        
        stats["avg_response_time"] = stats["total_response_time"] / stats["total_requests"]
    
    async def _health_check_task(self):
        """健康检查任务"""
        while True:
            try:
                await asyncio.sleep(30)  # 每30秒检查一次
                await self.api_config.health_check()
                await self.router.optimize_routing_weights()
            except Exception as e:
                logger.error(f"健康检查失败: {e}")
    
    async def _metrics_collection_task(self):
        """指标收集任务"""
        while True:
            try:
                await asyncio.sleep(60)  # 每分钟收集一次指标
                
                # 计算总体指标
                self.metrics.total_response_time = sum(
                    stats["total_response_time"] 
                    for stats in self.metrics.provider_stats.values()
                )
                self.metrics.total_cost = sum(
                    stats["total_cost"] 
                    for stats in self.metrics.provider_stats.values()
                )
                
                # 记录到Redis
                metrics_data = {
                    "timestamp": time.time(),
                    "total_requests": self.metrics.total_requests,
                    "successful_requests": self.metrics.successful_requests,
                    "failed_requests": self.metrics.failed_requests,
                    "cache_hits": self.metrics.cache_hits,
                    "total_cost": self.metrics.total_cost,
                    "current_requests": self.current_requests
                }
                
                await self.redis_client.set_with_ttl(
                    "tts_service_metrics",
                    json.dumps(metrics_data),
                    3600  # 1小时
                )
                
            except Exception as e:
                logger.error(f"指标收集失败: {e}")
    
    async def _cache_cleanup_task(self):
        """缓存清理任务"""
        while True:
            try:
                await asyncio.sleep(3600)  # 每小时清理一次
                # 这里可以实现缓存清理逻辑
                logger.info("执行缓存清理")
            except Exception as e:
                logger.error(f"缓存清理失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            "service": "enhanced_tts",
            "current_requests": self.current_requests,
            "max_concurrent": self.max_concurrent,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "cache_hits": self.metrics.cache_hits,
                "cache_hit_rate": self.metrics.cache_hits / max(1, self.metrics.total_requests),
                "success_rate": self.metrics.successful_requests / max(1, self.metrics.total_requests),
                "total_cost": self.metrics.total_cost,
                "avg_cost_per_request": self.metrics.total_cost / max(1, self.metrics.successful_requests)
            },
            "provider_stats": self.metrics.provider_stats,
            "circuit_breaker": self.circuit_breaker,
            "endpoints": [
                {
                    "provider": ep.provider.value,
                    "name": ep.name,
                    "health_status": ep.health_status,
                    "current_load": ep.current_load,
                    "max_concurrent": ep.max_concurrent,
                    "success_rate": ep.success_rate,
                    "avg_response_time": ep.avg_response_time,
                    "cost_per_1k_tokens": ep.cost_per_1k_tokens
                }
                for ep in self.api_config.tts_endpoints
            ] if self.api_config else []
        }
    
    async def close(self):
        """关闭服务"""
        if self.api_config:
            await self.api_config.close_session()
        if self.queue_manager:
            await self.queue_manager.stop()

# 创建FastAPI应用
app = FastAPI(title="Enhanced TTS Service", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务实例
enhanced_tts_service = EnhancedTTSService(max_concurrent=50)

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    await enhanced_tts_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    await enhanced_tts_service.close()

@app.post("/tts/synthesize", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """语音合成接口"""
    return await enhanced_tts_service.process_request(request)

@app.get("/tts/stats")
async def get_stats():
    """获取统计信息"""
    return enhanced_tts_service.get_stats()

@app.get("/tts/endpoints")
async def get_endpoints():
    """获取端点信息"""
    if not enhanced_tts_service.api_config:
        return {"endpoints": []}
    
    return {
        "endpoints": [
            {
                "provider": ep.provider.value,
                "name": ep.name,
                "base_url": ep.base_url,
                "health_status": ep.health_status,
                "current_load": ep.current_load,
                "max_concurrent": ep.max_concurrent,
                "max_qps": ep.max_qps,
                "weight": ep.weight,
                "success_rate": ep.success_rate,
                "avg_response_time": ep.avg_response_time,
                "cost_per_1k_tokens": ep.cost_per_1k_tokens
            }
            for ep in enhanced_tts_service.api_config.tts_endpoints
        ]
    }

@app.get("/tts/voices")
async def get_available_voices():
    """获取可用声音列表"""
    return {
        "local_voices": enhanced_tts_service.local_engines["edge_tts"]["voices"],
        "remote_voices": {
            "azure": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural", "en-US-AriaNeural"],
            "google": ["zh-CN-Wavenet-A", "zh-CN-Wavenet-B", "en-US-Wavenet-A"],
            "aws": ["Zhiyu", "Hiujin", "Joanna", "Matthew"],
            "baidu": ["度小宇", "度小美", "度逍遥"],
            "xunfei": ["xiaoyan", "aisjiuxu", "aisxping"]
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "enhanced_tts",
        "timestamp": time.time(),
        "current_requests": enhanced_tts_service.current_requests,
        "circuit_breaker_state": enhanced_tts_service.circuit_breaker["state"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003, workers=1)