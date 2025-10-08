"""
增强版LLM服务
集成远程API调用、智能路由、故障转移、缓存优化等功能
"""

import asyncio
import aiohttp
import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config.remote_api_config import get_remote_api_config, APIProvider, APIEndpoint
from config.redis_config import get_redis_client, OptimizedRedisClient
from core.intelligent_router import get_intelligent_router, RoutingContext, RequestPriority
from core.queue_manager import get_queue_manager, QueueRequest, Priority

logger = logging.getLogger(__name__)

class LLMRequest(BaseModel):
    """LLM请求模型"""
    session_id: str
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    stream: bool = False
    priority: int = 2  # 1=高优先级, 2=中优先级, 3=低优先级
    cache_enabled: bool = True
    timeout: Optional[float] = 30.0
    cost_budget: Optional[float] = None
    quality_requirement: float = 0.8

class LLMResponse(BaseModel):
    """LLM响应模型"""
    session_id: str
    content: str
    model: str
    provider: str
    tokens_used: int
    response_time: float
    cached: bool = False
    cost: float = 0.0
    timestamp: float = 0.0

@dataclass
class LLMMetrics:
    """LLM服务指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cache_hits: int = 0
    total_response_time: float = 0.0
    total_cost: float = 0.0
    provider_stats: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.provider_stats is None:
            self.provider_stats = {}

class EnhancedLLMService:
    """增强版LLM服务"""
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.current_requests = 0
        self.metrics = LLMMetrics()
        
        # 组件初始化
        self.api_config = None
        self.router = None
        self.redis_client = None
        self.queue_manager = None
        
        # 缓存配置
        self.cache_ttl = 1800  # 30分钟
        self.semantic_cache_enabled = True
        
        # 故障转移配置
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # 熔断器配置
        self.circuit_breaker = {
            "failure_threshold": 5,
            "recovery_timeout": 60,
            "half_open_max_calls": 3,
            "state": "closed",  # closed, open, half_open
            "failure_count": 0,
            "last_failure_time": 0
        }
    
    async def initialize(self):
        """初始化服务"""
        try:
            # 初始化API配置
            self.api_config = await get_remote_api_config()
            logger.info(f"初始化了 {len(self.api_config.llm_endpoints)} 个LLM端点")
            
            # 初始化智能路由器
            self.router = await get_intelligent_router()
            logger.info("智能路由器初始化完成")
            
            # 初始化Redis客户端
            self.redis_client = get_redis_client()
            await self.redis_client.health_check()
            logger.info("Redis客户端初始化完成")
            
            # 初始化队列管理器
            self.queue_manager = get_queue_manager(
                service_name="enhanced_llm",
                max_queue_size=1000,
                batch_timeout=100,  # 100ms
                max_concurrent=self.max_concurrent,
                batch_size=20
            )
            await self.queue_manager.start()
            logger.info("队列管理器初始化完成")
            
            # 启动后台任务
            asyncio.create_task(self._health_check_task())
            asyncio.create_task(self._metrics_collection_task())
            asyncio.create_task(self._cache_cleanup_task())
            
            logger.info("增强版LLM服务初始化完成")
            
        except Exception as e:
            logger.error(f"LLM服务初始化失败: {e}")
            raise
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """处理LLM请求"""
        start_time = time.time()
        self.metrics.total_requests += 1
        
        try:
            # 检查熔断器状态
            if not self._check_circuit_breaker():
                raise HTTPException(status_code=503, detail="服务暂时不可用")
            
            # 生成缓存键
            cache_key = self._generate_cache_key(request)
            
            # 尝试从缓存获取
            if request.cache_enabled:
                cached_response = await self._get_cached_response(cache_key)
                if cached_response:
                    self.metrics.cache_hits += 1
                    cached_response.cached = True
                    cached_response.response_time = time.time() - start_time
                    return cached_response
            
            # 创建路由上下文
            context = RoutingContext(
                request_id=f"{request.session_id}_{int(time.time())}",
                priority=RequestPriority(request.priority),
                text_length=sum(len(msg.get("content", "")) for msg in request.messages),
                expected_response_length=request.max_tokens or 2048,
                session_id=request.session_id,
                timeout=request.timeout or 30.0,
                cost_budget=request.cost_budget or 0.0,
                quality_requirement=request.quality_requirement
            )
            
            # 选择最佳端点
            endpoint = await self.router.select_llm_endpoint(context)
            if not endpoint:
                raise HTTPException(status_code=503, detail="没有可用的LLM端点")
            
            # 更新负载
            await self.router.update_endpoint_load(endpoint, 1)
            self.current_requests += 1
            
            try:
                # 调用API
                response = await self._call_llm_api(endpoint, request, context)
                
                # 缓存响应
                if request.cache_enabled and response:
                    await self._cache_response(cache_key, response)
                
                # 更新指标
                self.metrics.successful_requests += 1
                self._update_provider_stats(endpoint.provider, time.time() - start_time, True, response.cost)
                
                # 更新端点统计
                self.api_config.update_endpoint_stats(
                    endpoint, 
                    time.time() - start_time, 
                    True, 
                    response.cost
                )
                
                # 重置熔断器
                self._reset_circuit_breaker()
                
                response.response_time = time.time() - start_time
                return response
                
            finally:
                # 释放负载
                await self.router.update_endpoint_load(endpoint, -1)
                self.current_requests -= 1
        
        except Exception as e:
            self.metrics.failed_requests += 1
            self._record_circuit_breaker_failure()
            
            # 尝试故障转移
            if self.max_retries > 0:
                return await self._retry_with_fallback(request, context, str(e))
            
            logger.error(f"LLM请求处理失败: {e}")
            raise HTTPException(status_code=500, detail=f"LLM请求处理失败: {str(e)}")
    
    async def _call_llm_api(self, endpoint: APIEndpoint, request: LLMRequest, context: RoutingContext) -> LLMResponse:
        """调用LLM API"""
        if endpoint.provider == APIProvider.OPENAI:
            return await self._call_openai_api(endpoint, request)
        elif endpoint.provider == APIProvider.QWEN:
            return await self._call_qwen_api(endpoint, request)
        elif endpoint.provider == APIProvider.BAICHUAN:
            return await self._call_baichuan_api(endpoint, request)
        elif endpoint.provider == APIProvider.ZHIPU:
            return await self._call_zhipu_api(endpoint, request)
        elif endpoint.provider == APIProvider.MOONSHOT:
            return await self._call_moonshot_api(endpoint, request)
        elif endpoint.provider == APIProvider.DEEPSEEK:
            return await self._call_deepseek_api(endpoint, request)
        else:
            raise ValueError(f"不支持的LLM提供商: {endpoint.provider}")
    
    async def _call_openai_api(self, endpoint: APIEndpoint, request: LLMRequest) -> LLMResponse:
        """调用OpenAI API"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream
        }
        
        async with self.api_config.session.post(
            f"{endpoint.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=request.timeout
        ) as response:
            if response.status != 200:
                raise Exception(f"OpenAI API错误: {response.status}")
            
            data = await response.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data["usage"]["total_tokens"]
            
            # 计算成本
            cost = tokens_used * endpoint.cost_per_1k_tokens / 1000
            
            return LLMResponse(
                session_id=request.session_id,
                content=content,
                model=endpoint.model,
                provider=endpoint.provider.value,
                tokens_used=tokens_used,
                response_time=0.0,  # 将在外层设置
                cost=cost,
                timestamp=time.time()
            )
    
    async def _call_qwen_api(self, endpoint: APIEndpoint, request: LLMRequest) -> LLMResponse:
        """调用通义千问API"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        # 转换消息格式
        input_text = ""
        for msg in request.messages:
            if msg["role"] == "user":
                input_text += f"用户: {msg['content']}\n"
            elif msg["role"] == "assistant":
                input_text += f"助手: {msg['content']}\n"
        
        payload = {
            "model": endpoint.model,
            "input": {
                "messages": request.messages
            },
            "parameters": {
                "max_tokens": request.max_tokens,
                "temperature": request.temperature
            }
        }
        
        async with self.api_config.session.post(
            endpoint.base_url,
            headers=headers,
            json=payload,
            timeout=request.timeout
        ) as response:
            if response.status != 200:
                raise Exception(f"通义千问API错误: {response.status}")
            
            data = await response.json()
            content = data["output"]["text"]
            tokens_used = data["usage"]["total_tokens"]
            
            # 计算成本
            cost = tokens_used * endpoint.cost_per_1k_tokens / 1000
            
            return LLMResponse(
                session_id=request.session_id,
                content=content,
                model=endpoint.model,
                provider=endpoint.provider.value,
                tokens_used=tokens_used,
                response_time=0.0,
                cost=cost,
                timestamp=time.time()
            )
    
    async def _call_baichuan_api(self, endpoint: APIEndpoint, request: LLMRequest) -> LLMResponse:
        """调用百川API"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        async with self.api_config.session.post(
            endpoint.base_url,
            headers=headers,
            json=payload,
            timeout=request.timeout
        ) as response:
            if response.status != 200:
                raise Exception(f"百川API错误: {response.status}")
            
            data = await response.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data["usage"]["total_tokens"]
            
            cost = tokens_used * endpoint.cost_per_1k_tokens / 1000
            
            return LLMResponse(
                session_id=request.session_id,
                content=content,
                model=endpoint.model,
                provider=endpoint.provider.value,
                tokens_used=tokens_used,
                response_time=0.0,
                cost=cost,
                timestamp=time.time()
            )
    
    async def _call_zhipu_api(self, endpoint: APIEndpoint, request: LLMRequest) -> LLMResponse:
        """调用智谱API"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        async with self.api_config.session.post(
            endpoint.base_url,
            headers=headers,
            json=payload,
            timeout=request.timeout
        ) as response:
            if response.status != 200:
                raise Exception(f"智谱API错误: {response.status}")
            
            data = await response.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data["usage"]["total_tokens"]
            
            cost = tokens_used * endpoint.cost_per_1k_tokens / 1000
            
            return LLMResponse(
                session_id=request.session_id,
                content=content,
                model=endpoint.model,
                provider=endpoint.provider.value,
                tokens_used=tokens_used,
                response_time=0.0,
                cost=cost,
                timestamp=time.time()
            )
    
    async def _call_moonshot_api(self, endpoint: APIEndpoint, request: LLMRequest) -> LLMResponse:
        """调用月之暗面API"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        async with self.api_config.session.post(
            f"{endpoint.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=request.timeout
        ) as response:
            if response.status != 200:
                raise Exception(f"月之暗面API错误: {response.status}")
            
            data = await response.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data["usage"]["total_tokens"]
            
            cost = tokens_used * endpoint.cost_per_1k_tokens / 1000
            
            return LLMResponse(
                session_id=request.session_id,
                content=content,
                model=endpoint.model,
                provider=endpoint.provider.value,
                tokens_used=tokens_used,
                response_time=0.0,
                cost=cost,
                timestamp=time.time()
            )
    
    async def _call_deepseek_api(self, endpoint: APIEndpoint, request: LLMRequest) -> LLMResponse:
        """调用DeepSeek API"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature
        }
        
        async with self.api_config.session.post(
            f"{endpoint.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=request.timeout
        ) as response:
            if response.status != 200:
                raise Exception(f"DeepSeek API错误: {response.status}")
            
            data = await response.json()
            content = data["choices"][0]["message"]["content"]
            tokens_used = data["usage"]["total_tokens"]
            
            cost = tokens_used * endpoint.cost_per_1k_tokens / 1000
            
            return LLMResponse(
                session_id=request.session_id,
                content=content,
                model=endpoint.model,
                provider=endpoint.provider.value,
                tokens_used=tokens_used,
                response_time=0.0,
                cost=cost,
                timestamp=time.time()
            )
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """生成缓存键"""
        # 创建消息内容的哈希
        messages_str = json.dumps(request.messages, sort_keys=True, ensure_ascii=False)
        content_hash = hashlib.md5(messages_str.encode()).hexdigest()
        
        # 包含关键参数
        key_parts = [
            content_hash,
            str(request.max_tokens or 2048),
            str(request.temperature or 0.7),
            request.model or "default"
        ]
        
        return f"llm_cache:{':'.join(key_parts)}"
    
    async def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """获取缓存的响应"""
        try:
            cached_data = await self.redis_client.get_with_fallback(cache_key)
            if cached_data:
                return LLMResponse(**json.loads(cached_data))
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")
        return None
    
    async def _cache_response(self, cache_key: str, response: LLMResponse):
        """缓存响应"""
        try:
            response_data = response.dict()
            await self.redis_client.set_with_ttl(
                cache_key, 
                json.dumps(response_data, ensure_ascii=False), 
                self.cache_ttl
            )
        except Exception as e:
            logger.warning(f"缓存响应失败: {e}")
    
    async def _retry_with_fallback(self, request: LLMRequest, context: RoutingContext, error: str) -> LLMResponse:
        """故障转移重试"""
        for retry in range(self.max_retries):
            try:
                await asyncio.sleep(self.retry_delay * (retry + 1))
                
                # 选择新的端点
                endpoint = await self.router.select_llm_endpoint(context)
                if not endpoint:
                    continue
                
                # 重试调用
                await self.router.update_endpoint_load(endpoint, 1)
                try:
                    response = await self._call_llm_api(endpoint, request, context)
                    logger.info(f"故障转移成功，使用端点: {endpoint.name}")
                    return response
                finally:
                    await self.router.update_endpoint_load(endpoint, -1)
                    
            except Exception as e:
                logger.warning(f"重试 {retry + 1} 失败: {e}")
                continue
        
        raise HTTPException(status_code=500, detail=f"所有重试都失败了，最后错误: {error}")
    
    def _check_circuit_breaker(self) -> bool:
        """检查熔断器状态"""
        now = time.time()
        
        if self.circuit_breaker["state"] == "open":
            if now - self.circuit_breaker["last_failure_time"] > self.circuit_breaker["recovery_timeout"]:
                self.circuit_breaker["state"] = "half_open"
                self.circuit_breaker["failure_count"] = 0
                return True
            return False
        
        return True
    
    def _record_circuit_breaker_failure(self):
        """记录熔断器失败"""
        self.circuit_breaker["failure_count"] += 1
        self.circuit_breaker["last_failure_time"] = time.time()
        
        if self.circuit_breaker["failure_count"] >= self.circuit_breaker["failure_threshold"]:
            self.circuit_breaker["state"] = "open"
            logger.warning("熔断器已打开")
    
    def _reset_circuit_breaker(self):
        """重置熔断器"""
        if self.circuit_breaker["state"] == "half_open":
            self.circuit_breaker["state"] = "closed"
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
                    "llm_service_metrics",
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
            "service": "enhanced_llm",
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
                    "model": ep.model,
                    "health_status": ep.health_status,
                    "current_load": ep.current_load,
                    "max_concurrent": ep.max_concurrent,
                    "success_rate": ep.success_rate,
                    "avg_response_time": ep.avg_response_time,
                    "cost_per_1k_tokens": ep.cost_per_1k_tokens
                }
                for ep in self.api_config.llm_endpoints
            ] if self.api_config else []
        }
    
    async def close(self):
        """关闭服务"""
        if self.api_config:
            await self.api_config.close_session()
        if self.queue_manager:
            await self.queue_manager.stop()

# 创建FastAPI应用
app = FastAPI(title="Enhanced LLM Service", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 服务实例
enhanced_llm_service = EnhancedLLMService(max_concurrent=100)

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    await enhanced_llm_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    await enhanced_llm_service.close()

@app.post("/llm/chat", response_model=LLMResponse)
async def chat_completion(request: LLMRequest, background_tasks: BackgroundTasks):
    """聊天完成接口"""
    return await enhanced_llm_service.process_request(request)

@app.get("/llm/stats")
async def get_stats():
    """获取统计信息"""
    return enhanced_llm_service.get_stats()

@app.get("/llm/endpoints")
async def get_endpoints():
    """获取端点信息"""
    if not enhanced_llm_service.api_config:
        return {"endpoints": []}
    
    return {
        "endpoints": [
            {
                "provider": ep.provider.value,
                "name": ep.name,
                "model": ep.model,
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
            for ep in enhanced_llm_service.api_config.llm_endpoints
        ]
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "enhanced_llm",
        "timestamp": time.time(),
        "current_requests": enhanced_llm_service.current_requests,
        "circuit_breaker_state": enhanced_llm_service.circuit_breaker["state"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=1)