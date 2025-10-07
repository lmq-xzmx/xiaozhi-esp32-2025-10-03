#!/usr/bin/env python3
"""
LLM (Large Language Model) 微服务
支持负载均衡、缓存策略、连接池和智能路由
"""

import asyncio
import logging
import time
import hashlib
import json
import random
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    CLAUDE = "claude"
    QWEN = "qwen"
    BAICHUAN = "baichuan"
    LOCAL = "local"

@dataclass
class LLMEndpoint:
    """LLM端点配置"""
    provider: LLMProvider
    url: str
    api_key: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: float = 30.0
    weight: int = 1  # 负载均衡权重
    max_concurrent: int = 10
    current_load: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time: float = 0.0
    last_error_time: float = 0.0
    health_status: bool = True

class LLMRequest(BaseModel):
    """LLM请求模型"""
    session_id: str
    messages: List[Dict[str, str]]
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    priority: int = 2  # 1=高优先级, 2=中优先级, 3=低优先级
    cache_enabled: bool = True
    timeout: Optional[float] = None

class LLMResponse(BaseModel):
    """LLM响应模型"""
    session_id: str
    content: str
    model: str
    provider: str
    tokens_used: int
    response_time: float
    cached: bool = False
    timestamp: float = 0.0

class LLMLoadBalancer:
    """LLM负载均衡器"""
    
    def __init__(self):
        self.endpoints: List[LLMEndpoint] = []
        self.circuit_breaker_threshold = 5  # 连续错误阈值
        self.circuit_breaker_timeout = 300  # 熔断器超时时间（秒）
    
    def add_endpoint(self, endpoint: LLMEndpoint):
        """添加LLM端点"""
        self.endpoints.append(endpoint)
        logger.info(f"添加LLM端点: {endpoint.provider.value} - {endpoint.model}")
    
    def get_healthy_endpoints(self) -> List[LLMEndpoint]:
        """获取健康的端点"""
        current_time = time.time()
        healthy = []
        
        for endpoint in self.endpoints:
            # 检查熔断器状态
            if not endpoint.health_status:
                if current_time - endpoint.last_error_time > self.circuit_breaker_timeout:
                    endpoint.health_status = True
                    logger.info(f"端点恢复: {endpoint.provider.value}")
                else:
                    continue
            
            # 检查并发限制
            if endpoint.current_load < endpoint.max_concurrent:
                healthy.append(endpoint)
        
        return healthy
    
    def select_endpoint(self, request: LLMRequest) -> Optional[LLMEndpoint]:
        """选择最佳端点（加权轮询 + 负载考虑）"""
        healthy_endpoints = self.get_healthy_endpoints()
        
        if not healthy_endpoints:
            return None
        
        # 计算权重分数（权重 / (当前负载 + 1) / 平均响应时间）
        scored_endpoints = []
        for endpoint in healthy_endpoints:
            load_factor = endpoint.current_load + 1
            time_factor = max(endpoint.avg_response_time, 0.1)
            score = endpoint.weight / load_factor / time_factor
            scored_endpoints.append((score, endpoint))
        
        # 按分数排序，选择最高分的
        scored_endpoints.sort(key=lambda x: x[0], reverse=True)
        return scored_endpoints[0][1]
    
    def update_endpoint_stats(self, endpoint: LLMEndpoint, response_time: float, success: bool):
        """更新端点统计信息"""
        endpoint.total_requests += 1
        
        if success:
            # 更新平均响应时间
            if endpoint.avg_response_time == 0:
                endpoint.avg_response_time = response_time
            else:
                endpoint.avg_response_time = (endpoint.avg_response_time * 0.9 + response_time * 0.1)
        else:
            endpoint.total_errors += 1
            endpoint.last_error_time = time.time()
            
            # 检查是否需要熔断
            error_rate = endpoint.total_errors / endpoint.total_requests
            if error_rate > 0.5 and endpoint.total_errors >= self.circuit_breaker_threshold:
                endpoint.health_status = False
                logger.warning(f"端点熔断: {endpoint.provider.value}, 错误率: {error_rate:.2f}")

class LLMCache:
    """LLM缓存管理器"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.cache_ttl = 86400  # 24小时 (从1小时优化到24小时)
        self.max_cache_size = 1000000  # 1MB
    
    def generate_cache_key(self, messages: List[Dict], model: str, temperature: float) -> str:
        """生成缓存键"""
        content = json.dumps({
            "messages": messages,
            "model": model,
            "temperature": temperature
        }, sort_keys=True)
        return f"llm_cache:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """获取缓存响应"""
        try:
            cached = await self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"获取缓存失败: {e}")
        return None
    
    async def cache_response(self, cache_key: str, response: Dict):
        """缓存响应"""
        try:
            # 检查响应大小
            response_str = json.dumps(response)
            if len(response_str) <= self.max_cache_size:
                await self.redis_client.setex(cache_key, self.cache_ttl, response_str)
        except Exception as e:
            logger.error(f"缓存响应失败: {e}")

class LLMService:
    """LLM微服务主类"""
    
    def __init__(self):
        self.load_balancer = LLMLoadBalancer()
        self.cache = None
        self.session = None
        self.redis_client = None
        
        # 性能统计
        self.total_requests = 0
        self.cache_hits = 0
        self.total_response_time = 0.0
        
        # 初始化端点
        self.init_endpoints()
    
    def init_endpoints(self):
        """初始化LLM端点"""
        # 示例端点配置（实际使用时从配置文件读取）
        endpoints = [
            LLMEndpoint(
                provider=LLMProvider.QWEN,
                url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                api_key="your-qwen-api-key",
                model="qwen-turbo",
                max_tokens=2048,
                weight=3,
                max_concurrent=15
            ),
            LLMEndpoint(
                provider=LLMProvider.BAICHUAN,
                url="https://api.baichuan-ai.com/v1/chat/completions",
                api_key="your-baichuan-api-key",
                model="Baichuan2-Turbo",
                max_tokens=2048,
                weight=2,
                max_concurrent=10
            ),
            LLMEndpoint(
                provider=LLMProvider.LOCAL,
                url="http://localhost:11434/api/generate",
                api_key="",
                model="qwen2:7b",
                max_tokens=2048,
                weight=1,
                max_concurrent=5
            )
        ]
        
        for endpoint in endpoints:
            self.load_balancer.add_endpoint(endpoint)
    
    async def init_redis(self, redis_url: str = "redis://localhost:6379"):
        """初始化Redis连接"""
        try:
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            self.cache = LLMCache(self.redis_client)
            logger.info("Redis连接成功")
        except Exception as e:
            logger.error(f"Redis连接失败: {e}")
    
    async def init_session(self):
        """初始化HTTP会话，优化连接池配置"""
        connector = aiohttp.TCPConnector(
            limit=200,  # 从100提升到200
            limit_per_host=50,  # 从20提升到50
            ttl_dns_cache=600,  # 从300提升到600秒
            use_dns_cache=True,
            keepalive_timeout=60,  # 增加keepalive超时
            enable_cleanup_closed=True,  # 启用清理关闭的连接
            force_close=False,  # 不强制关闭连接
            limit_per_host_per_scheme=30  # 每个scheme的连接限制
        )
        
        # 设置超时配置
        timeout = aiohttp.ClientTimeout(
            total=60,  # 总超时时间
            connect=10,  # 连接超时
            sock_read=30  # 读取超时
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'Xiaozhi-LLM-Service/1.0',
                'Connection': 'keep-alive'
            }
        )
    
    async def close_session(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
    
    async def call_qwen_api(self, endpoint: LLMEndpoint, request: LLMRequest) -> Dict:
        """调用通义千问API"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint.model,
            "input": {
                "messages": request.messages
            },
            "parameters": {
                "max_tokens": request.max_tokens or endpoint.max_tokens,
                "temperature": request.temperature or endpoint.temperature
            }
        }
        
        async with self.session.post(
            endpoint.url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=request.timeout or endpoint.timeout)
        ) as response:
            result = await response.json()
            
            if response.status == 200:
                content = result["output"]["text"]
                tokens = result["usage"]["total_tokens"]
                return {"content": content, "tokens": tokens}
            else:
                raise Exception(f"API调用失败: {result}")
    
    async def call_baichuan_api(self, endpoint: LLMEndpoint, request: LLMRequest) -> Dict:
        """调用百川API"""
        headers = {
            "Authorization": f"Bearer {endpoint.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": endpoint.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens or endpoint.max_tokens,
            "temperature": request.temperature or endpoint.temperature
        }
        
        async with self.session.post(
            endpoint.url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=request.timeout or endpoint.timeout)
        ) as response:
            result = await response.json()
            
            if response.status == 200:
                content = result["choices"][0]["message"]["content"]
                tokens = result["usage"]["total_tokens"]
                return {"content": content, "tokens": tokens}
            else:
                raise Exception(f"API调用失败: {result}")
    
    async def call_local_api(self, endpoint: LLMEndpoint, request: LLMRequest) -> Dict:
        """调用本地Ollama API"""
        # 转换消息格式
        prompt = ""
        for msg in request.messages:
            if msg["role"] == "user":
                prompt += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n"
        prompt += "Assistant: "
        
        payload = {
            "model": endpoint.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": request.max_tokens or endpoint.max_tokens,
                "temperature": request.temperature or endpoint.temperature
            }
        }
        
        async with self.session.post(
            endpoint.url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=request.timeout or endpoint.timeout)
        ) as response:
            result = await response.json()
            
            if response.status == 200:
                content = result["response"]
                tokens = len(content.split())  # 简化的token计算
                return {"content": content, "tokens": tokens}
            else:
                raise Exception(f"API调用失败: {result}")
    
    async def call_llm_api(self, endpoint: LLMEndpoint, request: LLMRequest) -> Dict:
        """调用LLM API"""
        if endpoint.provider == LLMProvider.QWEN:
            return await self.call_qwen_api(endpoint, request)
        elif endpoint.provider == LLMProvider.BAICHUAN:
            return await self.call_baichuan_api(endpoint, request)
        elif endpoint.provider == LLMProvider.LOCAL:
            return await self.call_local_api(endpoint, request)
        else:
            raise Exception(f"不支持的LLM提供商: {endpoint.provider}")
    
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """处理LLM请求"""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # 检查缓存
            if request.cache_enabled and self.cache:
                cache_key = self.cache.generate_cache_key(
                    request.messages,
                    request.model or "default",
                    request.temperature or 0.7
                )
                
                cached_response = await self.cache.get_cached_response(cache_key)
                if cached_response:
                    self.cache_hits += 1
                    return LLMResponse(
                        session_id=request.session_id,
                        content=cached_response["content"],
                        model=cached_response["model"],
                        provider=cached_response["provider"],
                        tokens_used=cached_response["tokens_used"],
                        response_time=cached_response["response_time"],
                        cached=True,
                        timestamp=time.time()
                    )
            
            # 选择端点
            endpoint = self.load_balancer.select_endpoint(request)
            if not endpoint:
                raise Exception("没有可用的LLM端点")
            
            # 增加负载计数
            endpoint.current_load += 1
            
            try:
                # 调用API
                result = await self.call_llm_api(endpoint, request)
                response_time = time.time() - start_time
                
                # 更新端点统计
                self.load_balancer.update_endpoint_stats(endpoint, response_time, True)
                
                # 创建响应
                llm_response = LLMResponse(
                    session_id=request.session_id,
                    content=result["content"],
                    model=endpoint.model,
                    provider=endpoint.provider.value,
                    tokens_used=result["tokens"],
                    response_time=response_time,
                    timestamp=time.time()
                )
                
                # 缓存响应
                if request.cache_enabled and self.cache:
                    await self.cache.cache_response(cache_key, {
                        "content": llm_response.content,
                        "model": llm_response.model,
                        "provider": llm_response.provider,
                        "tokens_used": llm_response.tokens_used,
                        "response_time": llm_response.response_time
                    })
                
                # 更新统计
                self.total_response_time += response_time
                
                return llm_response
                
            finally:
                # 减少负载计数
                endpoint.current_load -= 1
        
        except Exception as e:
            # 更新端点统计（如果有选中的端点）
            if 'endpoint' in locals():
                self.load_balancer.update_endpoint_stats(endpoint, 0, False)
            
            logger.error(f"LLM请求处理失败: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """获取服务统计"""
        avg_response_time = self.total_response_time / max(self.total_requests, 1)
        cache_hit_rate = self.cache_hits / max(self.total_requests, 1)
        
        endpoint_stats = []
        for endpoint in self.load_balancer.endpoints:
            endpoint_stats.append({
                "provider": endpoint.provider.value,
                "model": endpoint.model,
                "health_status": endpoint.health_status,
                "current_load": endpoint.current_load,
                "max_concurrent": endpoint.max_concurrent,
                "total_requests": endpoint.total_requests,
                "total_errors": endpoint.total_errors,
                "error_rate": endpoint.total_errors / max(endpoint.total_requests, 1),
                "avg_response_time": endpoint.avg_response_time
            })
        
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_response_time": avg_response_time,
            "requests_per_second": self.total_requests / max(self.total_response_time, 1),
            "endpoints": endpoint_stats
        }

# FastAPI应用
app = FastAPI(title="LLM Service", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局LLM服务实例
llm_service = LLMService()

@app.on_event("startup")
async def startup_event():
    """启动事件"""
    await llm_service.init_redis()
    await llm_service.init_session()

@app.on_event("shutdown")
async def shutdown_event():
    """关闭事件"""
    await llm_service.close_session()

@app.post("/llm/chat", response_model=LLMResponse)
async def chat_completion(request: LLMRequest, background_tasks: BackgroundTasks):
    """LLM对话完成API"""
    try:
        response = await llm_service.process_request(request)
        return response
    except Exception as e:
        logger.error(f"LLM对话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm/stats")
async def get_stats():
    """获取服务统计信息"""
    return llm_service.get_stats()

@app.get("/health")
async def health_check():
    """健康检查"""
    healthy_endpoints = len(llm_service.load_balancer.get_healthy_endpoints())
    total_endpoints = len(llm_service.load_balancer.endpoints)
    
    return {
        "status": "healthy" if healthy_endpoints > 0 else "unhealthy",
        "service": "llm",
        "healthy_endpoints": healthy_endpoints,
        "total_endpoints": total_endpoints
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002, workers=1)