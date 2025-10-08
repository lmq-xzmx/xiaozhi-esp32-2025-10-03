"""
远程API集成配置
支持多厂商LLM和TTS服务的智能调度和负载均衡
"""

import os
import asyncio
import aiohttp
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class APIProvider(Enum):
    """API服务提供商"""
    # LLM提供商
    OPENAI = "openai"
    CLAUDE = "claude"
    QWEN = "qwen"
    BAICHUAN = "baichuan"
    ZHIPU = "zhipu"
    MOONSHOT = "moonshot"
    DEEPSEEK = "deepseek"
    
    # TTS提供商
    EDGE_TTS = "edge_tts"
    AZURE_TTS = "azure_tts"
    XUNFEI_TTS = "xunfei_tts"
    BAIDU_TTS = "baidu_tts"
    TENCENT_TTS = "tencent_tts"
    ALIBABA_TTS = "alibaba_tts"

@dataclass
class APIEndpoint:
    """API端点配置"""
    provider: APIProvider
    name: str
    base_url: str
    api_key: str
    model: str
    max_concurrent: int = 50
    max_qps: int = 100
    timeout: float = 30.0
    weight: float = 1.0
    cost_per_1k_tokens: float = 0.0
    
    # 运行时状态
    current_load: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_response_time: float = 0.0
    last_error_time: float = 0.0
    health_status: bool = True
    
    # 性能指标
    success_rate: float = 1.0
    cost_efficiency: float = 1.0  # 性价比评分
    latency_score: float = 1.0    # 延迟评分

class RemoteAPIConfig:
    """远程API配置管理"""
    
    def __init__(self):
        self.llm_endpoints = self._init_llm_endpoints()
        self.tts_endpoints = self._init_tts_endpoints()
        self.session = None
        
    def _init_llm_endpoints(self) -> List[APIEndpoint]:
        """初始化LLM API端点"""
        endpoints = []
        
        # OpenAI GPT系列
        if os.getenv("OPENAI_API_KEY"):
            endpoints.extend([
                APIEndpoint(
                    provider=APIProvider.OPENAI,
                    name="gpt-3.5-turbo",
                    base_url="https://api.openai.com/v1",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model="gpt-3.5-turbo",
                    max_concurrent=100,
                    max_qps=200,
                    timeout=30.0,
                    weight=0.8,
                    cost_per_1k_tokens=0.002
                ),
                APIEndpoint(
                    provider=APIProvider.OPENAI,
                    name="gpt-4-turbo",
                    base_url="https://api.openai.com/v1",
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model="gpt-4-turbo-preview",
                    max_concurrent=50,
                    max_qps=100,
                    timeout=45.0,
                    weight=0.6,
                    cost_per_1k_tokens=0.01
                )
            ])
        
        # 通义千问
        if os.getenv("QWEN_API_KEY"):
            endpoints.extend([
                APIEndpoint(
                    provider=APIProvider.QWEN,
                    name="qwen-turbo",
                    base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                    api_key=os.getenv("QWEN_API_KEY"),
                    model="qwen-turbo",
                    max_concurrent=200,
                    max_qps=300,
                    timeout=25.0,
                    weight=1.0,
                    cost_per_1k_tokens=0.0008
                ),
                APIEndpoint(
                    provider=APIProvider.QWEN,
                    name="qwen-plus",
                    base_url="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation",
                    api_key=os.getenv("QWEN_API_KEY"),
                    model="qwen-plus",
                    max_concurrent=150,
                    max_qps=200,
                    timeout=30.0,
                    weight=0.9,
                    cost_per_1k_tokens=0.004
                )
            ])
        
        # 百川智能
        if os.getenv("BAICHUAN_API_KEY"):
            endpoints.append(
                APIEndpoint(
                    provider=APIProvider.BAICHUAN,
                    name="baichuan2-turbo",
                    base_url="https://api.baichuan-ai.com/v1/chat/completions",
                    api_key=os.getenv("BAICHUAN_API_KEY"),
                    model="Baichuan2-Turbo",
                    max_concurrent=100,
                    max_qps=150,
                    timeout=30.0,
                    weight=0.7,
                    cost_per_1k_tokens=0.012
                )
            )
        
        # 智谱AI
        if os.getenv("ZHIPU_API_KEY"):
            endpoints.append(
                APIEndpoint(
                    provider=APIProvider.ZHIPU,
                    name="glm-4",
                    base_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
                    api_key=os.getenv("ZHIPU_API_KEY"),
                    model="glm-4",
                    max_concurrent=80,
                    max_qps=120,
                    timeout=35.0,
                    weight=0.8,
                    cost_per_1k_tokens=0.1
                )
            )
        
        # 月之暗面
        if os.getenv("MOONSHOT_API_KEY"):
            endpoints.append(
                APIEndpoint(
                    provider=APIProvider.MOONSHOT,
                    name="moonshot-v1-8k",
                    base_url="https://api.moonshot.cn/v1",
                    api_key=os.getenv("MOONSHOT_API_KEY"),
                    model="moonshot-v1-8k",
                    max_concurrent=60,
                    max_qps=100,
                    timeout=30.0,
                    weight=0.7,
                    cost_per_1k_tokens=0.012
                )
            )
        
        # DeepSeek
        if os.getenv("DEEPSEEK_API_KEY"):
            endpoints.append(
                APIEndpoint(
                    provider=APIProvider.DEEPSEEK,
                    name="deepseek-chat",
                    base_url="https://api.deepseek.com/v1",
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    model="deepseek-chat",
                    max_concurrent=100,
                    max_qps=150,
                    timeout=25.0,
                    weight=0.9,
                    cost_per_1k_tokens=0.0014
                )
            )
        
        return endpoints
    
    def _init_tts_endpoints(self) -> List[APIEndpoint]:
        """初始化TTS API端点"""
        endpoints = []
        
        # Edge TTS (免费)
        endpoints.append(
            APIEndpoint(
                provider=APIProvider.EDGE_TTS,
                name="edge-tts",
                base_url="",  # 本地调用
                api_key="",
                model="zh-CN-XiaoxiaoNeural",
                max_concurrent=200,
                max_qps=500,
                timeout=15.0,
                weight=1.0,
                cost_per_1k_tokens=0.0
            )
        )
        
        # Azure TTS
        if os.getenv("AZURE_TTS_KEY"):
            endpoints.append(
                APIEndpoint(
                    provider=APIProvider.AZURE_TTS,
                    name="azure-tts",
                    base_url=f"https://{os.getenv('AZURE_TTS_REGION', 'eastus')}.tts.speech.microsoft.com",
                    api_key=os.getenv("AZURE_TTS_KEY"),
                    model="zh-CN-XiaoxiaoNeural",
                    max_concurrent=100,
                    max_qps=200,
                    timeout=20.0,
                    weight=0.8,
                    cost_per_1k_tokens=0.016
                )
            )
        
        # 讯飞TTS
        if os.getenv("XUNFEI_APP_ID"):
            endpoints.append(
                APIEndpoint(
                    provider=APIProvider.XUNFEI_TTS,
                    name="xunfei-tts",
                    base_url="https://tts-api.xfyun.cn/v2/tts",
                    api_key=os.getenv("XUNFEI_API_KEY"),
                    model="xiaoyan",
                    max_concurrent=80,
                    max_qps=150,
                    timeout=20.0,
                    weight=0.7,
                    cost_per_1k_tokens=0.033
                )
            )
        
        # 百度TTS
        if os.getenv("BAIDU_TTS_API_KEY"):
            endpoints.append(
                APIEndpoint(
                    provider=APIProvider.BAIDU_TTS,
                    name="baidu-tts",
                    base_url="https://tsn.baidu.com/text2audio",
                    api_key=os.getenv("BAIDU_TTS_API_KEY"),
                    model="0",  # 度小美
                    max_concurrent=60,
                    max_qps=100,
                    timeout=25.0,
                    weight=0.6,
                    cost_per_1k_tokens=0.04
                )
            )
        
        # 腾讯TTS
        if os.getenv("TENCENT_SECRET_ID"):
            endpoints.append(
                APIEndpoint(
                    provider=APIProvider.TENCENT_TTS,
                    name="tencent-tts",
                    base_url="https://tts.tencentcloudapi.com",
                    api_key=os.getenv("TENCENT_SECRET_KEY"),
                    model="101001",  # 智逍遥
                    max_concurrent=50,
                    max_qps=80,
                    timeout=30.0,
                    weight=0.5,
                    cost_per_1k_tokens=0.05
                )
            )
        
        # 阿里云TTS
        if os.getenv("ALIBABA_ACCESS_KEY_ID"):
            endpoints.append(
                APIEndpoint(
                    provider=APIProvider.ALIBABA_TTS,
                    name="alibaba-tts",
                    base_url="https://nls-gateway.cn-shanghai.aliyuncs.com",
                    api_key=os.getenv("ALIBABA_ACCESS_KEY_SECRET"),
                    model="xiaoyun",
                    max_concurrent=70,
                    max_qps=120,
                    timeout=25.0,
                    weight=0.6,
                    cost_per_1k_tokens=0.032
                )
            )
        
        return endpoints
    
    async def init_session(self):
        """初始化HTTP会话"""
        connector = aiohttp.TCPConnector(
            limit=500,  # 总连接池大小
            limit_per_host=100,  # 每个主机的连接数
            ttl_dns_cache=300,  # DNS缓存时间
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=60,
            connect=10,
            sock_read=30
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                "User-Agent": "XiaoZhi-Server/1.0",
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
        )
    
    async def close_session(self):
        """关闭HTTP会话"""
        if self.session:
            await self.session.close()
    
    def get_llm_endpoints(self, healthy_only: bool = True) -> List[APIEndpoint]:
        """获取LLM端点列表"""
        endpoints = self.llm_endpoints
        if healthy_only:
            endpoints = [ep for ep in endpoints if ep.health_status]
        return sorted(endpoints, key=lambda x: x.weight * x.success_rate * x.cost_efficiency, reverse=True)
    
    def get_tts_endpoints(self, healthy_only: bool = True) -> List[APIEndpoint]:
        """获取TTS端点列表"""
        endpoints = self.tts_endpoints
        if healthy_only:
            endpoints = [ep for ep in endpoints if ep.health_status]
        return sorted(endpoints, key=lambda x: x.weight * x.success_rate, reverse=True)
    
    def update_endpoint_stats(self, endpoint: APIEndpoint, response_time: float, success: bool, cost: float = 0.0):
        """更新端点统计信息"""
        endpoint.total_requests += 1
        
        if success:
            # 更新平均响应时间
            if endpoint.avg_response_time == 0:
                endpoint.avg_response_time = response_time
            else:
                endpoint.avg_response_time = (endpoint.avg_response_time * 0.9) + (response_time * 0.1)
            
            # 更新成功率
            endpoint.success_rate = min(1.0, endpoint.success_rate * 0.99 + 0.01)
            
            # 更新延迟评分 (响应时间越短评分越高)
            endpoint.latency_score = max(0.1, 1.0 - (response_time / 60.0))
            
            # 更新性价比评分
            if cost > 0:
                endpoint.cost_efficiency = max(0.1, 1.0 / (cost * response_time))
        else:
            endpoint.total_errors += 1
            endpoint.last_error_time = time.time()
            
            # 降低成功率
            endpoint.success_rate = max(0.1, endpoint.success_rate * 0.95)
            
            # 检查是否需要标记为不健康
            error_rate = endpoint.total_errors / endpoint.total_requests
            if error_rate > 0.1 or (time.time() - endpoint.last_error_time < 60 and endpoint.total_errors > 3):
                endpoint.health_status = False
    
    async def health_check(self):
        """健康检查"""
        for endpoint in self.llm_endpoints + self.tts_endpoints:
            # 如果端点被标记为不健康，且距离上次错误超过5分钟，尝试恢复
            if not endpoint.health_status and time.time() - endpoint.last_error_time > 300:
                endpoint.health_status = True
                endpoint.success_rate = 0.5  # 给一个中等的初始成功率

# 全局配置实例
remote_api_config = RemoteAPIConfig()

async def get_remote_api_config() -> RemoteAPIConfig:
    """获取远程API配置实例"""
    if not remote_api_config.session:
        await remote_api_config.init_session()
    return remote_api_config