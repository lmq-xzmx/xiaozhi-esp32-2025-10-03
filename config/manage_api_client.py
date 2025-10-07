#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import asyncio
import aiohttp
from typing import Dict, Any, Optional
from datetime import datetime

class ManageAPIClient:
    """
    管理API客户端
    """
    
    def __init__(self):
        self.base_url = os.getenv("MANAGE_API_URL", "")
        self.api_key = os.getenv("MANAGE_API_KEY", "")
        self.timeout = int(os.getenv("MANAGE_API_TIMEOUT", "30"))
        self.enabled = os.getenv("ENABLE_MANAGE_API", "false").lower() == "true"
        
    async def report(self, data: Dict[str, Any]) -> bool:
        """
        发送报告数据
        """
        if not self.enabled or not self.base_url:
            return True  # 如果未启用，直接返回成功
            
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                }
                
                # 添加时间戳
                data["timestamp"] = datetime.now().isoformat()
                data["server_id"] = os.getenv("SERVER_ID", "xiaozhi-server")
                
                async with session.post(
                    f"{self.base_url}/api/reports",
                    json=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return True
                    else:
                        print(f"Report failed with status {response.status}: {await response.text()}")
                        return False
                        
        except Exception as e:
            print(f"Report error: {e}")
            return False
    
    async def get_config(self) -> Optional[Dict[str, Any]]:
        """
        获取远程配置
        """
        if not self.enabled or not self.base_url:
            return None
            
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                }
                
                async with session.get(
                    f"{self.base_url}/api/config",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        print(f"Get config failed with status {response.status}")
                        return None
                        
        except Exception as e:
            print(f"Get config error: {e}")
            return None

# 全局实例
_client = ManageAPIClient()

def report(data: Dict[str, Any]) -> bool:
    """
    同步报告函数
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_client.report(data))
    except RuntimeError:
        # 如果没有事件循环，创建一个新的
        return asyncio.run(_client.report(data))
    except Exception as e:
        print(f"Report sync error: {e}")
        return False

async def async_report(data: Dict[str, Any]) -> bool:
    """
    异步报告函数
    """
    return await _client.report(data)

def get_config() -> Optional[Dict[str, Any]]:
    """
    同步获取配置函数
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_client.get_config())
    except RuntimeError:
        # 如果没有事件循环，创建一个新的
        return asyncio.run(_client.get_config())
    except Exception as e:
        print(f"Get config sync error: {e}")
        return None

async def async_get_config() -> Optional[Dict[str, Any]]:
    """
    异步获取配置函数
    """
    return await _client.get_config()

# 报告类型常量
REPORT_TYPES = {
    "ASR_REQUEST": "asr_request",
    "ASR_SUCCESS": "asr_success", 
    "ASR_ERROR": "asr_error",
    "TTS_REQUEST": "tts_request",
    "TTS_SUCCESS": "tts_success",
    "TTS_ERROR": "tts_error",
    "LLM_REQUEST": "llm_request",
    "LLM_SUCCESS": "llm_success",
    "LLM_ERROR": "llm_error",
    "VAD_REQUEST": "vad_request",
    "VAD_SUCCESS": "vad_success",
    "VAD_ERROR": "vad_error",
    "SYSTEM_STATUS": "system_status",
    "PERFORMANCE": "performance",
    "ERROR": "error",
}

def create_report(report_type: str, **kwargs) -> Dict[str, Any]:
    """
    创建标准报告格式
    """
    return {
        "type": report_type,
        "data": kwargs,
        "timestamp": datetime.now().isoformat(),
        "server_id": os.getenv("SERVER_ID", "xiaozhi-server"),
    }