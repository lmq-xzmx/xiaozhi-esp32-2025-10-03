#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import re
import logging
import requests
from typing import Optional, Dict, Any, List
import base64
import io

logger = logging.getLogger(__name__)

def get_local_ip():
    """获取本地IP地址"""
    try:
        # 创建一个UDP socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到一个远程地址（不会实际发送数据）
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def validate_mcp_endpoint(endpoint: str) -> bool:
    """验证MCP端点格式"""
    if not endpoint:
        return False
    
    # 简单的URL格式验证
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return url_pattern.match(endpoint) is not None

def check_model_key(key: str) -> bool:
    """检查模型密钥是否有效"""
    if not key or len(key.strip()) == 0:
        return False
    return True

def remove_punctuation_and_length(text: str) -> str:
    """移除标点符号并返回处理后的文本"""
    if not text:
        return ""
    
    # 移除常见的标点符号
    punctuation = '！？。，、；：""''（）【】《》〈〉「」『』〔〕〖〗〘〙〚〛.,!?;:()"\'[]{}/<>\\|`~@#$%^&*+=_-'
    for p in punctuation:
        text = text.replace(p, '')
    
    return text.strip()

def get_ip_info() -> Dict[str, Any]:
    """获取IP信息"""
    try:
        local_ip = get_local_ip()
        return {
            "local_ip": local_ip,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"获取IP信息失败: {e}")
        return {
            "local_ip": "127.0.0.1",
            "status": "error",
            "error": str(e)
        }

def get_vision_url(image_data: bytes) -> str:
    """将图像数据转换为base64 URL"""
    try:
        if not image_data:
            return ""
        
        # 将字节数据转换为base64
        base64_data = base64.b64encode(image_data).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_data}"
    except Exception as e:
        logger.error(f"转换图像数据失败: {e}")
        return ""

def sanitize_tool_name(name: str) -> str:
    """清理工具名称，移除特殊字符"""
    if not name:
        return ""
    
    # 只保留字母、数字、下划线和连字符
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    return sanitized

def check_vad_update() -> bool:
    """检查VAD更新"""
    # 简单的检查逻辑
    return True

def check_asr_update() -> bool:
    """检查ASR更新"""
    # 简单的检查逻辑
    return True

def audio_bytes_to_data_stream(audio_bytes: bytes) -> io.BytesIO:
    """将音频字节转换为数据流"""
    return io.BytesIO(audio_bytes)

def audio_to_data_stream(audio_data: Any) -> io.BytesIO:
    """将音频数据转换为数据流"""
    if isinstance(audio_data, bytes):
        return io.BytesIO(audio_data)
    elif hasattr(audio_data, 'read'):
        return audio_data
    else:
        return io.BytesIO(b'')

# 其他可能需要的工具函数
def safe_get(data: Dict, key: str, default: Any = None) -> Any:
    """安全获取字典值"""
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default

def format_error_message(error: Exception) -> str:
    """格式化错误消息"""
    return f"{type(error).__name__}: {str(error)}"