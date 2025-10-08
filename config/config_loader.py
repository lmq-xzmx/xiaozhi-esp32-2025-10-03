#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import yaml
from typing import Dict, Any, Optional

def get_private_config_from_api() -> Dict[str, Any]:
    """
    从API获取私有配置
    """
    # 默认配置
    # 注意：TTS引擎配置现在通过 http://182.44.78.40:8002/#/model-config 统一管理
    config = {
        "api_keys": {
            "qwen": os.getenv("QWEN_API_KEY", ""),
            "baichuan": os.getenv("BAICHUAN_API_KEY", ""),
            "openai": os.getenv("OPENAI_API_KEY", ""),
        },
        "endpoints": {
            "qwen_base_url": os.getenv("QWEN_BASE_URL", ""),
            "baichuan_base_url": os.getenv("BAICHUAN_BASE_URL", ""),
            "openai_base_url": os.getenv("OPENAI_BASE_URL", ""),
        },
        "models": {
            "qwen_model": os.getenv("QWEN_MODEL", "qwen-turbo"),
            "baichuan_model": os.getenv("BAICHUAN_MODEL", "Baichuan2-Turbo"),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        },

        "security": {
            "jwt_secret": os.getenv("JWT_SECRET", "xiaozhi-secret-key"),
            "jwt_algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
            "jwt_expire_hours": int(os.getenv("JWT_EXPIRE_HOURS", "24")),
        },
        "features": {
            "enable_vad": os.getenv("ENABLE_VAD", "true").lower() == "true",
            "enable_asr": os.getenv("ENABLE_ASR", "true").lower() == "true",
            "enable_llm": os.getenv("ENABLE_LLM", "true").lower() == "true",
            "enable_tts": os.getenv("ENABLE_TTS", "true").lower() == "true",
        },
        "server": {
            "host": os.getenv("SERVER_HOST", "0.0.0.0"),
            "port": int(os.getenv("SERVER_PORT", "8000")),
            "websocket_port": int(os.getenv("WEBSOCKET_PORT", "8003")),
            "auth_key": os.getenv("AUTH_KEY", ""),
            "debug": os.getenv("DEBUG", "false").lower() == "true",
        },
    }
    
    # 尝试从配置文件加载
    config_file = os.getenv("PRIVATE_CONFIG_FILE", "/opt/xiaozhi-esp32-server/config/private.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
                config.update(file_config)
        except Exception as e:
            print(f"Warning: Failed to load private config from {config_file}: {e}")
    
    return config


def get_config_from_api() -> Dict[str, Any]:
    """
    从API获取配置（兼容性函数）
    """
    return get_private_config_from_api()


def load_config_file(file_path: str) -> Optional[Dict[str, Any]]:
    """
    加载配置文件
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                return yaml.safe_load(f)
            elif file_path.endswith('.json'):
                return json.load(f)
            else:
                # 尝试作为YAML解析
                return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file {file_path}: {e}")
        return None

def get_config_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    从配置字典中获取值
    """
    keys = key.split('.')
    value = config
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个配置字典
    """
    result = {}
    for config in configs:
        if config:
            result.update(config)
    return result

def get_project_dir() -> str:
    """
    获取项目根目录
    """
    return os.getenv("PROJECT_DIR", "/opt/xiaozhi-esp32-server")