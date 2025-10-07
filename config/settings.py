#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
from typing import Dict, Any, Optional

def load_config() -> Dict[str, Any]:
    """
    加载系统配置
    """
    config = {
        # 基础配置
        "app": {
            "name": "xiaozhi-esp32-server",
            "version": "1.0.0",
            "debug": os.getenv("DEBUG", "false").lower() == "true",
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "websocket_port": int(os.getenv("WEBSOCKET_PORT", "8003")),
        },
        
        # 数据库配置
        "database": {
            "host": os.getenv("DB_HOST", "xiaozhi-esp32-server-db"),
            "port": int(os.getenv("DB_PORT", "3306")),
            "name": os.getenv("DB_NAME", "xiaozhi"),
            "user": os.getenv("DB_USER", "xiaozhi"),
            "password": os.getenv("DB_PASSWORD", "xiaozhi123"),
            "charset": "utf8mb4",
            "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
        },
        
        # Redis配置
        "redis": {
            "host": os.getenv("REDIS_HOST", "xiaozhi-esp32-server-redis"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD", ""),
            "decode_responses": True,
            "socket_keepalive": True,
            "socket_keepalive_options": {},
            "health_check_interval": 30,
        },
        
        # VAD配置
        "vad": {
            "model_path": os.getenv("VAD_MODEL_PATH", "/opt/xiaozhi-esp32-server/models/silero_vad.onnx"),
            "batch_size": int(os.getenv("VAD_BATCH_SIZE", "32")),
            "max_concurrent": int(os.getenv("VAD_MAX_CONCURRENT", "48")),
            "worker_threads": int(os.getenv("VAD_WORKER_THREADS", "4")),
            "enable_onnx": os.getenv("VAD_ENABLE_ONNX", "true").lower() == "true",
            "onnx_providers": [os.getenv("VAD_ONNX_PROVIDERS", "CPUExecutionProvider")],
            "graph_optimization": os.getenv("VAD_GRAPH_OPTIMIZATION", "all"),
            "intra_op_threads": int(os.getenv("VAD_INTRA_OP_THREADS", "2")),
            "inter_op_threads": int(os.getenv("VAD_INTER_OP_THREADS", "2")),
            "enable_memory_pattern": os.getenv("VAD_ENABLE_MEMORY_PATTERN", "true").lower() == "true",
            "enable_memory_arena": os.getenv("VAD_ENABLE_MEMORY_ARENA", "true").lower() == "true",
        },
        
        # ASR配置
        "asr": {
            "model_path": os.getenv("ASR_MODEL_PATH", "/opt/xiaozhi-esp32-server/models/SenseVoiceSmall"),
            "batch_size": int(os.getenv("ASR_BATCH_SIZE", "16")),
            "max_concurrent": int(os.getenv("ASR_MAX_CONCURRENT", "32")),
            "worker_threads": int(os.getenv("ASR_WORKER_THREADS", "3")),
            "stream_workers": int(os.getenv("ASR_STREAM_WORKERS", "2")),
            "enable_realtime": os.getenv("ENABLE_REALTIME", "true").lower() == "true",
            "chunk_size": int(os.getenv("CHUNK_SIZE", "1024")),
            "stream_buffer_size": int(os.getenv("STREAM_BUFFER_SIZE", "8192")),
            "dynamic_batching": os.getenv("DYNAMIC_BATCHING", "true").lower() == "true",
            "batch_timeout_ms": int(os.getenv("BATCH_TIMEOUT_MS", "50")),
            "enable_fp16": os.getenv("ENABLE_FP16", "true").lower() == "true",
            "enable_quantization": os.getenv("ENABLE_QUANTIZATION", "true").lower() == "true",
        },
        
        # LLM配置
        "llm": {
            "max_concurrent": int(os.getenv("LLM_MAX_CONCURRENT", "20")),
            "queue_size": int(os.getenv("LLM_QUEUE_SIZE", "50")),
            "request_timeout": int(os.getenv("LLM_REQUEST_TIMEOUT", "30")),
            "worker_threads": int(os.getenv("LLM_WORKER_THREADS", "2")),
            "enable_cache": os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true",
            "cache_ttl": int(os.getenv("LLM_CACHE_TTL", "1800")),
            "semantic_cache": os.getenv("SEMANTIC_CACHE", "true").lower() == "true",
            "apis": {
                "qwen": {
                    "api_key": os.getenv("QWEN_API_KEY", ""),
                    "base_url": os.getenv("QWEN_BASE_URL", ""),
                    "model": os.getenv("QWEN_MODEL", "qwen-turbo"),
                    "weight": 0.4,
                },
                "baichuan": {
                    "api_key": os.getenv("BAICHUAN_API_KEY", ""),
                    "base_url": os.getenv("BAICHUAN_BASE_URL", ""),
                    "model": os.getenv("BAICHUAN_MODEL", "Baichuan2-Turbo"),
                    "weight": 0.3,
                },
                "openai": {
                    "api_key": os.getenv("OPENAI_API_KEY", ""),
                    "base_url": os.getenv("OPENAI_BASE_URL", ""),
                    "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
                    "weight": 0.3,
                },
            },
        },
        
        # TTS配置
        "tts": {
            "max_concurrent": int(os.getenv("TTS_MAX_CONCURRENT", "15")),
            "queue_size": int(os.getenv("TTS_QUEUE_SIZE", "40")),
            "request_timeout": int(os.getenv("TTS_REQUEST_TIMEOUT", "20")),
            "worker_threads": int(os.getenv("TTS_WORKER_THREADS", "2")),
            "audio_format": os.getenv("AUDIO_FORMAT", "mp3"),
            "audio_quality": os.getenv("AUDIO_QUALITY", "medium"),
            "sample_rate": int(os.getenv("SAMPLE_RATE", "22050")),
            "bit_rate": int(os.getenv("BIT_RATE", "64")),
            "enable_compression": os.getenv("ENABLE_COMPRESSION", "true").lower() == "true",
            "enable_cache": os.getenv("ENABLE_TTS_CACHE", "true").lower() == "true",
            "cache_ttl": int(os.getenv("TTS_CACHE_TTL", "3600")),
            "engines": {
                "edge": {
                    "weight": 0.4,
                    "voice": "zh-CN-XiaoxiaoNeural",
                },
                "azure": {
                    "api_key": os.getenv("AZURE_TTS_KEY", ""),
                    "region": os.getenv("AZURE_TTS_REGION", "eastasia"),
                    "weight": 0.3,
                    "voice": "zh-CN-XiaoxiaoNeural",
                },
                "xunfei": {
                    "api_key": os.getenv("XUNFEI_TTS_KEY", ""),
                    "weight": 0.2,
                    "voice": "xiaoyan",
                },
                "local": {
                    "weight": 0.1,
                    "model_path": "/opt/xiaozhi-esp32-server/models/tts",
                },
            },
        },
        
        # 缓存配置
        "cache": {
            "enable": os.getenv("ENABLE_CACHE", "true").lower() == "true",
            "ttl": int(os.getenv("CACHE_TTL", "600")),
            "max_size": int(os.getenv("CACHE_MAX_SIZE", "1000")),
        },
        
        # 监控配置
        "monitoring": {
            "enable": os.getenv("ENABLE_MONITORING", "true").lower() == "true",
            "cpu_warning_threshold": int(os.getenv("CPU_WARNING_THRESHOLD", "80")),
            "cpu_critical_threshold": int(os.getenv("CPU_CRITICAL_THRESHOLD", "90")),
            "memory_warning_threshold": int(os.getenv("MEMORY_WARNING_THRESHOLD", "80")),
            "memory_critical_threshold": int(os.getenv("MEMORY_CRITICAL_THRESHOLD", "85")),
            "enable_overload_protection": os.getenv("ENABLE_OVERLOAD_PROTECTION", "true").lower() == "true",
        },
        
        # 系统配置
        "system": {
            "omp_num_threads": int(os.getenv("OMP_NUM_THREADS", "4")),
            "mkl_num_threads": int(os.getenv("MKL_NUM_THREADS", "4")),
            "numba_num_threads": int(os.getenv("NUMBA_NUM_THREADS", "4")),
            "malloc_arena_max": int(os.getenv("MALLOC_ARENA_MAX", "2")),
            "malloc_mmap_threshold": int(os.getenv("MALLOC_MMAP_THRESHOLD_", "131072")),
        },
        
        # 日志配置
        "logging": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": os.getenv("LOG_FILE", "/opt/xiaozhi-esp32-server/logs/app.log"),
            "max_bytes": int(os.getenv("LOG_MAX_BYTES", "10485760")),  # 10MB
            "backup_count": int(os.getenv("LOG_BACKUP_COUNT", "5")),
        },
    }
    
    return config

def get_config_value(key: str, default: Any = None) -> Any:
    """
    获取配置值
    """
    config = load_config()
    keys = key.split('.')
    value = config
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def load_yaml_config(file_path: str) -> Optional[Dict[str, Any]]:
    """
    加载YAML配置文件
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML config {file_path}: {e}")
    return None