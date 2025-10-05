#!/usr/bin/env python3
"""
SenseVoice优化版本 - 解决多设备并发性能问题
主要优化点：
1. 连接级别的并发控制
2. 模型实例复用
3. 内存管理优化
4. 音频处理流水线优化
"""

import asyncio
import threading
import time
import tempfile
import os
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import queue
import gc

from funasr import AutoModel
from loguru import logger
from asr_providers.asr_provider_base import ASRProviderBase, InterfaceType
from utils.capture_output import CaptureOutput

TAG = "SenseVoiceOptimized"

class ModelPool:
    """SenseVoice模型池，支持模型复用和并发控制"""
    
    def __init__(self, model_dir: str, max_instances: int = 2):
        self.model_dir = model_dir
        self.max_instances = max_instances
        self.models = queue.Queue(maxsize=max_instances)
        self.lock = threading.Lock()
        self.initialized = False
        
    def initialize(self):
        """初始化模型池"""
        if self.initialized:
            return
            
        with self.lock:
            if self.initialized:
                return
                
            logger.bind(tag=TAG).info(f"初始化SenseVoice模型池，最大实例数: {self.max_instances}")
            
            # VAD配置
            vad_kwargs = {
                "vad_model": "fsmn-vad",
                "vad_kwargs": {"max_single_segment_time": 15000}
            }
            
            for i in range(self.max_instances):
                try:
                    with CaptureOutput():
                        model = AutoModel(model=self.model_dir, **vad_kwargs)
                    self.models.put(model)
                    logger.bind(tag=TAG).info(f"模型实例 {i+1} 初始化成功")
                except Exception as e:
                    logger.bind(tag=TAG).error(f"模型实例 {i+1} 初始化失败: {e}")
                    raise
                    
            self.initialized = True
            logger.bind(tag=TAG).info("SenseVoice模型池初始化完成")
    
    def get_model(self, timeout: float = 30.0):
        """获取模型实例"""
        try:
            return self.models.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("获取模型实例超时，所有实例都在使用中")
    
    def return_model(self, model):
        """归还模型实例"""
        try:
            self.models.put_nowait(model)
        except queue.Full:
            logger.bind(tag=TAG).warning("模型池已满，丢弃模型实例")

class ASRProvider(ASRProviderBase):
    # 类级别的模型池和线程池
    _model_pool = None
    _thread_pool = None
    _pool_lock = threading.Lock()
    
    def __init__(self, config: dict, delete_audio_file: bool):
        super().__init__()
        
        # 内存检测
        min_mem_bytes = 2 * 1024 * 1024 * 1024
        total_mem = psutil.virtual_memory().total
        if total_mem < min_mem_bytes:
            logger.bind(tag=TAG).error(f"可用内存不足2G，当前仅有 {total_mem / (1024*1024):.2f} MB")
        
        self.interface_type = InterfaceType.LOCAL
        self.model_dir = config.get("model_dir", "/opt/xiaozhi-esp32-server/models/SenseVoiceSmall")
        self.output_dir = config.get("output_dir", "/tmp")
        self.delete_audio_file = delete_audio_file
        
        # 并发控制配置
        self.max_concurrent_sessions = config.get("max_concurrent_sessions", 2)
        self.max_workers = config.get("asr_max_workers", 2)
        
        # 初始化类级别的资源
        with ASRProvider._pool_lock:
            if ASRProvider._model_pool is None:
                ASRProvider._model_pool = ModelPool(
                    self.model_dir, 
                    max_instances=self.max_concurrent_sessions
                )
                ASRProvider._model_pool.initialize()
                
            if ASRProvider._thread_pool is None:
                ASRProvider._thread_pool = ThreadPoolExecutor(
                    max_workers=self.max_workers,
                    thread_name_prefix="SenseVoice"
                )
                logger.bind(tag=TAG).info(f"ASR线程池初始化完成，最大工作线程数: {self.max_workers}")
    
    async def _recognize_audio_chunk_optimized(self, audio_data: bytes, session_id: str) -> str:
        """优化的音频识别方法"""
        start_time = time.time()
        
        # 获取模型实例
        try:
            model = ASRProvider._model_pool.get_model(timeout=10.0)
        except TimeoutError:
            logger.bind(tag=TAG).warning(f"Session {session_id}: 获取模型实例超时，当前并发过高")
            return ""
        
        try:
            # 在线程池中执行推理
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                ASRProvider._thread_pool,
                self._do_recognition,
                model,
                audio_data,
                session_id
            )
            
            recognition_time = time.time() - start_time
            logger.bind(tag=TAG).debug(f"Session {session_id}: 音频识别完成，耗时 {recognition_time:.2f}s")
            
            return result
            
        finally:
            # 归还模型实例
            ASRProvider._model_pool.return_model(model)
    
    def _do_recognition(self, model, audio_data: bytes, session_id: str) -> str:
        """执行实际的语音识别"""
        temp_file = None
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=self.output_dir) as f:
                f.write(audio_data)
                temp_file = f.name
            
            # 执行识别
            res = model.generate(input=temp_file)
            
            # 处理结果
            if res and len(res) > 0 and len(res[0]) > 0:
                text = res[0].get("text", "").strip()
                # 移除特殊标记
                text = text.replace("<|zh|>", "").replace("<|en|>", "").replace("<|nospeech|>", "").strip()
                return text
            
            return ""
            
        except Exception as e:
            logger.bind(tag=TAG).error(f"Session {session_id}: 语音识别失败: {e}")
            return ""
        finally:
            # 清理临时文件
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
            # 强制垃圾回收
            gc.collect()
    
    async def recognize_audio_chunk(self, audio_data: bytes, conn) -> str:
        """连接级别的音频识别接口"""
        session_id = getattr(conn, 'session_id', 'unknown')
        
        # 连接级别的并发控制
        if not hasattr(conn, 'asr_processing_lock'):
            conn.asr_processing_lock = asyncio.Lock()
        
        async with conn.asr_processing_lock:
            return await self._recognize_audio_chunk_optimized(audio_data, session_id)
    
    async def recognize_audio_complete(self, combined_audio_data: bytes, conn) -> str:
        """完整音频识别"""
        session_id = getattr(conn, 'session_id', 'unknown')
        logger.bind(tag=TAG).info(f"Session {session_id}: 开始完整音频识别，数据大小: {len(combined_audio_data)} bytes")
        
        return await self._recognize_audio_chunk_optimized(combined_audio_data, session_id)
    
    @classmethod
    def cleanup(cls):
        """清理资源"""
        with cls._pool_lock:
            if cls._thread_pool:
                cls._thread_pool.shutdown(wait=True)
                cls._thread_pool = None
            if cls._model_pool:
                cls._model_pool = None
        logger.bind(tag=TAG).info("ASR资源清理完成")