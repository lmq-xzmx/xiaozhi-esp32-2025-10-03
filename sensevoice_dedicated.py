#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专用服务器激进优化的SenseVoice ASR实现
适用于: 4核3GHz + 7.5GB内存的专用小智ESP32服务器
优化目标: 支持3-4台设备稳定并发，消除卡顿现象
"""

import asyncio
import gc
import logging
import os
import psutil
import tempfile
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, List, Optional, Any
import torch
import torchaudio

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    inference_time: float = 0.0
    queue_wait_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_sessions: int = 0
    total_requests: int = 0
    error_count: int = 0

class MemoryPool:
    """内存池管理器 - 专用服务器优化版本"""
    
    def __init__(self, initial_size: int = 3, max_size: int = 5):
        self.initial_size = initial_size
        self.max_size = max_size
        self.pool = Queue(maxsize=max_size)
        self.allocated_count = 0
        self.lock = threading.Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化内存池"""
        for _ in range(self.initial_size):
            # 预分配音频缓冲区
            buffer = torch.zeros(16000 * 30, dtype=torch.float32)  # 30秒音频缓冲
            self.pool.put(buffer)
            self.allocated_count += 1
    
    def get_buffer(self, size: int) -> torch.Tensor:
        """获取音频缓冲区"""
        try:
            buffer = self.pool.get_nowait()
            if buffer.size(0) >= size:
                return buffer[:size]
            else:
                # 扩展缓冲区
                return torch.zeros(size, dtype=torch.float32)
        except Empty:
            # 池为空，创建新缓冲区
            with self.lock:
                if self.allocated_count < self.max_size:
                    self.allocated_count += 1
                    return torch.zeros(size, dtype=torch.float32)
                else:
                    # 达到最大限制，等待可用缓冲区
                    return self.pool.get(timeout=1.0)
    
    def return_buffer(self, buffer: torch.Tensor):
        """归还缓冲区到池中"""
        try:
            # 清零缓冲区
            buffer.zero_()
            self.pool.put_nowait(buffer)
        except:
            # 池已满，丢弃缓冲区
            with self.lock:
                self.allocated_count -= 1

class ModelPool:
    """模型池 - 专用服务器激进优化版本"""
    
    def __init__(self, model_path: str, max_instances: int = 3, min_instances: int = 2):
        self.model_path = model_path
        self.max_instances = max_instances
        self.min_instances = min_instances
        self.available_models = Queue(maxsize=max_instances)
        self.total_instances = 0
        self.lock = threading.Lock()
        self.model_refs = []  # 弱引用列表
        self.warmup_complete = False
        
        # 性能监控
        self.metrics = PerformanceMetrics()
        self.last_gc_time = time.time()
        
        # 预加载模型实例
        self._preload_models()
    
    def _preload_models(self):
        """预加载模型实例"""
        logger.info(f"预加载 {self.min_instances} 个SenseVoice模型实例...")
        
        for i in range(self.min_instances):
            try:
                model = self._create_model_instance()
                self.available_models.put(model)
                self.total_instances += 1
                logger.info(f"成功预加载模型实例 {i+1}/{self.min_instances}")
            except Exception as e:
                logger.error(f"预加载模型实例 {i+1} 失败: {e}")
        
        # 预热模型
        self._warmup_models()
        self.warmup_complete = True
        logger.info("模型预加载和预热完成")
    
    def _create_model_instance(self):
        """创建模型实例"""
        try:
            # 导入SenseVoice模型
            from funasr import AutoModel
            
            # 创建模型实例，使用优化配置
            model = AutoModel(
                model=self.model_path,
                device="cpu",  # 专用服务器使用CPU
                ncpu=2,  # 每个模型实例使用2个CPU核心
                disable_update=True,
                disable_log=True
            )
            
            # 设置模型为评估模式
            if hasattr(model.model, 'eval'):
                model.model.eval()
            
            # 禁用梯度计算
            if hasattr(model.model, 'requires_grad_'):
                for param in model.model.parameters():
                    param.requires_grad_(False)
            
            # 添加弱引用
            self.model_refs.append(weakref.ref(model))
            
            return model
            
        except Exception as e:
            logger.error(f"创建模型实例失败: {e}")
            raise
    
    def _warmup_models(self):
        """预热所有模型实例"""
        logger.info("开始预热模型实例...")
        
        # 创建测试音频数据
        test_audio = torch.randn(16000, dtype=torch.float32)  # 1秒测试音频
        
        # 预热每个模型
        models_to_warmup = []
        while not self.available_models.empty():
            try:
                model = self.available_models.get_nowait()
                models_to_warmup.append(model)
            except Empty:
                break
        
        for i, model in enumerate(models_to_warmup):
            try:
                # 执行预热推理
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    torchaudio.save(temp_file.name, test_audio.unsqueeze(0), 16000)
                    
                    # 预热推理
                    start_time = time.time()
                    result = model.generate(input=temp_file.name)
                    warmup_time = time.time() - start_time
                    
                    logger.info(f"模型实例 {i+1} 预热完成，耗时: {warmup_time:.3f}s")
                    
                    # 清理临时文件
                    os.unlink(temp_file.name)
                    
            except Exception as e:
                logger.warning(f"模型实例 {i+1} 预热失败: {e}")
            finally:
                # 归还模型到池中
                self.available_models.put(model)
    
    @asynccontextmanager
    async def get_model(self):
        """异步获取模型实例"""
        model = None
        start_time = time.time()
        
        try:
            # 尝试从池中获取模型
            model = await asyncio.get_event_loop().run_in_executor(
                None, self._get_model_sync
            )
            
            self.metrics.queue_wait_time = time.time() - start_time
            self.metrics.active_sessions += 1
            
            yield model
            
        finally:
            if model is not None:
                # 归还模型到池中
                await asyncio.get_event_loop().run_in_executor(
                    None, self._return_model_sync, model
                )
                self.metrics.active_sessions -= 1
    
    def _get_model_sync(self):
        """同步获取模型"""
        try:
            # 首先尝试从可用池获取
            return self.available_models.get(timeout=2.0)
        except Empty:
            # 池为空，尝试创建新实例
            with self.lock:
                if self.total_instances < self.max_instances:
                    try:
                        model = self._create_model_instance()
                        self.total_instances += 1
                        logger.info(f"创建新模型实例，当前总数: {self.total_instances}")
                        return model
                    except Exception as e:
                        logger.error(f"创建新模型实例失败: {e}")
                        raise
                else:
                    # 达到最大实例数，等待可用模型
                    logger.warning("达到最大模型实例数，等待可用模型...")
                    return self.available_models.get(timeout=5.0)
    
    def _return_model_sync(self, model):
        """同步归还模型"""
        try:
            # 执行垃圾回收（定期）
            current_time = time.time()
            if current_time - self.last_gc_time > 60:  # 每分钟执行一次
                gc.collect()
                self.last_gc_time = current_time
            
            self.available_models.put_nowait(model)
        except:
            # 池已满，减少实例计数
            with self.lock:
                self.total_instances -= 1
                logger.info(f"模型池已满，释放实例，当前总数: {self.total_instances}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息"""
        return {
            "total_instances": self.total_instances,
            "available_instances": self.available_models.qsize(),
            "active_sessions": self.metrics.active_sessions,
            "total_requests": self.metrics.total_requests,
            "error_count": self.metrics.error_count,
            "warmup_complete": self.warmup_complete,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024
        }

class DedicatedSenseVoiceASR:
    """专用服务器激进优化的SenseVoice ASR提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 模型路径
        self.model_path = config.get('model_path', '/opt/xiaozhi-esp32-server/models/SenseVoiceSmall')
        
        # 并发控制配置 - 专用服务器激进设置
        self.max_concurrent_sessions = config.get('max_concurrent_sessions', 3)
        self.max_workers = config.get('max_workers', 3)
        
        # 性能配置
        self.chunk_size = config.get('chunk_size', 384)
        self.audio_buffer_size = config.get('audio_buffer_size', 4096)
        
        # 内存管理
        self.reserved_memory_gb = config.get('reserved_memory_gb', 3.0)
        
        # 初始化组件
        self._initialize_components()
        
        # 性能监控
        self.performance_monitor = PerformanceMetrics()
        self.start_time = time.time()
        
        logger.info("专用服务器SenseVoice ASR初始化完成")
    
    def _initialize_components(self):
        """初始化组件"""
        # 检查系统资源
        self._check_system_resources()
        
        # 初始化线程池 - 专用服务器可以使用更多线程
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="SenseVoice-Dedicated"
        )
        
        # 初始化模型池 - 专用服务器支持更多实例
        self.model_pool = ModelPool(
            model_path=self.model_path,
            max_instances=3,
            min_instances=2
        )
        
        # 初始化内存池
        self.memory_pool = MemoryPool(initial_size=3, max_size=5)
        
        # 连接级并发控制
        self.connection_locks = {}
        self.global_semaphore = asyncio.Semaphore(self.max_concurrent_sessions)
        
        # 性能优化设置
        self._setup_performance_optimizations()
    
    def _check_system_resources(self):
        """检查系统资源"""
        # 检查可用内存
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb < self.reserved_memory_gb:
            logger.warning(f"可用内存不足: {available_gb:.1f}GB < {self.reserved_memory_gb}GB")
        else:
            logger.info(f"系统内存检查通过: {available_gb:.1f}GB 可用")
        
        # 检查CPU核心数
        cpu_count = psutil.cpu_count()
        logger.info(f"CPU核心数: {cpu_count}")
        
        # 检查当前负载
        load_avg = psutil.getloadavg()
        logger.info(f"系统负载: {load_avg}")
    
    def _setup_performance_optimizations(self):
        """设置性能优化"""
        # 设置CPU亲和性（如果支持）
        try:
            import os
            if hasattr(os, 'sched_setaffinity'):
                # 绑定到前3个CPU核心
                os.sched_setaffinity(0, {0, 1, 2})
                logger.info("CPU亲和性设置成功: 核心 0, 1, 2")
        except Exception as e:
            logger.warning(f"设置CPU亲和性失败: {e}")
        
        # 设置进程优先级
        try:
            import psutil
            p = psutil.Process()
            p.nice(-5)  # 提高优先级
            logger.info("进程优先级设置成功")
        except Exception as e:
            logger.warning(f"设置进程优先级失败: {e}")
        
        # 禁用Python垃圾回收器的自动运行（手动控制）
        import gc
        gc.disable()
        logger.info("禁用自动垃圾回收，改为手动控制")
    
    async def recognize_audio(self, audio_data: bytes, connection_id: str) -> str:
        """识别音频 - 专用服务器优化版本"""
        start_time = time.time()
        
        # 获取连接级锁
        if connection_id not in self.connection_locks:
            self.connection_locks[connection_id] = asyncio.Lock()
        
        async with self.connection_locks[connection_id]:
            async with self.global_semaphore:
                try:
                    self.performance_monitor.total_requests += 1
                    
                    # 在线程池中执行识别
                    result = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._do_recognition,
                        audio_data
                    )
                    
                    # 更新性能指标
                    self.performance_monitor.inference_time = time.time() - start_time
                    
                    return result
                    
                except Exception as e:
                    self.performance_monitor.error_count += 1
                    logger.error(f"音频识别失败: {e}")
                    raise
                finally:
                    # 定期清理连接锁
                    if len(self.connection_locks) > 10:
                        self._cleanup_connection_locks()
    
    def _do_recognition(self, audio_data: bytes) -> str:
        """执行音频识别 - 同步版本"""
        temp_file = None
        
        try:
            # 获取内存缓冲区
            audio_buffer = self.memory_pool.get_buffer(len(audio_data) // 2)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio_data)
                temp_file = f.name
            
            # 使用模型池进行识别
            with self.model_pool.get_model() as model:
                # 执行识别
                result = model.generate(input=temp_file)
                
                # 处理结果
                if result and len(result) > 0:
                    text = result[0].get('text', '').strip()
                    return text
                else:
                    return ""
        
        except Exception as e:
            logger.error(f"音频识别执行失败: {e}")
            raise
        
        finally:
            # 清理临时文件
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
            # 归还内存缓冲区
            if 'audio_buffer' in locals():
                self.memory_pool.return_buffer(audio_buffer)
            
            # 定期执行垃圾回收
            if self.performance_monitor.total_requests % 50 == 0:
                import gc
                gc.collect()
    
    def _cleanup_connection_locks(self):
        """清理连接锁"""
        # 保留最近的5个连接锁
        if len(self.connection_locks) > 5:
            keys_to_remove = list(self.connection_locks.keys())[:-5]
            for key in keys_to_remove:
                del self.connection_locks[key]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        uptime = time.time() - self.start_time
        
        # 系统资源使用情况
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 模型池统计
        pool_stats = self.model_pool.get_stats()
        
        return {
            "uptime_seconds": uptime,
            "total_requests": self.performance_monitor.total_requests,
            "error_count": self.performance_monitor.error_count,
            "error_rate": self.performance_monitor.error_count / max(1, self.performance_monitor.total_requests),
            "avg_inference_time": self.performance_monitor.inference_time,
            "active_connections": len(self.connection_locks),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "load_average": psutil.getloadavg()
            },
            "model_pool": pool_stats,
            "thread_pool": {
                "active_threads": self.executor._threads,
                "max_workers": self.max_workers
            }
        }
    
    async def cleanup(self):
        """清理资源"""
        logger.info("开始清理SenseVoice ASR资源...")
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        # 清理连接锁
        self.connection_locks.clear()
        
        # 执行最终垃圾回收
        import gc
        gc.collect()
        
        logger.info("SenseVoice ASR资源清理完成")

# 工厂函数
def create_dedicated_sensevoice_asr(config: Dict[str, Any]) -> DedicatedSenseVoiceASR:
    """创建专用服务器SenseVoice ASR实例"""
    return DedicatedSenseVoiceASR(config)

# 性能测试函数
async def performance_test():
    """性能测试函数"""
    config = {
        'model_path': '/opt/xiaozhi-esp32-server/models/SenseVoiceSmall',
        'max_concurrent_sessions': 3,
        'max_workers': 3,
        'chunk_size': 384,
        'reserved_memory_gb': 3.0
    }
    
    asr = create_dedicated_sensevoice_asr(config)
    
    # 模拟并发测试
    test_audio = b'\x00' * 16000 * 2  # 1秒的空音频数据
    
    async def test_recognition(session_id: int):
        try:
            result = await asr.recognize_audio(test_audio, f"test_session_{session_id}")
            print(f"会话 {session_id} 识别结果: {result}")
        except Exception as e:
            print(f"会话 {session_id} 识别失败: {e}")
    
    # 并发测试
    tasks = [test_recognition(i) for i in range(5)]
    await asyncio.gather(*tasks)
    
    # 打印性能统计
    stats = asr.get_performance_stats()
    print("性能统计:", stats)
    
    # 清理
    await asr.cleanup()

if __name__ == "__main__":
    # 运行性能测试
    asyncio.run(performance_test())