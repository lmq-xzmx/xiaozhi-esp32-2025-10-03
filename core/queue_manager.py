#!/usr/bin/env python3
"""
智能队列管理和优先级调度系统
支持动态负载均衡、优先级调度、批处理优化
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

class Priority(Enum):
    """请求优先级"""
    CRITICAL = 1    # 紧急请求
    HIGH = 2        # 高优先级
    MEDIUM = 3      # 中等优先级
    LOW = 4         # 低优先级

@dataclass
class QueueRequest:
    """队列请求对象"""
    id: str
    data: Any
    priority: Priority
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0  # 超时时间
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """优先级比较，数值越小优先级越高"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp
    
    def is_expired(self) -> bool:
        """检查是否超时"""
        return time.time() - self.timestamp > self.timeout

@dataclass
class QueueStats:
    """队列统计信息"""
    total_requests: int = 0
    processed_requests: int = 0
    failed_requests: int = 0
    avg_processing_time: float = 0.0
    queue_length: int = 0
    current_load: float = 0.0
    
class IntelligentQueueManager:
    """智能队列管理器"""
    
    def __init__(self, 
                 max_queue_size: int = 10000,
                 batch_size: int = 16,
                 batch_timeout: float = 0.1,
                 max_concurrent: int = 32):
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.max_concurrent = max_concurrent
        
        # 优先级队列
        self.priority_queue = []
        self.queue_lock = asyncio.Lock()
        
        # 批处理队列
        self.batch_queue = deque()
        self.batch_event = asyncio.Event()
        
        # 统计信息
        self.stats = QueueStats()
        self.processing_times = deque(maxlen=1000)
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = set()
        
        # 负载均衡
        self.load_history = deque(maxlen=100)
        self.last_load_check = time.time()
        
        # 启动后台任务
        self.running = True
        self.batch_processor_task = None
        self.cleanup_task = None
        self.stats_task = None
    
    async def start(self):
        """启动队列管理器"""
        self.batch_processor_task = asyncio.create_task(self._batch_processor())
        self.cleanup_task = asyncio.create_task(self._cleanup_expired())
        self.stats_task = asyncio.create_task(self._update_stats())
        logger.info("智能队列管理器启动成功")
    
    async def stop(self):
        """停止队列管理器"""
        self.running = False
        
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.stats_task:
            self.stats_task.cancel()
        
        # 等待所有活跃任务完成
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)
        
        logger.info("智能队列管理器已停止")
    
    async def enqueue(self, request: QueueRequest) -> bool:
        """添加请求到队列"""
        async with self.queue_lock:
            if len(self.priority_queue) >= self.max_queue_size:
                logger.warning(f"队列已满，拒绝请求 {request.id}")
                return False
            
            heapq.heappush(self.priority_queue, request)
            self.stats.total_requests += 1
            self.batch_event.set()  # 通知批处理器
            
            logger.debug(f"请求 {request.id} 已加入队列，优先级: {request.priority.name}")
            return True
    
    async def process_request(self, processor_func: Callable, *args, **kwargs) -> Any:
        """处理单个请求"""
        async with self.semaphore:
            start_time = time.time()
            task = asyncio.current_task()
            self.active_tasks.add(task)
            
            try:
                result = await processor_func(*args, **kwargs)
                self.stats.processed_requests += 1
                
                # 记录处理时间
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                return result
            except Exception as e:
                self.stats.failed_requests += 1
                logger.error(f"请求处理失败: {e}")
                raise
            finally:
                self.active_tasks.discard(task)
    
    async def _batch_processor(self):
        """批处理器"""
        while self.running:
            try:
                # 等待有请求或超时
                await asyncio.wait_for(self.batch_event.wait(), timeout=self.batch_timeout)
                self.batch_event.clear()
                
                # 收集批处理请求
                batch = await self._collect_batch()
                if batch:
                    # 创建批处理任务
                    task = asyncio.create_task(self._process_batch(batch))
                    self.active_tasks.add(task)
                
            except asyncio.TimeoutError:
                # 超时也要检查是否有请求需要处理
                batch = await self._collect_batch()
                if batch:
                    task = asyncio.create_task(self._process_batch(batch))
                    self.active_tasks.add(task)
            except Exception as e:
                logger.error(f"批处理器错误: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[QueueRequest]:
        """收集批处理请求"""
        batch = []
        async with self.queue_lock:
            while len(batch) < self.batch_size and self.priority_queue:
                request = heapq.heappop(self.priority_queue)
                if not request.is_expired():
                    batch.append(request)
                else:
                    logger.warning(f"请求 {request.id} 已超时，丢弃")
                    self.stats.failed_requests += 1
        
        return batch
    
    async def _process_batch(self, batch: List[QueueRequest]):
        """处理批次请求"""
        if not batch:
            return
        
        start_time = time.time()
        task = asyncio.current_task()
        
        try:
            # 按优先级分组处理
            priority_groups = defaultdict(list)
            for request in batch:
                priority_groups[request.priority].append(request)
            
            # 优先处理高优先级请求
            for priority in sorted(priority_groups.keys(), key=lambda x: x.value):
                requests = priority_groups[priority]
                await self._process_priority_group(requests)
            
            # 更新统计
            processing_time = time.time() - start_time
            logger.debug(f"批处理完成，处理 {len(batch)} 个请求，耗时 {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"批处理失败: {e}")
            for request in batch:
                self.stats.failed_requests += 1
        finally:
            self.active_tasks.discard(task)
    
    async def _process_priority_group(self, requests: List[QueueRequest]):
        """处理同优先级请求组"""
        tasks = []
        for request in requests:
            if request.callback:
                task = asyncio.create_task(self._execute_request(request))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _execute_request(self, request: QueueRequest):
        """执行单个请求"""
        try:
            if request.callback:
                result = await request.callback(request.data)
                self.stats.processed_requests += 1
                return result
        except Exception as e:
            logger.error(f"请求 {request.id} 执行失败: {e}")
            self.stats.failed_requests += 1
            raise
    
    async def _cleanup_expired(self):
        """清理过期请求"""
        while self.running:
            try:
                async with self.queue_lock:
                    # 清理过期请求
                    valid_requests = []
                    expired_count = 0
                    
                    while self.priority_queue:
                        request = heapq.heappop(self.priority_queue)
                        if not request.is_expired():
                            valid_requests.append(request)
                        else:
                            expired_count += 1
                            self.stats.failed_requests += 1
                    
                    # 重建队列
                    self.priority_queue = valid_requests
                    heapq.heapify(self.priority_queue)
                    
                    if expired_count > 0:
                        logger.info(f"清理了 {expired_count} 个过期请求")
                
                await asyncio.sleep(10)  # 每10秒清理一次
            except Exception as e:
                logger.error(f"清理过期请求失败: {e}")
                await asyncio.sleep(1)
    
    async def _update_stats(self):
        """更新统计信息"""
        while self.running:
            try:
                # 更新队列长度
                async with self.queue_lock:
                    self.stats.queue_length = len(self.priority_queue)
                
                # 更新平均处理时间
                if self.processing_times:
                    self.stats.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
                
                # 更新当前负载
                current_load = len(self.active_tasks) / self.max_concurrent
                self.stats.current_load = current_load
                self.load_history.append(current_load)
                
                await asyncio.sleep(1)  # 每秒更新一次
            except Exception as e:
                logger.error(f"更新统计信息失败: {e}")
                await asyncio.sleep(1)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_rate = 0.0
        if self.stats.total_requests > 0:
            success_rate = (self.stats.processed_requests / self.stats.total_requests) * 100
        
        avg_load = 0.0
        if self.load_history:
            avg_load = sum(self.load_history) / len(self.load_history)
        
        return {
            "total_requests": self.stats.total_requests,
            "processed_requests": self.stats.processed_requests,
            "failed_requests": self.stats.failed_requests,
            "success_rate": round(success_rate, 2),
            "avg_processing_time": round(self.stats.avg_processing_time, 3),
            "queue_length": self.stats.queue_length,
            "current_load": round(self.stats.current_load, 2),
            "avg_load": round(avg_load, 2),
            "active_tasks": len(self.active_tasks),
            "max_concurrent": self.max_concurrent
        }
    
    def adjust_concurrency(self, new_max_concurrent: int):
        """动态调整并发数"""
        if new_max_concurrent > 0:
            self.max_concurrent = new_max_concurrent
            self.semaphore = asyncio.Semaphore(new_max_concurrent)
            logger.info(f"并发数已调整为 {new_max_concurrent}")
    
    def adjust_batch_size(self, new_batch_size: int):
        """动态调整批处理大小"""
        if new_batch_size > 0:
            self.batch_size = new_batch_size
            logger.info(f"批处理大小已调整为 {new_batch_size}")

# 全局队列管理器实例
_queue_managers: Dict[str, IntelligentQueueManager] = {}

def get_queue_manager(service_name: str, **kwargs) -> IntelligentQueueManager:
    """获取服务的队列管理器"""
    if service_name not in _queue_managers:
        _queue_managers[service_name] = IntelligentQueueManager(**kwargs)
    return _queue_managers[service_name]

async def start_all_queue_managers():
    """启动所有队列管理器"""
    for manager in _queue_managers.values():
        await manager.start()

async def stop_all_queue_managers():
    """停止所有队列管理器"""
    for manager in _queue_managers.values():
        await manager.stop()