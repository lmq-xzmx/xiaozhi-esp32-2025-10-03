#!/usr/bin/env python3
"""
Xiaozhi ESP32 Server - 多级缓存系统
实现L1内存缓存、L2 Redis缓存、L3对象存储的多级缓存架构
"""

import asyncio
import json
import time
import hashlib
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

import aioredis
import aiofiles
import numpy as np
from cachetools import TTLCache, LRUCache
import boto3
from botocore.exceptions import ClientError

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheLevel(Enum):
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_STORAGE = "l3_storage"

class CacheType(Enum):
    VAD_RESULT = "vad_result"
    ASR_RESULT = "asr_result"
    LLM_RESPONSE = "llm_response"
    TTS_AUDIO = "tts_audio"
    MODEL_WEIGHTS = "model_weights"
    SESSION_CONTEXT = "session_context"
    PRECOMPUTED_FEATURES = "precomputed_features"

@dataclass
class CacheItem:
    """缓存项"""
    key: str
    value: Any
    cache_type: CacheType
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = None

@dataclass
class CacheStats:
    """缓存统计"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: int = 0
    redis_usage: int = 0
    storage_usage: int = 0

class IntelligentPreloader:
    """智能预加载器"""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.access_patterns = {}  # 访问模式分析
        self.prediction_models = {}  # 预测模型
        self.preload_queue = asyncio.Queue()
        self.is_running = False
        
    async def start(self):
        """启动预加载器"""
        self.is_running = True
        asyncio.create_task(self._preload_worker())
        asyncio.create_task(self._pattern_analyzer())
        logger.info("Intelligent preloader started")
    
    async def stop(self):
        """停止预加载器"""
        self.is_running = False
        logger.info("Intelligent preloader stopped")
    
    def record_access(self, key: str, cache_type: CacheType, context: Dict[str, Any] = None):
        """记录访问模式"""
        current_time = datetime.now()
        
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                'access_times': [],
                'contexts': [],
                'frequency': 0,
                'last_access': None
            }
        
        pattern = self.access_patterns[key]
        pattern['access_times'].append(current_time)
        pattern['contexts'].append(context or {})
        pattern['frequency'] += 1
        pattern['last_access'] = current_time
        
        # 保持最近1000次访问记录
        if len(pattern['access_times']) > 1000:
            pattern['access_times'] = pattern['access_times'][-1000:]
            pattern['contexts'] = pattern['contexts'][-1000:]
    
    async def predict_and_preload(self, current_context: Dict[str, Any]):
        """预测并预加载可能需要的数据"""
        predictions = await self._predict_next_access(current_context)
        
        for prediction in predictions:
            if prediction['confidence'] > 0.7:  # 置信度阈值
                await self.preload_queue.put(prediction)
    
    async def _predict_next_access(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """预测下一次访问"""
        predictions = []
        
        # 基于时间模式的预测
        time_predictions = self._predict_by_time_pattern()
        predictions.extend(time_predictions)
        
        # 基于上下文的预测
        context_predictions = self._predict_by_context(context)
        predictions.extend(context_predictions)
        
        # 基于序列模式的预测
        sequence_predictions = self._predict_by_sequence()
        predictions.extend(sequence_predictions)
        
        # 按置信度排序
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions[:10]  # 返回前10个预测
    
    def _predict_by_time_pattern(self) -> List[Dict[str, Any]]:
        """基于时间模式预测"""
        predictions = []
        current_time = datetime.now()
        
        for key, pattern in self.access_patterns.items():
            if len(pattern['access_times']) < 3:
                continue
            
            # 分析访问时间间隔
            intervals = []
            for i in range(1, len(pattern['access_times'])):
                interval = (pattern['access_times'][i] - pattern['access_times'][i-1]).total_seconds()
                intervals.append(interval)
            
            if intervals:
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                
                # 预测下次访问时间
                last_access = pattern['last_access']
                if last_access:
                    time_since_last = (current_time - last_access).total_seconds()
                    
                    # 如果接近平均间隔时间，增加预测置信度
                    if abs(time_since_last - avg_interval) < std_interval:
                        confidence = max(0.5, 1.0 - abs(time_since_last - avg_interval) / avg_interval)
                        predictions.append({
                            'key': key,
                            'confidence': confidence,
                            'reason': 'time_pattern',
                            'predicted_time': current_time + timedelta(seconds=avg_interval - time_since_last)
                        })
        
        return predictions
    
    def _predict_by_context(self, current_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于上下文预测"""
        predictions = []
        
        for key, pattern in self.access_patterns.items():
            if not pattern['contexts']:
                continue
            
            # 计算上下文相似度
            similarities = []
            for historical_context in pattern['contexts'][-50:]:  # 最近50次
                similarity = self._calculate_context_similarity(current_context, historical_context)
                similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                if avg_similarity > 0.6:  # 相似度阈值
                    predictions.append({
                        'key': key,
                        'confidence': avg_similarity,
                        'reason': 'context_similarity',
                        'similarity_score': avg_similarity
                    })
        
        return predictions
    
    def _predict_by_sequence(self) -> List[Dict[str, Any]]:
        """基于序列模式预测"""
        predictions = []
        
        # 分析访问序列
        recent_accesses = []
        for key, pattern in self.access_patterns.items():
            if pattern['last_access'] and (datetime.now() - pattern['last_access']).seconds < 300:  # 5分钟内
                recent_accesses.append((pattern['last_access'], key))
        
        # 按时间排序
        recent_accesses.sort(key=lambda x: x[0])
        recent_keys = [key for _, key in recent_accesses[-5:]]  # 最近5次访问
        
        # 查找历史序列模式
        for key, pattern in self.access_patterns.items():
            if len(pattern['access_times']) < 10:
                continue
            
            # 简单的序列匹配
            historical_sequences = self._extract_sequences(pattern['access_times'])
            match_score = self._match_sequence(recent_keys, historical_sequences)
            
            if match_score > 0.5:
                predictions.append({
                    'key': key,
                    'confidence': match_score,
                    'reason': 'sequence_pattern',
                    'match_score': match_score
                })
        
        return predictions
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """计算上下文相似度"""
        if not context1 or not context2:
            return 0.0
        
        # 简单的Jaccard相似度
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        intersection = len(keys1.intersection(keys2))
        union = len(keys1.union(keys2))
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        
        # 考虑值的相似度
        value_similarity = 0.0
        common_keys = keys1.intersection(keys2)
        
        if common_keys:
            similarities = []
            for key in common_keys:
                val1, val2 = context1[key], context2[key]
                if val1 == val2:
                    similarities.append(1.0)
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # 数值相似度
                    max_val = max(abs(val1), abs(val2))
                    if max_val > 0:
                        similarities.append(1.0 - abs(val1 - val2) / max_val)
                    else:
                        similarities.append(1.0)
                else:
                    similarities.append(0.0)
            
            value_similarity = np.mean(similarities)
        
        return (jaccard + value_similarity) / 2
    
    def _extract_sequences(self, access_times: List[datetime]) -> List[List[str]]:
        """提取访问序列"""
        # 这里简化实现，实际应用中可以使用更复杂的序列分析
        return []
    
    def _match_sequence(self, recent_keys: List[str], historical_sequences: List[List[str]]) -> float:
        """匹配序列模式"""
        # 简化实现
        return 0.0
    
    async def _preload_worker(self):
        """预加载工作线程"""
        while self.is_running:
            try:
                # 等待预加载任务
                prediction = await asyncio.wait_for(self.preload_queue.get(), timeout=1.0)
                
                # 执行预加载
                await self._execute_preload(prediction)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Preload worker error: {e}")
    
    async def _execute_preload(self, prediction: Dict[str, Any]):
        """执行预加载"""
        key = prediction['key']
        
        # 检查是否已经在缓存中
        if await self.cache_manager.exists(key):
            return
        
        # 根据key类型执行相应的预加载逻辑
        try:
            if 'llm_' in key:
                await self._preload_llm_response(key)
            elif 'tts_' in key:
                await self._preload_tts_audio(key)
            elif 'asr_' in key:
                await self._preload_asr_result(key)
            
            logger.info(f"Preloaded: {key} (confidence: {prediction['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to preload {key}: {e}")
    
    async def _preload_llm_response(self, key: str):
        """预加载LLM响应"""
        # 这里应该调用LLM服务生成响应
        # 暂时跳过实际实现
        pass
    
    async def _preload_tts_audio(self, key: str):
        """预加载TTS音频"""
        # 这里应该调用TTS服务生成音频
        # 暂时跳过实际实现
        pass
    
    async def _preload_asr_result(self, key: str):
        """预加载ASR结果"""
        # 这里应该调用ASR服务处理音频
        # 暂时跳过实际实现
        pass
    
    async def _pattern_analyzer(self):
        """模式分析器"""
        while self.is_running:
            try:
                await self._analyze_patterns()
                await asyncio.sleep(300)  # 每5分钟分析一次
            except Exception as e:
                logger.error(f"Pattern analyzer error: {e}")
    
    async def _analyze_patterns(self):
        """分析访问模式"""
        # 清理过期的访问记录
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for key in list(self.access_patterns.keys()):
            pattern = self.access_patterns[key]
            
            # 移除24小时前的访问记录
            pattern['access_times'] = [
                t for t in pattern['access_times'] if t > cutoff_time
            ]
            pattern['contexts'] = pattern['contexts'][-len(pattern['access_times']):]
            
            # 如果没有最近的访问记录，删除该模式
            if not pattern['access_times']:
                del self.access_patterns[key]

class L1MemoryCache:
    """L1内存缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.stats = CacheStats()
        self.lock = threading.RLock()
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self.lock:
            self.stats.total_requests += 1
            
            if key in self.cache:
                self.stats.hits += 1
                value = self.cache[key]
                
                # 更新访问信息
                if isinstance(value, CacheItem):
                    value.access_count += 1
                    value.last_accessed = datetime.now()
                    return value.value
                return value
            else:
                self.stats.misses += 1
                return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        with self.lock:
            try:
                if isinstance(value, CacheItem):
                    self.cache[key] = value
                else:
                    cache_item = CacheItem(
                        key=key,
                        value=value,
                        cache_type=CacheType.SESSION_CONTEXT,  # 默认类型
                        created_at=datetime.now(),
                        expires_at=datetime.now() + timedelta(seconds=ttl) if ttl else None,
                        size_bytes=len(pickle.dumps(value))
                    )
                    self.cache[key] = cache_item
                
                return True
            except Exception as e:
                logger.error(f"L1 cache set error: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """检查缓存项是否存在"""
        with self.lock:
            return key in self.cache
    
    async def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        with self.lock:
            self.stats.hit_rate = self.stats.hits / max(self.stats.total_requests, 1)
            self.stats.memory_usage = sum(
                item.size_bytes if isinstance(item, CacheItem) else len(pickle.dumps(item))
                for item in self.cache.values()
            )
            return self.stats

class L2RedisCache:
    """L2 Redis缓存"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.stats = CacheStats()
    
    async def initialize(self):
        """初始化Redis连接"""
        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            logger.info("L2 Redis cache initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        if not self.redis_client:
            return None
        
        try:
            self.stats.total_requests += 1
            
            data = await self.redis_client.get(key)
            if data:
                self.stats.hits += 1
                
                # 尝试反序列化
                try:
                    cache_item = pickle.loads(data)
                    if isinstance(cache_item, CacheItem):
                        cache_item.access_count += 1
                        cache_item.last_accessed = datetime.now()
                        
                        # 更新Redis中的访问信息
                        await self.redis_client.set(key, pickle.dumps(cache_item))
                        
                        return cache_item.value
                    return cache_item
                except:
                    # 如果反序列化失败，返回原始数据
                    return data.decode('utf-8')
            else:
                self.stats.misses += 1
                return None
                
        except Exception as e:
            logger.error(f"L2 cache get error: {e}")
            self.stats.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        if not self.redis_client:
            return False
        
        try:
            if isinstance(value, CacheItem):
                data = pickle.dumps(value)
            else:
                cache_item = CacheItem(
                    key=key,
                    value=value,
                    cache_type=CacheType.SESSION_CONTEXT,
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=ttl) if ttl else None,
                    size_bytes=len(pickle.dumps(value))
                )
                data = pickle.dumps(cache_item)
            
            if ttl:
                await self.redis_client.setex(key, ttl, data)
            else:
                await self.redis_client.set(key, data)
            
            return True
            
        except Exception as e:
            logger.error(f"L2 cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"L2 cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """检查缓存项是否存在"""
        if not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"L2 cache exists error: {e}")
            return False
    
    async def clear(self):
        """清空缓存"""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.flushdb()
        except Exception as e:
            logger.error(f"L2 cache clear error: {e}")
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        self.stats.hit_rate = self.stats.hits / max(self.stats.total_requests, 1)
        return self.stats

class L3StorageCache:
    """L3对象存储缓存"""
    
    def __init__(self, bucket_name: str = "xiaozhi-cache", 
                 aws_access_key: Optional[str] = None,
                 aws_secret_key: Optional[str] = None,
                 endpoint_url: Optional[str] = None):
        self.bucket_name = bucket_name
        self.s3_client = None
        self.stats = CacheStats()
        self.local_cache_dir = "/tmp/xiaozhi_l3_cache"
        
        # 创建本地缓存目录
        import os
        os.makedirs(self.local_cache_dir, exist_ok=True)
        
        # 初始化S3客户端
        try:
            if aws_access_key and aws_secret_key:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    endpoint_url=endpoint_url
                )
            else:
                # 使用默认凭证
                self.s3_client = boto3.client('s3', endpoint_url=endpoint_url)
            
            # 创建bucket（如果不存在）
            self._create_bucket_if_not_exists()
            
        except Exception as e:
            logger.warning(f"S3 client initialization failed, using local storage only: {e}")
            self.s3_client = None
    
    def _create_bucket_if_not_exists(self):
        """创建bucket（如果不存在）"""
        if not self.s3_client:
            return
        
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                try:
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                    logger.info(f"Created S3 bucket: {self.bucket_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create S3 bucket: {create_error}")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        self.stats.total_requests += 1
        
        # 首先尝试从本地缓存获取
        local_path = f"{self.local_cache_dir}/{self._safe_filename(key)}"
        
        try:
            if os.path.exists(local_path):
                async with aiofiles.open(local_path, 'rb') as f:
                    data = await f.read()
                    cache_item = pickle.loads(data)
                    
                    # 检查是否过期
                    if isinstance(cache_item, CacheItem) and cache_item.expires_at:
                        if datetime.now() > cache_item.expires_at:
                            await self.delete(key)
                            self.stats.misses += 1
                            return None
                    
                    self.stats.hits += 1
                    
                    if isinstance(cache_item, CacheItem):
                        cache_item.access_count += 1
                        cache_item.last_accessed = datetime.now()
                        return cache_item.value
                    return cache_item
        except Exception as e:
            logger.error(f"Local cache read error: {e}")
        
        # 如果本地缓存没有，尝试从S3获取
        if self.s3_client:
            try:
                response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
                data = response['Body'].read()
                cache_item = pickle.loads(data)
                
                # 保存到本地缓存
                async with aiofiles.open(local_path, 'wb') as f:
                    await f.write(data)
                
                self.stats.hits += 1
                
                if isinstance(cache_item, CacheItem):
                    return cache_item.value
                return cache_item
                
            except ClientError as e:
                if e.response['Error']['Code'] != 'NoSuchKey':
                    logger.error(f"S3 get error: {e}")
            except Exception as e:
                logger.error(f"S3 get error: {e}")
        
        self.stats.misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        try:
            if isinstance(value, CacheItem):
                cache_item = value
            else:
                cache_item = CacheItem(
                    key=key,
                    value=value,
                    cache_type=CacheType.TTS_AUDIO,  # L3主要用于大文件
                    created_at=datetime.now(),
                    expires_at=datetime.now() + timedelta(seconds=ttl) if ttl else None,
                    size_bytes=len(pickle.dumps(value))
                )
            
            data = pickle.dumps(cache_item)
            
            # 保存到本地缓存
            local_path = f"{self.local_cache_dir}/{self._safe_filename(key)}"
            async with aiofiles.open(local_path, 'wb') as f:
                await f.write(data)
            
            # 保存到S3（如果可用）
            if self.s3_client:
                try:
                    self.s3_client.put_object(
                        Bucket=self.bucket_name,
                        Key=key,
                        Body=data,
                        Metadata={
                            'cache_type': cache_item.cache_type.value,
                            'created_at': cache_item.created_at.isoformat(),
                            'expires_at': cache_item.expires_at.isoformat() if cache_item.expires_at else ''
                        }
                    )
                except Exception as e:
                    logger.error(f"S3 put error: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"L3 cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存项"""
        success = True
        
        # 删除本地缓存
        local_path = f"{self.local_cache_dir}/{self._safe_filename(key)}"
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except Exception as e:
            logger.error(f"Local cache delete error: {e}")
            success = False
        
        # 删除S3对象
        if self.s3_client:
            try:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            except Exception as e:
                logger.error(f"S3 delete error: {e}")
                success = False
        
        return success
    
    async def exists(self, key: str) -> bool:
        """检查缓存项是否存在"""
        # 检查本地缓存
        local_path = f"{self.local_cache_dir}/{self._safe_filename(key)}"
        if os.path.exists(local_path):
            return True
        
        # 检查S3
        if self.s3_client:
            try:
                self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
                return True
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    logger.error(f"S3 head error: {e}")
            except Exception as e:
                logger.error(f"S3 head error: {e}")
        
        return False
    
    async def clear(self):
        """清空缓存"""
        # 清空本地缓存
        try:
            import shutil
            shutil.rmtree(self.local_cache_dir)
            os.makedirs(self.local_cache_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Local cache clear error: {e}")
        
        # 清空S3（谨慎操作）
        if self.s3_client:
            try:
                # 列出所有对象
                response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
                
                if 'Contents' in response:
                    # 批量删除
                    objects = [{'Key': obj['Key']} for obj in response['Contents']]
                    self.s3_client.delete_objects(
                        Bucket=self.bucket_name,
                        Delete={'Objects': objects}
                    )
            except Exception as e:
                logger.error(f"S3 clear error: {e}")
    
    def _safe_filename(self, key: str) -> str:
        """生成安全的文件名"""
        # 使用MD5哈希生成安全的文件名
        return hashlib.md5(key.encode()).hexdigest()
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        self.stats.hit_rate = self.stats.hits / max(self.stats.total_requests, 1)
        
        # 计算存储使用量
        try:
            total_size = 0
            for filename in os.listdir(self.local_cache_dir):
                filepath = os.path.join(self.local_cache_dir, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
            self.stats.storage_usage = total_size
        except Exception as e:
            logger.error(f"Storage usage calculation error: {e}")
        
        return self.stats

class MultiLevelCacheManager:
    """多级缓存管理器"""
    
    def __init__(self, 
                 l1_max_size: int = 1000,
                 l1_ttl: int = 300,
                 redis_url: str = "redis://localhost:6379",
                 s3_bucket: str = "xiaozhi-cache"):
        
        self.l1_cache = L1MemoryCache(l1_max_size, l1_ttl)
        self.l2_cache = L2RedisCache(redis_url)
        self.l3_cache = L3StorageCache(s3_bucket)
        
        self.preloader = IntelligentPreloader(self)
        
        self.global_stats = CacheStats()
        self.cache_policies = {
            CacheType.VAD_RESULT: {'l1': True, 'l2': True, 'l3': False, 'ttl': 300},
            CacheType.ASR_RESULT: {'l1': True, 'l2': True, 'l3': True, 'ttl': 3600},
            CacheType.LLM_RESPONSE: {'l1': True, 'l2': True, 'l3': True, 'ttl': 7200},
            CacheType.TTS_AUDIO: {'l1': False, 'l2': True, 'l3': True, 'ttl': 86400},
            CacheType.MODEL_WEIGHTS: {'l1': False, 'l2': False, 'l3': True, 'ttl': None},
            CacheType.SESSION_CONTEXT: {'l1': True, 'l2': True, 'l3': False, 'ttl': 3600},
            CacheType.PRECOMPUTED_FEATURES: {'l1': True, 'l2': True, 'l3': True, 'ttl': 86400}
        }
    
    async def initialize(self):
        """初始化缓存管理器"""
        await self.l2_cache.initialize()
        await self.preloader.start()
        logger.info("Multi-level cache manager initialized")
    
    async def get(self, key: str, cache_type: CacheType = CacheType.SESSION_CONTEXT) -> Optional[Any]:
        """获取缓存项"""
        start_time = time.time()
        self.global_stats.total_requests += 1
        
        policy = self.cache_policies.get(cache_type, self.cache_policies[CacheType.SESSION_CONTEXT])
        
        # 记录访问模式
        self.preloader.record_access(key, cache_type)
        
        # L1缓存
        if policy['l1']:
            value = await self.l1_cache.get(key)
            if value is not None:
                self.global_stats.hits += 1
                self._update_response_time(time.time() - start_time)
                return value
        
        # L2缓存
        if policy['l2']:
            value = await self.l2_cache.get(key)
            if value is not None:
                # 回填到L1
                if policy['l1']:
                    await self.l1_cache.set(key, value, policy['ttl'])
                
                self.global_stats.hits += 1
                self._update_response_time(time.time() - start_time)
                return value
        
        # L3缓存
        if policy['l3']:
            value = await self.l3_cache.get(key)
            if value is not None:
                # 回填到L2和L1
                if policy['l2']:
                    await self.l2_cache.set(key, value, policy['ttl'])
                if policy['l1']:
                    await self.l1_cache.set(key, value, policy['ttl'])
                
                self.global_stats.hits += 1
                self._update_response_time(time.time() - start_time)
                return value
        
        self.global_stats.misses += 1
        self._update_response_time(time.time() - start_time)
        return None
    
    async def set(self, key: str, value: Any, cache_type: CacheType = CacheType.SESSION_CONTEXT, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        policy = self.cache_policies.get(cache_type, self.cache_policies[CacheType.SESSION_CONTEXT])
        
        if ttl is None:
            ttl = policy['ttl']
        
        # 创建缓存项
        cache_item = CacheItem(
            key=key,
            value=value,
            cache_type=cache_type,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(seconds=ttl) if ttl else None,
            size_bytes=len(pickle.dumps(value))
        )
        
        success = True
        
        # 根据策略设置到不同级别的缓存
        if policy['l1']:
            success &= await self.l1_cache.set(key, cache_item, ttl)
        
        if policy['l2']:
            success &= await self.l2_cache.set(key, cache_item, ttl)
        
        if policy['l3']:
            success &= await self.l3_cache.set(key, cache_item, ttl)
        
        return success
    
    async def delete(self, key: str, cache_type: CacheType = CacheType.SESSION_CONTEXT) -> bool:
        """删除缓存项"""
        policy = self.cache_policies.get(cache_type, self.cache_policies[CacheType.SESSION_CONTEXT])
        
        success = True
        
        if policy['l1']:
            success &= await self.l1_cache.delete(key)
        
        if policy['l2']:
            success &= await self.l2_cache.delete(key)
        
        if policy['l3']:
            success &= await self.l3_cache.delete(key)
        
        return success
    
    async def exists(self, key: str, cache_type: CacheType = CacheType.SESSION_CONTEXT) -> bool:
        """检查缓存项是否存在"""
        policy = self.cache_policies.get(cache_type, self.cache_policies[CacheType.SESSION_CONTEXT])
        
        if policy['l1'] and await self.l1_cache.exists(key):
            return True
        
        if policy['l2'] and await self.l2_cache.exists(key):
            return True
        
        if policy['l3'] and await self.l3_cache.exists(key):
            return True
        
        return False
    
    async def clear_all(self):
        """清空所有缓存"""
        await self.l1_cache.clear()
        await self.l2_cache.clear()
        await self.l3_cache.clear()
    
    async def predict_and_preload(self, context: Dict[str, Any]):
        """预测并预加载"""
        await self.preloader.predict_and_preload(context)
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()
        
        self.global_stats.hit_rate = self.global_stats.hits / max(self.global_stats.total_requests, 1)
        
        return {
            'global': asdict(self.global_stats),
            'l1_memory': asdict(l1_stats),
            'l2_redis': asdict(l2_stats),
            'l3_storage': asdict(l3_stats),
            'cache_policies': self.cache_policies,
            'preloader_patterns': len(self.preloader.access_patterns)
        }
    
    def _update_response_time(self, response_time: float):
        """更新响应时间统计"""
        if self.global_stats.total_requests == 1:
            self.global_stats.avg_response_time = response_time
        else:
            # 计算移动平均
            total = self.global_stats.total_requests
            current_avg = self.global_stats.avg_response_time
            self.global_stats.avg_response_time = (current_avg * (total - 1) + response_time) / total

# 使用示例
async def main():
    """主函数 - 使用示例"""
    # 创建缓存管理器
    cache_manager = MultiLevelCacheManager(
        l1_max_size=2000,
        l1_ttl=600,
        redis_url="redis://localhost:6379",
        s3_bucket="xiaozhi-cache"
    )
    
    # 初始化
    await cache_manager.initialize()
    
    # 设置不同类型的缓存
    await cache_manager.set("vad_result_123", {"is_speech": True, "confidence": 0.95}, CacheType.VAD_RESULT)
    await cache_manager.set("asr_result_456", {"text": "你好世界", "confidence": 0.98}, CacheType.ASR_RESULT)
    await cache_manager.set("llm_response_789", {"response": "很高兴为您服务！", "tokens": 50}, CacheType.LLM_RESPONSE)
    
    # 获取缓存
    vad_result = await cache_manager.get("vad_result_123", CacheType.VAD_RESULT)
    print(f"VAD Result: {vad_result}")
    
    # 预测和预加载
    context = {"user_id": "user123", "session_type": "conversation", "time_of_day": "morning"}
    await cache_manager.predict_and_preload(context)
    
    # 获取统计信息
    stats = cache_manager.get_comprehensive_stats()
    print(f"Cache Stats: {json.dumps(stats, indent=2, default=str)}")

if __name__ == "__main__":
    asyncio.run(main())