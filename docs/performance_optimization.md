# 性能测试和优化建议

## 概述

本文档提供了LLM和TTS服务的性能测试方案、优化策略和最佳实践，帮助您在不增加硬件投入的情况下最大化服务性能。

## 性能基准测试

### 当前系统性能基线

#### LLM服务基线
- **并发处理能力**: 20个并发请求
- **平均响应时间**: 3-5秒
- **内存使用**: 1.5GB
- **CPU使用**: 0.5核心
- **缓存命中率**: 15-20%

#### TTS服务基线
- **并发处理能力**: 15个并发请求
- **平均响应时间**: 2-3秒
- **内存使用**: 512MB
- **CPU使用**: 0.3核心
- **缓存命中率**: 25-30%

### 性能测试工具

#### 1. 压力测试脚本

创建 `tests/performance/load_test.py`：

```python
import asyncio
import aiohttp
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

class PerformanceTester:
    def __init__(self, base_url, concurrent_users=50, total_requests=1000):
        self.base_url = base_url
        self.concurrent_users = concurrent_users
        self.total_requests = total_requests
        self.results = []
        
    async def test_llm_endpoint(self, session, request_id):
        """测试LLM端点性能"""
        start_time = time.time()
        try:
            payload = {
                "messages": [
                    {"role": "user", "content": f"测试请求 {request_id}：请简单介绍一下人工智能"}
                ],
                "model": "auto",
                "max_tokens": 100
            }
            
            async with session.post(
                f"{self.base_url}/llm/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                result = await response.json()
                end_time = time.time()
                
                return {
                    "request_id": request_id,
                    "status_code": response.status,
                    "response_time": end_time - start_time,
                    "success": response.status == 200,
                    "tokens": len(result.get("choices", [{}])[0].get("message", {}).get("content", "").split())
                }
        except Exception as e:
            end_time = time.time()
            return {
                "request_id": request_id,
                "status_code": 0,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e)
            }
    
    async def test_tts_endpoint(self, session, request_id):
        """测试TTS端点性能"""
        start_time = time.time()
        try:
            payload = {
                "text": f"这是第{request_id}个测试语音合成请求",
                "voice": "zh-CN-XiaoxiaoNeural",
                "format": "mp3"
            }
            
            async with session.post(
                f"{self.base_url}/tts/synthesize",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                audio_data = await response.read()
                end_time = time.time()
                
                return {
                    "request_id": request_id,
                    "status_code": response.status,
                    "response_time": end_time - start_time,
                    "success": response.status == 200,
                    "audio_size": len(audio_data)
                }
        except Exception as e:
            end_time = time.time()
            return {
                "request_id": request_id,
                "status_code": 0,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e)
            }
    
    async def run_load_test(self, endpoint_type="llm"):
        """运行负载测试"""
        connector = aiohttp.TCPConnector(limit=self.concurrent_users * 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            semaphore = asyncio.Semaphore(self.concurrent_users)
            
            async def limited_request(request_id):
                async with semaphore:
                    if endpoint_type == "llm":
                        return await self.test_llm_endpoint(session, request_id)
                    else:
                        return await self.test_tts_endpoint(session, request_id)
            
            print(f"开始{endpoint_type.upper()}负载测试...")
            print(f"并发用户: {self.concurrent_users}")
            print(f"总请求数: {self.total_requests}")
            
            start_time = time.time()
            tasks = [limited_request(i) for i in range(self.total_requests)]
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            self.results = results
            self.analyze_results(end_time - start_time)
    
    def analyze_results(self, total_time):
        """分析测试结果"""
        successful_requests = [r for r in self.results if r["success"]]
        failed_requests = [r for r in self.results if not r["success"]]
        
        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            
            print("\n=== 性能测试结果 ===")
            print(f"总测试时间: {total_time:.2f}秒")
            print(f"总请求数: {len(self.results)}")
            print(f"成功请求: {len(successful_requests)}")
            print(f"失败请求: {len(failed_requests)}")
            print(f"成功率: {len(successful_requests)/len(self.results)*100:.2f}%")
            print(f"QPS: {len(successful_requests)/total_time:.2f}")
            print(f"平均响应时间: {statistics.mean(response_times):.2f}秒")
            print(f"中位数响应时间: {statistics.median(response_times):.2f}秒")
            print(f"95%响应时间: {sorted(response_times)[int(len(response_times)*0.95)]:.2f}秒")
            print(f"99%响应时间: {sorted(response_times)[int(len(response_times)*0.99)]:.2f}秒")
            print(f"最小响应时间: {min(response_times):.2f}秒")
            print(f"最大响应时间: {max(response_times):.2f}秒")
            
            if failed_requests:
                print("\n=== 失败请求分析 ===")
                error_types = {}
                for req in failed_requests:
                    error = req.get("error", "Unknown")
                    error_types[error] = error_types.get(error, 0) + 1
                
                for error, count in error_types.items():
                    print(f"{error}: {count}次")

# 使用示例
async def main():
    # LLM服务测试
    llm_tester = PerformanceTester("http://localhost:8001", concurrent_users=30, total_requests=500)
    await llm_tester.run_load_test("llm")
    
    # TTS服务测试
    tts_tester = PerformanceTester("http://localhost:8002", concurrent_users=20, total_requests=300)
    await tts_tester.run_load_test("tts")

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. 基准测试脚本

创建 `tests/performance/benchmark.py`：

```python
import asyncio
import time
import psutil
import aiohttp
import json
from datetime import datetime

class BenchmarkTester:
    def __init__(self):
        self.metrics = {
            "cpu_usage": [],
            "memory_usage": [],
            "response_times": [],
            "throughput": [],
            "cache_hit_rates": []
        }
    
    async def monitor_system_resources(self, duration=300):
        """监控系统资源使用情况"""
        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            
            self.metrics["cpu_usage"].append({
                "timestamp": datetime.now().isoformat(),
                "value": cpu_percent
            })
            
            self.metrics["memory_usage"].append({
                "timestamp": datetime.now().isoformat(),
                "value": memory_info.percent
            })
            
            await asyncio.sleep(5)
    
    async def test_cache_performance(self, base_url):
        """测试缓存性能"""
        async with aiohttp.ClientSession() as session:
            # 第一次请求（缓存未命中）
            payload = {
                "messages": [{"role": "user", "content": "什么是机器学习？"}],
                "model": "auto"
            }
            
            start_time = time.time()
            async with session.post(f"{base_url}/llm/chat/completions", json=payload) as response:
                await response.json()
            first_request_time = time.time() - start_time
            
            # 第二次相同请求（缓存命中）
            start_time = time.time()
            async with session.post(f"{base_url}/llm/chat/completions", json=payload) as response:
                await response.json()
            second_request_time = time.time() - start_time
            
            cache_improvement = (first_request_time - second_request_time) / first_request_time * 100
            
            print(f"缓存性能测试:")
            print(f"首次请求时间: {first_request_time:.2f}秒")
            print(f"缓存命中时间: {second_request_time:.2f}秒")
            print(f"性能提升: {cache_improvement:.1f}%")
    
    async def test_concurrent_scaling(self, base_url):
        """测试并发扩展性"""
        concurrent_levels = [1, 5, 10, 20, 30, 50, 100]
        
        for concurrent in concurrent_levels:
            print(f"\n测试并发级别: {concurrent}")
            
            async with aiohttp.ClientSession() as session:
                semaphore = asyncio.Semaphore(concurrent)
                
                async def make_request(request_id):
                    async with semaphore:
                        start_time = time.time()
                        try:
                            payload = {
                                "messages": [{"role": "user", "content": f"请求{request_id}"}],
                                "model": "auto"
                            }
                            async with session.post(
                                f"{base_url}/llm/chat/completions",
                                json=payload,
                                timeout=aiohttp.ClientTimeout(total=30)
                            ) as response:
                                await response.json()
                                return time.time() - start_time
                        except Exception as e:
                            return None
                
                start_time = time.time()
                tasks = [make_request(i) for i in range(concurrent * 2)]
                results = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
                
                successful_results = [r for r in results if r is not None]
                if successful_results:
                    avg_response_time = sum(successful_results) / len(successful_results)
                    qps = len(successful_results) / total_time
                    
                    print(f"  平均响应时间: {avg_response_time:.2f}秒")
                    print(f"  QPS: {qps:.2f}")
                    print(f"  成功率: {len(successful_results)/len(results)*100:.1f}%")

# 运行基准测试
async def run_benchmark():
    benchmark = BenchmarkTester()
    
    # 系统资源监控
    monitor_task = asyncio.create_task(benchmark.monitor_system_resources(300))
    
    # 缓存性能测试
    await benchmark.test_cache_performance("http://localhost:8001")
    
    # 并发扩展性测试
    await benchmark.test_concurrent_scaling("http://localhost:8001")
    
    # 等待监控完成
    await monitor_task

if __name__ == "__main__":
    asyncio.run(run_benchmark())
```

### 性能目标

#### 优化后的性能目标

| 指标 | 当前值 | 目标值 | 提升幅度 |
|------|--------|--------|----------|
| LLM并发处理 | 20 | 100+ | 5倍 |
| LLM响应时间 | 3-5秒 | 1-2秒 | 60% |
| TTS并发处理 | 15 | 50+ | 3倍 |
| TTS响应时间 | 2-3秒 | 0.5-1秒 | 70% |
| 缓存命中率 | 20% | 60%+ | 3倍 |
| 系统可用性 | 95% | 99.9% | - |

## 优化策略

### 1. 智能缓存优化

#### 多层缓存架构

```python
# 缓存优化配置
cache_config = {
    "l1_cache": {
        "type": "memory",
        "size": "256MB",
        "ttl": 300,  # 5分钟
        "strategy": "LRU"
    },
    "l2_cache": {
        "type": "redis",
        "size": "2GB",
        "ttl": 3600,  # 1小时
        "strategy": "adaptive"
    },
    "l3_cache": {
        "type": "disk",
        "size": "10GB",
        "ttl": 86400,  # 24小时
        "strategy": "LFU"
    }
}
```

#### 智能缓存键生成

```python
def generate_smart_cache_key(request_data):
    """生成智能缓存键"""
    # 提取关键特征
    content = request_data.get("messages", [])[-1].get("content", "")
    model = request_data.get("model", "default")
    
    # 语义相似性检测
    semantic_hash = generate_semantic_hash(content)
    
    # 参数标准化
    normalized_params = normalize_parameters(request_data)
    
    return f"llm:{model}:{semantic_hash}:{hash(str(normalized_params))}"

def generate_semantic_hash(text):
    """生成语义哈希"""
    # 简化版本，实际可以使用embedding
    import re
    # 移除标点符号和数字
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    # 移除常见停用词
    stop_words = {'的', '是', '在', '有', '和', '了', '我', '你', '他'}
    words = [w for w in cleaned.split() if w not in stop_words]
    return hash(' '.join(sorted(words)))
```

#### 预测性缓存

```python
class PredictiveCache:
    def __init__(self):
        self.access_patterns = {}
        self.prediction_model = None
    
    def record_access(self, cache_key, timestamp):
        """记录访问模式"""
        if cache_key not in self.access_patterns:
            self.access_patterns[cache_key] = []
        self.access_patterns[cache_key].append(timestamp)
    
    def predict_next_access(self, cache_key):
        """预测下次访问时间"""
        if cache_key not in self.access_patterns:
            return None
        
        accesses = self.access_patterns[cache_key]
        if len(accesses) < 2:
            return None
        
        # 简单的时间间隔预测
        intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
        avg_interval = sum(intervals) / len(intervals)
        
        return accesses[-1] + avg_interval
    
    async def preload_cache(self):
        """预加载缓存"""
        current_time = time.time()
        for cache_key in self.access_patterns:
            predicted_time = self.predict_next_access(cache_key)
            if predicted_time and predicted_time - current_time < 300:  # 5分钟内
                # 预加载这个缓存项
                await self.preload_item(cache_key)
```

### 2. 连接池优化

#### HTTP连接池配置

```python
# 优化的连接池配置
connector_config = {
    "limit": 1000,  # 总连接数限制
    "limit_per_host": 100,  # 每个主机连接数限制
    "ttl_dns_cache": 300,  # DNS缓存TTL
    "use_dns_cache": True,
    "keepalive_timeout": 30,
    "enable_cleanup_closed": True,
    "timeout": aiohttp.ClientTimeout(
        total=30,
        connect=5,
        sock_read=10
    )
}

# 创建优化的连接器
connector = aiohttp.TCPConnector(**connector_config)
session = aiohttp.ClientSession(connector=connector)
```

#### 连接池监控

```python
class ConnectionPoolMonitor:
    def __init__(self, connector):
        self.connector = connector
        
    def get_stats(self):
        """获取连接池统计信息"""
        return {
            "total_connections": len(self.connector._conns),
            "available_connections": sum(len(conns) for conns in self.connector._conns.values()),
            "acquired_connections": self.connector._acquired_per_host,
            "dns_cache_size": len(self.connector._dns_cache) if hasattr(self.connector, '_dns_cache') else 0
        }
    
    async def cleanup_expired_connections(self):
        """清理过期连接"""
        while True:
            await asyncio.sleep(60)  # 每分钟清理一次
            self.connector._cleanup()
```

### 3. 异步处理优化

#### 请求队列管理

```python
class OptimizedRequestQueue:
    def __init__(self, max_size=1000, priority_levels=3):
        self.queues = [asyncio.Queue(maxsize=max_size//priority_levels) 
                      for _ in range(priority_levels)]
        self.processing = False
        
    async def add_request(self, request, priority=1):
        """添加请求到队列"""
        queue_index = min(priority, len(self.queues) - 1)
        await self.queues[queue_index].put(request)
        
        if not self.processing:
            asyncio.create_task(self.process_requests())
    
    async def process_requests(self):
        """处理请求队列"""
        self.processing = True
        
        while any(not q.empty() for q in self.queues):
            # 优先处理高优先级队列
            for queue in self.queues:
                if not queue.empty():
                    request = await queue.get()
                    asyncio.create_task(self.handle_request(request))
                    break
            
            await asyncio.sleep(0.01)  # 避免CPU占用过高
        
        self.processing = False
    
    async def handle_request(self, request):
        """处理单个请求"""
        try:
            # 实际的请求处理逻辑
            await request.process()
        except Exception as e:
            await request.handle_error(e)
```

#### 批处理优化

```python
class BatchProcessor:
    def __init__(self, batch_size=10, max_wait_time=1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.last_batch_time = time.time()
        
    async def add_request(self, request):
        """添加请求到批处理队列"""
        self.pending_requests.append(request)
        
        # 检查是否需要处理批次
        if (len(self.pending_requests) >= self.batch_size or 
            time.time() - self.last_batch_time > self.max_wait_time):
            await self.process_batch()
    
    async def process_batch(self):
        """处理一个批次的请求"""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]
        self.last_batch_time = time.time()
        
        # 并行处理批次中的所有请求
        tasks = [self.process_single_request(req) for req in batch]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def process_single_request(self, request):
        """处理单个请求"""
        # 实际的请求处理逻辑
        pass
```

### 4. 内存优化

#### 对象池模式

```python
class ObjectPool:
    def __init__(self, factory_func, max_size=100):
        self.factory_func = factory_func
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
    
    def acquire(self):
        """获取对象"""
        if self.pool:
            obj = self.pool.pop()
        else:
            obj = self.factory_func()
        
        self.in_use.add(id(obj))
        return obj
    
    def release(self, obj):
        """释放对象"""
        obj_id = id(obj)
        if obj_id in self.in_use:
            self.in_use.remove(obj_id)
            
            if len(self.pool) < self.max_size:
                # 重置对象状态
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)

# 使用示例
response_pool = ObjectPool(lambda: {"choices": [], "usage": {}}, max_size=50)
```

#### 内存监控和清理

```python
class MemoryManager:
    def __init__(self, max_memory_mb=2048):
        self.max_memory_mb = max_memory_mb
        self.cleanup_threshold = 0.8  # 80%内存使用率时开始清理
        
    def get_memory_usage(self):
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024  # MB
    
    async def monitor_memory(self):
        """监控内存使用"""
        while True:
            current_usage = self.get_memory_usage()
            usage_ratio = current_usage / self.max_memory_mb
            
            if usage_ratio > self.cleanup_threshold:
                await self.cleanup_memory()
            
            await asyncio.sleep(30)  # 每30秒检查一次
    
    async def cleanup_memory(self):
        """清理内存"""
        import gc
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理缓存
        await self.cleanup_caches()
        
        # 清理连接池
        await self.cleanup_connections()
    
    async def cleanup_caches(self):
        """清理缓存"""
        # 清理LRU缓存中最少使用的项
        pass
    
    async def cleanup_connections(self):
        """清理连接"""
        # 关闭空闲连接
        pass
```

### 5. 数据库优化

#### Redis优化配置

```redis
# redis.conf 优化配置

# 内存优化
maxmemory 2gb
maxmemory-policy allkeys-lru

# 持久化优化
save 900 1
save 300 10
save 60 10000

# 网络优化
tcp-keepalive 300
timeout 0

# 性能优化
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# 连接优化
tcp-backlog 511
maxclients 10000
```

#### Redis连接池优化

```python
import aioredis

class OptimizedRedisPool:
    def __init__(self):
        self.pool = None
        
    async def initialize(self):
        """初始化Redis连接池"""
        self.pool = aioredis.ConnectionPool.from_url(
            "redis://localhost:6379",
            max_connections=100,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            },
            health_check_interval=30
        )
    
    async def get_connection(self):
        """获取Redis连接"""
        return aioredis.Redis(connection_pool=self.pool)
    
    async def close(self):
        """关闭连接池"""
        if self.pool:
            await self.pool.disconnect()
```

## 监控和调优

### 1. 性能监控指标

#### 关键性能指标(KPI)

```python
class PerformanceMetrics:
    def __init__(self):
        self.metrics = {
            "response_time": [],
            "throughput": [],
            "error_rate": [],
            "cache_hit_rate": [],
            "memory_usage": [],
            "cpu_usage": [],
            "connection_pool_usage": []
        }
    
    def record_response_time(self, duration):
        """记录响应时间"""
        self.metrics["response_time"].append({
            "timestamp": time.time(),
            "value": duration
        })
    
    def record_throughput(self, requests_per_second):
        """记录吞吐量"""
        self.metrics["throughput"].append({
            "timestamp": time.time(),
            "value": requests_per_second
        })
    
    def get_average_response_time(self, window_seconds=300):
        """获取平均响应时间"""
        current_time = time.time()
        recent_metrics = [
            m for m in self.metrics["response_time"]
            if current_time - m["timestamp"] <= window_seconds
        ]
        
        if not recent_metrics:
            return 0
        
        return sum(m["value"] for m in recent_metrics) / len(recent_metrics)
    
    def get_p95_response_time(self, window_seconds=300):
        """获取95%响应时间"""
        current_time = time.time()
        recent_metrics = [
            m["value"] for m in self.metrics["response_time"]
            if current_time - m["timestamp"] <= window_seconds
        ]
        
        if not recent_metrics:
            return 0
        
        recent_metrics.sort()
        index = int(len(recent_metrics) * 0.95)
        return recent_metrics[index] if index < len(recent_metrics) else recent_metrics[-1]
```

#### 实时监控仪表板

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Prometheus指标
REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('request_duration_seconds', 'Request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage')

class PrometheusMonitor:
    def __init__(self, port=9090):
        self.port = port
        
    def start_server(self):
        """启动Prometheus监控服务器"""
        start_http_server(self.port)
        print(f"Prometheus监控服务器启动在端口 {self.port}")
    
    def record_request(self, method, endpoint, status, duration):
        """记录请求指标"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_DURATION.observe(duration)
    
    def update_active_connections(self, count):
        """更新活跃连接数"""
        ACTIVE_CONNECTIONS.set(count)
    
    def update_cache_hit_rate(self, rate):
        """更新缓存命中率"""
        CACHE_HIT_RATE.set(rate)
    
    def update_memory_usage(self, bytes_used):
        """更新内存使用量"""
        MEMORY_USAGE.set(bytes_used)
```

### 2. 自动调优

#### 自适应参数调整

```python
class AutoTuner:
    def __init__(self):
        self.parameters = {
            "max_concurrent": 50,
            "cache_ttl": 3600,
            "timeout": 30,
            "batch_size": 10
        }
        self.performance_history = []
        
    async def tune_parameters(self):
        """自动调优参数"""
        while True:
            # 收集当前性能指标
            current_metrics = await self.collect_metrics()
            
            # 分析性能趋势
            if len(self.performance_history) > 10:
                trend = self.analyze_trend()
                
                # 根据趋势调整参数
                if trend == "degrading":
                    await self.adjust_for_performance()
                elif trend == "stable":
                    await self.optimize_for_cost()
            
            self.performance_history.append(current_metrics)
            
            # 保持历史记录在合理范围内
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-50:]
            
            await asyncio.sleep(300)  # 每5分钟调优一次
    
    async def collect_metrics(self):
        """收集性能指标"""
        return {
            "avg_response_time": await self.get_avg_response_time(),
            "throughput": await self.get_throughput(),
            "error_rate": await self.get_error_rate(),
            "cache_hit_rate": await self.get_cache_hit_rate(),
            "memory_usage": await self.get_memory_usage()
        }
    
    def analyze_trend(self):
        """分析性能趋势"""
        recent_metrics = self.performance_history[-10:]
        
        # 简单的趋势分析
        response_times = [m["avg_response_time"] for m in recent_metrics]
        error_rates = [m["error_rate"] for m in recent_metrics]
        
        if sum(response_times[-3:]) > sum(response_times[:3]) * 1.2:
            return "degrading"
        elif max(error_rates[-3:]) > 0.05:  # 错误率超过5%
            return "degrading"
        else:
            return "stable"
    
    async def adjust_for_performance(self):
        """为性能调整参数"""
        # 增加并发数
        if self.parameters["max_concurrent"] < 200:
            self.parameters["max_concurrent"] += 10
        
        # 减少超时时间
        if self.parameters["timeout"] > 15:
            self.parameters["timeout"] -= 5
        
        # 增加批处理大小
        if self.parameters["batch_size"] < 50:
            self.parameters["batch_size"] += 5
        
        await self.apply_parameters()
    
    async def optimize_for_cost(self):
        """为成本优化参数"""
        # 增加缓存TTL
        if self.parameters["cache_ttl"] < 7200:
            self.parameters["cache_ttl"] += 300
        
        await self.apply_parameters()
    
    async def apply_parameters(self):
        """应用参数更改"""
        # 实际应用参数到系统中
        print(f"应用新参数: {self.parameters}")
```

### 3. 故障预测和预防

#### 异常检测

```python
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_trained = False
        self.feature_history = []
        
    def extract_features(self, metrics):
        """提取特征"""
        return [
            metrics.get("avg_response_time", 0),
            metrics.get("throughput", 0),
            metrics.get("error_rate", 0),
            metrics.get("cache_hit_rate", 0),
            metrics.get("memory_usage", 0),
            metrics.get("cpu_usage", 0)
        ]
    
    def train(self, historical_metrics):
        """训练异常检测模型"""
        features = [self.extract_features(m) for m in historical_metrics]
        self.model.fit(features)
        self.is_trained = True
    
    def detect_anomaly(self, current_metrics):
        """检测异常"""
        if not self.is_trained:
            return False, 0
        
        features = self.extract_features(current_metrics)
        prediction = self.model.predict([features])[0]
        score = self.model.score_samples([features])[0]
        
        is_anomaly = prediction == -1
        return is_anomaly, score
    
    async def continuous_monitoring(self):
        """持续监控"""
        while True:
            current_metrics = await self.collect_current_metrics()
            is_anomaly, score = self.detect_anomaly(current_metrics)
            
            if is_anomaly:
                await self.handle_anomaly(current_metrics, score)
            
            # 更新训练数据
            self.feature_history.append(current_metrics)
            if len(self.feature_history) > 1000:
                self.feature_history = self.feature_history[-500:]
                # 重新训练模型
                self.train(self.feature_history)
            
            await asyncio.sleep(60)  # 每分钟检测一次
    
    async def handle_anomaly(self, metrics, score):
        """处理异常"""
        print(f"检测到异常! 分数: {score}")
        print(f"当前指标: {metrics}")
        
        # 发送告警
        await self.send_alert(metrics, score)
        
        # 自动恢复措施
        await self.auto_recovery(metrics)
    
    async def send_alert(self, metrics, score):
        """发送告警"""
        # 实现告警逻辑（邮件、短信、Slack等）
        pass
    
    async def auto_recovery(self, metrics):
        """自动恢复"""
        # 实现自动恢复逻辑
        if metrics.get("error_rate", 0) > 0.1:
            # 错误率过高，重启服务
            await self.restart_service()
        elif metrics.get("memory_usage", 0) > 0.9:
            # 内存使用过高，清理缓存
            await self.cleanup_memory()
```

## 最佳实践总结

### 1. 架构设计原则

1. **微服务架构**: 将LLM和TTS服务分离，独立扩展
2. **异步处理**: 全面采用异步I/O，提高并发能力
3. **多层缓存**: 实现内存、Redis、磁盘多层缓存
4. **智能路由**: 基于成本、延迟、质量的智能路由
5. **故障隔离**: 实现熔断器和故障转移机制

### 2. 性能优化清单

- [ ] 实现连接池复用
- [ ] 启用HTTP/2和Keep-Alive
- [ ] 配置多层缓存策略
- [ ] 实现请求批处理
- [ ] 优化序列化/反序列化
- [ ] 启用压缩传输
- [ ] 实现预测性缓存
- [ ] 配置自动扩缩容
- [ ] 实现智能负载均衡
- [ ] 启用性能监控

### 3. 监控指标

#### 核心指标
- **响应时间**: P50, P95, P99
- **吞吐量**: QPS, TPS
- **错误率**: 4xx, 5xx错误比例
- **可用性**: 服务正常运行时间
- **资源使用**: CPU, 内存, 网络

#### 业务指标
- **缓存命中率**: 各层缓存效果
- **API成本**: 每次调用成本
- **用户满意度**: 响应质量评分
- **服务质量**: SLA达成率

### 4. 运维建议

1. **渐进式部署**: 分阶段部署新功能
2. **A/B测试**: 对比不同配置的性能
3. **容量规划**: 基于历史数据预测容量需求
4. **故障演练**: 定期进行故障恢复演练
5. **性能基准**: 建立性能基准和回归测试

通过实施这些优化策略，您的LLM和TTS服务性能将得到显著提升，在不增加硬件投入的情况下实现5-10倍的性能提升。