# ⚖️ 负载均衡优化指南 - 4核8GB硬件最大化利用

## 🎯 **目标**
在4核8GB硬件限制下，通过智能负载均衡策略，将设备支撑能力从10台提升到30-50台，改善潜力评估为**300-400%**。

---

## 📊 **当前性能瓶颈分析**

### **资源使用现状**

| 服务 | CPU使用 | 内存使用 | 瓶颈类型 | 优化潜力 |
|------|---------|----------|----------|----------|
| **xiaozhi-esp32-server** | 0.11% | 1.8GB/4GB | 内存密集 | ⭐⭐⭐⭐⭐ |
| **xiaozhi-esp32-server-web** | 0.26% | 407MB/768MB | CPU密集 | ⭐⭐⭐⭐ |
| **xiaozhi-esp32-server-db** | 3.54% | 489MB/512MB | 内存瓶颈 | ⭐⭐⭐⭐⭐ |
| **xiaozhi-esp32-server-redis** | 0.68% | 20MB/256MB | 轻负载 | ⭐⭐⭐ |

### **负载分布问题**
```yaml
current_issues:
  # 主要问题
  primary_bottlenecks:
    - "数据库内存使用率95% (489MB/512MB)"
    - "主服务内存使用率45% (1.8GB/4GB)"
    - "LLM/TTS远程API延迟1.5-3秒"
    - "单点故障风险高"
    
  # 资源浪费
  resource_waste:
    - "CPU总使用率 < 5%，大量闲置"
    - "Redis内存使用率仅8%"
    - "网络带宽利用率低"
    - "缺乏服务间负载分担"
    
  # 扩展性限制
  scalability_limits:
    - "所有服务运行在单机"
    - "缺乏水平扩展能力"
    - "无服务降级机制"
    - "缺乏智能路由"
```

---

## 🏗️ **多层负载均衡架构**

### **架构设计图**

```
                    ┌─────────────────┐
                    │   Nginx Proxy   │ ← 入口负载均衡
                    │   (反向代理)     │
                    └─────────┬───────┘
                              │
                    ┌─────────▼───────┐
                    │  API Gateway    │ ← 智能路由层
                    │  (路由分发)     │
                    └─────────┬───────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐         ┌─────▼─────┐         ┌─────▼─────┐
   │ ASR服务  │         │ LLM服务   │         │ TTS服务   │
   │ 实例1-2  │         │ 实例1-3   │         │ 实例1-2   │
   └─────────┘         └───────────┘         └───────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼───────┐
                    │   共享存储层     │
                    │ Redis + MySQL   │
                    └─────────────────┘
```

### **负载均衡策略配置**

```yaml
# load_balancing_config.yaml
load_balancing:
  # 全局配置
  global_settings:
    algorithm: "weighted_round_robin"    # 加权轮询
    health_check_interval: 30           # 健康检查间隔
    failure_threshold: 3                # 失败阈值
    recovery_threshold: 2               # 恢复阈值
    
  # Nginx前端负载均衡
  nginx_proxy:
    upstream_servers:
      - server: "127.0.0.1:8000"
        weight: 3                       # 主服务权重3
        max_fails: 2
        fail_timeout: 30
        
      - server: "127.0.0.1:8001"
        weight: 2                       # 备用服务权重2
        max_fails: 2
        fail_timeout: 30
        
    # 连接池配置
    connection_pool:
      keepalive: 32                     # 保持连接数
      keepalive_requests: 100           # 每连接最大请求数
      keepalive_timeout: 60             # 连接超时时间
      
  # API网关路由
  api_gateway:
    # ASR服务负载均衡
    asr_services:
      algorithm: "least_connections"    # 最少连接数
      instances:
        - endpoint: "http://localhost:8100"
          weight: 2
          max_concurrent: 10
          
        - endpoint: "http://localhost:8101"
          weight: 1
          max_concurrent: 5
          
    # LLM服务负载均衡
    llm_services:
      algorithm: "response_time"        # 响应时间优先
      instances:
        - endpoint: "http://localhost:11434"  # 本地LLM
          weight: 4
          priority: "high"
          max_concurrent: 8
          
        - endpoint: "https://dashscope.aliyuncs.com"  # 远程LLM
          weight: 1
          priority: "low"
          max_concurrent: 3
          
    # TTS服务负载均衡
    tts_services:
      algorithm: "weighted_round_robin"
      instances:
        - endpoint: "http://localhost:5000"   # 本地Edge-TTS
          weight: 4
          priority: "high"
          max_concurrent: 15
          
        - endpoint: "https://tts-api.xfyun.cn"  # 远程TTS
          weight: 1
          priority: "low"
          max_concurrent: 5
```

---

## 🔄 **智能路由算法**

### **多维度路由决策**

```python
# intelligent_router.py
import asyncio
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class LoadBalanceAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESPONSE_TIME = "response_time"
    RESOURCE_BASED = "resource_based"

@dataclass
class ServiceInstance:
    endpoint: str
    weight: int
    priority: str
    max_concurrent: int
    current_connections: int = 0
    avg_response_time: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    is_healthy: bool = True

class IntelligentRouter:
    def __init__(self):
        self.instances: Dict[str, List[ServiceInstance]] = {}
        self.request_counts: Dict[str, int] = {}
        self.response_times: Dict[str, List[float]] = {}
        
    def register_service(self, service_name: str, instances: List[ServiceInstance]):
        """注册服务实例"""
        self.instances[service_name] = instances
        self.request_counts[service_name] = 0
        self.response_times[service_name] = []
        
    async def route_request(
        self, 
        service_name: str, 
        request_context: Dict,
        algorithm: LoadBalanceAlgorithm = LoadBalanceAlgorithm.WEIGHTED_ROUND_ROBIN
    ) -> Optional[ServiceInstance]:
        """智能路由请求"""
        
        instances = self.instances.get(service_name, [])
        if not instances:
            return None
            
        # 过滤健康实例
        healthy_instances = [inst for inst in instances if inst.is_healthy]
        if not healthy_instances:
            return None
            
        # 根据算法选择实例
        if algorithm == LoadBalanceAlgorithm.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(service_name, healthy_instances)
        elif algorithm == LoadBalanceAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections(healthy_instances)
        elif algorithm == LoadBalanceAlgorithm.RESPONSE_TIME:
            return self._response_time_based(healthy_instances)
        elif algorithm == LoadBalanceAlgorithm.RESOURCE_BASED:
            return self._resource_based(healthy_instances, request_context)
        else:
            return self._round_robin(service_name, healthy_instances)
            
    def _weighted_round_robin(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """加权轮询算法"""
        total_weight = sum(inst.weight for inst in instances)
        request_count = self.request_counts[service_name]
        
        # 计算当前应该选择的实例
        current_weight = request_count % total_weight
        cumulative_weight = 0
        
        for instance in instances:
            cumulative_weight += instance.weight
            if current_weight < cumulative_weight:
                self.request_counts[service_name] += 1
                return instance
                
        return instances[0]
        
    def _least_connections(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """最少连接数算法"""
        return min(instances, key=lambda x: x.current_connections)
        
    def _response_time_based(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """响应时间优先算法"""
        # 优先选择响应时间最短的实例
        return min(instances, key=lambda x: x.avg_response_time)
        
    def _resource_based(self, instances: List[ServiceInstance], context: Dict) -> ServiceInstance:
        """基于资源使用率的智能路由"""
        
        def calculate_score(instance: ServiceInstance) -> float:
            """计算实例得分（越低越好）"""
            # 基础得分
            score = 0.0
            
            # CPU使用率权重 (30%)
            score += instance.cpu_usage * 0.3
            
            # 内存使用率权重 (25%)
            score += instance.memory_usage * 0.25
            
            # 当前连接数权重 (20%)
            connection_ratio = instance.current_connections / instance.max_concurrent
            score += connection_ratio * 0.2
            
            # 响应时间权重 (15%)
            normalized_response_time = min(instance.avg_response_time / 1000, 1.0)
            score += normalized_response_time * 0.15
            
            # 错误率权重 (10%)
            score += instance.error_rate * 0.1
            
            # 优先级调整
            if instance.priority == "high":
                score *= 0.8
            elif instance.priority == "low":
                score *= 1.2
                
            return score
            
        # 选择得分最低的实例
        return min(instances, key=calculate_score)
        
    async def update_instance_metrics(self, service_name: str, endpoint: str, metrics: Dict):
        """更新实例指标"""
        instances = self.instances.get(service_name, [])
        for instance in instances:
            if instance.endpoint == endpoint:
                instance.avg_response_time = metrics.get('response_time', instance.avg_response_time)
                instance.error_rate = metrics.get('error_rate', instance.error_rate)
                instance.cpu_usage = metrics.get('cpu_usage', instance.cpu_usage)
                instance.memory_usage = metrics.get('memory_usage', instance.memory_usage)
                instance.current_connections = metrics.get('connections', instance.current_connections)
                break
                
    async def health_check(self):
        """健康检查"""
        for service_name, instances in self.instances.items():
            for instance in instances:
                try:
                    # 执行健康检查
                    start_time = time.time()
                    # 这里应该实际调用健康检查接口
                    # response = await self.http_client.get(f"{instance.endpoint}/health")
                    response_time = (time.time() - start_time) * 1000
                    
                    instance.is_healthy = True  # response.status_code == 200
                    instance.avg_response_time = response_time
                    
                except Exception as e:
                    instance.is_healthy = False
                    print(f"健康检查失败 {instance.endpoint}: {e}")
```

### **动态权重调整**

```python
# dynamic_weight_adjuster.py
class DynamicWeightAdjuster:
    def __init__(self, router: IntelligentRouter):
        self.router = router
        self.adjustment_interval = 60  # 60秒调整一次
        
    async def start_adjustment_loop(self):
        """启动动态权重调整循环"""
        while True:
            await self.adjust_weights()
            await asyncio.sleep(self.adjustment_interval)
            
    async def adjust_weights(self):
        """动态调整权重"""
        for service_name, instances in self.router.instances.items():
            await self._adjust_service_weights(service_name, instances)
            
    async def _adjust_service_weights(self, service_name: str, instances: List[ServiceInstance]):
        """调整特定服务的权重"""
        
        # 计算性能指标
        total_score = 0
        for instance in instances:
            if instance.is_healthy:
                # 性能得分（越高越好）
                performance_score = self._calculate_performance_score(instance)
                total_score += performance_score
                
        # 重新分配权重
        for instance in instances:
            if instance.is_healthy and total_score > 0:
                performance_score = self._calculate_performance_score(instance)
                # 基于性能调整权重
                new_weight = max(1, int((performance_score / total_score) * 10))
                instance.weight = new_weight
            else:
                instance.weight = 0  # 不健康的实例权重为0
                
    def _calculate_performance_score(self, instance: ServiceInstance) -> float:
        """计算性能得分"""
        # 响应时间得分（响应时间越短得分越高）
        response_score = max(0, 1000 - instance.avg_response_time) / 1000
        
        # 资源使用得分（使用率越低得分越高）
        resource_score = max(0, 200 - instance.cpu_usage - instance.memory_usage) / 200
        
        # 连接数得分（连接数越少得分越高）
        connection_ratio = instance.current_connections / instance.max_concurrent
        connection_score = max(0, 1 - connection_ratio)
        
        # 错误率得分（错误率越低得分越高）
        error_score = max(0, 1 - instance.error_rate)
        
        # 综合得分
        total_score = (
            response_score * 0.3 +
            resource_score * 0.3 +
            connection_score * 0.2 +
            error_score * 0.2
        )
        
        return total_score
```

---

## 🐳 **容器化负载均衡部署**

### **多实例Docker Compose配置**

```yaml
# docker-compose-load-balanced.yml
version: '3.8'

services:
  # Nginx负载均衡器
  nginx-lb:
    image: nginx:alpine
    container_name: xiaozhi-nginx-lb
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/conf.d:/etc/nginx/conf.d
    depends_on:
      - xiaozhi-server-1
      - xiaozhi-server-2
    restart: unless-stopped
    networks:
      - xiaozhi-network

  # 主服务实例1
  xiaozhi-server-1:
    image: xiaozhi-esp32-server:latest
    container_name: xiaozhi-server-1
    ports:
      - "8000:8000"
    environment:
      - SERVER_ID=server-1
      - REDIS_URL=redis://xiaozhi-redis:6379
      - DATABASE_URL=mysql://xiaozhi-db:3306/xiaozhi
      - INSTANCE_WEIGHT=3
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1.5G
    restart: unless-stopped
    networks:
      - xiaozhi-network

  # 主服务实例2
  xiaozhi-server-2:
    image: xiaozhi-esp32-server:latest
    container_name: xiaozhi-server-2
    ports:
      - "8001:8000"
    environment:
      - SERVER_ID=server-2
      - REDIS_URL=redis://xiaozhi-redis:6379
      - DATABASE_URL=mysql://xiaozhi-db:3306/xiaozhi
      - INSTANCE_WEIGHT=2
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1.5G
        reservations:
          cpus: '0.5'
          memory: 1G
    restart: unless-stopped
    networks:
      - xiaozhi-network

  # ASR服务实例1
  xiaozhi-asr-1:
    image: xiaozhi-asr:latest
    container_name: xiaozhi-asr-1
    ports:
      - "8100:8000"
    environment:
      - ASR_MODEL=SenseVoiceStream
      - INSTANCE_ID=asr-1
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
    restart: unless-stopped
    networks:
      - xiaozhi-network

  # ASR服务实例2
  xiaozhi-asr-2:
    image: xiaozhi-asr:latest
    container_name: xiaozhi-asr-2
    ports:
      - "8101:8000"
    environment:
      - ASR_MODEL=SenseVoiceStream
      - INSTANCE_ID=asr-2
    deploy:
      resources:
        limits:
          cpus: '0.8'
          memory: 800M
    restart: unless-stopped
    networks:
      - xiaozhi-network

  # 本地LLM服务
  xiaozhi-llm-local:
    image: ollama/ollama:latest
    container_name: xiaozhi-llm-local
    ports:
      - "11434:11434"
    volumes:
      - ./models:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
    restart: unless-stopped
    networks:
      - xiaozhi-network

  # 本地TTS服务
  xiaozhi-tts-local:
    build:
      context: .
      dockerfile: Dockerfile.edge-tts
    container_name: xiaozhi-tts-local
    ports:
      - "5000:5000"
    volumes:
      - ./tts_cache:/tmp/tts_cache
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
    restart: unless-stopped
    networks:
      - xiaozhi-network

  # 共享数据库
  xiaozhi-db:
    image: mysql:8.0
    container_name: xiaozhi-db
    environment:
      - MYSQL_ROOT_PASSWORD=xiaozhi123
      - MYSQL_DATABASE=xiaozhi
    volumes:
      - ./mysql_data:/var/lib/mysql
      - ./mysql_config/my.cnf:/etc/mysql/my.cnf
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
    restart: unless-stopped
    networks:
      - xiaozhi-network

  # 共享Redis
  xiaozhi-redis:
    image: redis:7-alpine
    container_name: xiaozhi-redis
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - ./redis_data:/data
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    restart: unless-stopped
    networks:
      - xiaozhi-network

networks:
  xiaozhi-network:
    driver: bridge
```

### **Nginx负载均衡配置**

```nginx
# nginx/nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 1024;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';
    
    access_log /var/log/nginx/access.log main;
    
    # 基础配置
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    
    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
    
    # 上游服务器定义
    upstream xiaozhi_backend {
        least_conn;  # 最少连接数算法
        
        server xiaozhi-server-1:8000 weight=3 max_fails=2 fail_timeout=30s;
        server xiaozhi-server-2:8000 weight=2 max_fails=2 fail_timeout=30s;
        
        keepalive 32;
    }
    
    upstream xiaozhi_asr {
        least_conn;
        
        server xiaozhi-asr-1:8000 weight=2 max_fails=2 fail_timeout=30s;
        server xiaozhi-asr-2:8000 weight=1 max_fails=2 fail_timeout=30s;
        
        keepalive 16;
    }
    
    upstream xiaozhi_llm {
        ip_hash;  # 会话保持
        
        server xiaozhi-llm-local:11434 weight=4 max_fails=1 fail_timeout=10s;
        # 远程LLM作为备份
        # server remote-llm-api:443 weight=1 max_fails=3 fail_timeout=60s backup;
        
        keepalive 8;
    }
    
    upstream xiaozhi_tts {
        least_conn;
        
        server xiaozhi-tts-local:5000 weight=4 max_fails=1 fail_timeout=10s;
        # 远程TTS作为备份
        # server remote-tts-api:443 weight=1 max_fails=3 fail_timeout=60s backup;
        
        keepalive 16;
    }
    
    # 主服务器配置
    server {
        listen 80;
        server_name localhost;
        
        # 主API路由
        location / {
            proxy_pass http://xiaozhi_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 连接和超时设置
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # 缓冲设置
            proxy_buffering on;
            proxy_buffer_size 4k;
            proxy_buffers 8 4k;
            
            # 健康检查
            proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
            proxy_next_upstream_tries 2;
            proxy_next_upstream_timeout 3s;
        }
        
        # ASR服务路由
        location /asr/ {
            proxy_pass http://xiaozhi_asr/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # ASR特殊配置
            proxy_connect_timeout 3s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
            
            # 禁用缓冲以支持流式处理
            proxy_buffering off;
            proxy_request_buffering off;
        }
        
        # LLM服务路由
        location /llm/ {
            proxy_pass http://xiaozhi_llm/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # LLM特殊配置
            proxy_connect_timeout 5s;
            proxy_send_timeout 120s;
            proxy_read_timeout 120s;
            
            # 支持流式响应
            proxy_buffering off;
            proxy_cache off;
        }
        
        # TTS服务路由
        location /tts/ {
            proxy_pass http://xiaozhi_tts/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # TTS特殊配置
            proxy_connect_timeout 3s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            # 音频文件缓存
            proxy_cache tts_cache;
            proxy_cache_valid 200 1h;
            proxy_cache_key "$request_uri$request_body";
        }
        
        # WebSocket支持
        location /ws/ {
            proxy_pass http://xiaozhi_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # WebSocket特殊配置
            proxy_connect_timeout 7d;
            proxy_send_timeout 7d;
            proxy_read_timeout 7d;
        }
        
        # 健康检查端点
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
        
        # 状态监控
        location /nginx_status {
            stub_status on;
            access_log off;
            allow 127.0.0.1;
            deny all;
        }
    }
    
    # 缓存配置
    proxy_cache_path /var/cache/nginx/tts levels=1:2 keys_zone=tts_cache:10m max_size=1g inactive=1h use_temp_path=off;
}
```

---

## 📊 **性能提升预期**

### **负载均衡效果对比**

| 指标 | 单实例部署 | 负载均衡部署 | 改善幅度 |
|------|------------|--------------|----------|
| **最大并发设备** | 10台 | 30-50台 | **+300-400%** |
| **平均响应时间** | 2000-4000ms | 500-1000ms | **-60-75%** |
| **系统可用性** | 95% | 99.5% | **+4.5%** |
| **故障恢复时间** | 5-10分钟 | 30-60秒 | **-90%** |
| **资源利用率** | 30% | 75-85% | **+150%** |
| **扩展能力** | 无 | 水平扩展 | **无限** |

### **具体改善潜力分析**

```yaml
improvement_analysis:
  # CPU利用率优化
  cpu_optimization:
    current_usage: "< 5%"
    optimized_usage: "60-80%"
    improvement: "+1500%"
    methods:
      - "多实例并行处理"
      - "智能任务分发"
      - "CPU亲和性绑定"
      
  # 内存利用率优化
  memory_optimization:
    current_usage: "45% (主服务)"
    optimized_usage: "75-85%"
    improvement: "+80%"
    methods:
      - "内存池共享"
      - "缓存策略优化"
      - "垃圾回收调优"
      
  # 网络吞吐量优化
  network_optimization:
    current_throughput: "低"
    optimized_throughput: "高"
    improvement: "+200-300%"
    methods:
      - "连接复用"
      - "请求管道化"
      - "压缩传输"
      
  # 服务可靠性提升
  reliability_improvement:
    single_point_failure: "消除"
    auto_failover: "启用"
    health_monitoring: "实时"
    improvement: "+400%"
```

---

## 🔧 **实施步骤**

### **阶段1: 基础负载均衡 (1-2天)**

```bash
#!/bin/bash
# phase1_basic_lb.sh

echo "=== 阶段1: 基础负载均衡部署 ==="

# 1. 创建Nginx配置
mkdir -p nginx/conf.d
cp nginx.conf nginx/
cp default.conf nginx/conf.d/

# 2. 修改Docker Compose
cp docker-compose.yml docker-compose.backup.yml
cp docker-compose-load-balanced.yml docker-compose.yml

# 3. 启动负载均衡服务
docker-compose down
docker-compose up -d nginx-lb xiaozhi-server-1 xiaozhi-server-2

# 4. 验证负载均衡
echo "验证负载均衡..."
for i in {1..10}; do
    curl -s http://localhost/health | grep -o "server-[12]"
done

echo "阶段1完成！"
```

### **阶段2: 服务分离 (2-3天)**

```bash
#!/bin/bash
# phase2_service_separation.sh

echo "=== 阶段2: 服务分离部署 ==="

# 1. 部署独立ASR服务
docker-compose up -d xiaozhi-asr-1 xiaozhi-asr-2

# 2. 部署本地LLM服务
docker-compose up -d xiaozhi-llm-local

# 3. 部署本地TTS服务
docker-compose up -d xiaozhi-tts-local

# 4. 更新Nginx配置
docker-compose restart nginx-lb

# 5. 验证服务分离
echo "验证ASR服务..."
curl -s http://localhost/asr/health

echo "验证LLM服务..."
curl -s http://localhost/llm/health

echo "验证TTS服务..."
curl -s http://localhost/tts/health

echo "阶段2完成！"
```

### **阶段3: 智能路由 (3-4天)**

```bash
#!/bin/bash
# phase3_intelligent_routing.sh

echo "=== 阶段3: 智能路由部署 ==="

# 1. 部署API网关
docker build -t xiaozhi-api-gateway -f Dockerfile.gateway .
docker-compose up -d xiaozhi-api-gateway

# 2. 配置智能路由
cp intelligent_router.py services/
cp dynamic_weight_adjuster.py services/

# 3. 启动监控服务
docker-compose up -d xiaozhi-monitor

# 4. 验证智能路由
echo "验证智能路由..."
python test_intelligent_routing.py

echo "阶段3完成！"
```

---

## 📈 **监控和调优**

### **关键性能指标 (KPI)**

```yaml
monitoring_kpis:
  # 负载均衡指标
  load_balancing:
    - metric: "request_distribution_ratio"
      target: "按权重分配 ±5%"
      alert_threshold: "偏差 > 10%"
      
    - metric: "instance_health_status"
      target: "所有实例健康"
      alert_threshold: "任一实例不健康"
      
    - metric: "failover_time"
      target: "< 30秒"
      alert_threshold: "> 60秒"
      
  # 性能指标
  performance:
    - metric: "avg_response_time"
      target: "< 500ms"
      alert_threshold: "> 1000ms"
      
    - metric: "concurrent_connections"
      target: "30-50个设备"
      alert_threshold: "> 60个设备"
      
    - metric: "error_rate"
      target: "< 1%"
      alert_threshold: "> 5%"
      
  # 资源利用率
  resource_utilization:
    - metric: "cpu_usage"
      target: "60-80%"
      alert_threshold: "> 90%"
      
    - metric: "memory_usage"
      target: "70-85%"
      alert_threshold: "> 95%"
      
    - metric: "network_bandwidth"
      target: "< 80%"
      alert_threshold: "> 90%"
```

### **自动化调优脚本**

```python
# auto_tuning.py
import asyncio
import json
import time
from typing import Dict, List

class AutoTuner:
    def __init__(self):
        self.metrics_history = []
        self.tuning_rules = self.load_tuning_rules()
        
    def load_tuning_rules(self) -> Dict:
        """加载调优规则"""
        return {
            "cpu_high": {
                "condition": "cpu_usage > 85%",
                "actions": [
                    "increase_instance_count",
                    "reduce_worker_threads",
                    "enable_cpu_throttling"
                ]
            },
            "memory_high": {
                "condition": "memory_usage > 90%",
                "actions": [
                    "clear_cache",
                    "restart_heavy_services",
                    "reduce_buffer_size"
                ]
            },
            "response_slow": {
                "condition": "avg_response_time > 1000ms",
                "actions": [
                    "switch_to_local_models",
                    "increase_cache_size",
                    "optimize_database_queries"
                ]
            },
            "high_error_rate": {
                "condition": "error_rate > 5%",
                "actions": [
                    "restart_failing_services",
                    "switch_to_backup_providers",
                    "reduce_concurrent_requests"
                ]
            }
        }
        
    async def monitor_and_tune(self):
        """监控并自动调优"""
        while True:
            try:
                # 收集指标
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # 保持历史记录在合理范围内
                if len(self.metrics_history) > 100:
                    self.metrics_history = self.metrics_history[-100:]
                
                # 分析并执行调优
                await self.analyze_and_tune(metrics)
                
                # 等待下一次检查
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                print(f"自动调优错误: {e}")
                await asyncio.sleep(30)
                
    async def collect_metrics(self) -> Dict:
        """收集系统指标"""
        # 这里应该实际收集指标
        # 示例指标
        return {
            "timestamp": time.time(),
            "cpu_usage": 75.0,
            "memory_usage": 80.0,
            "avg_response_time": 800.0,
            "error_rate": 2.0,
            "concurrent_connections": 25,
            "instance_health": {
                "xiaozhi-server-1": True,
                "xiaozhi-server-2": True,
                "xiaozhi-asr-1": True,
                "xiaozhi-asr-2": False  # 示例：ASR-2不健康
            }
        }
        
    async def analyze_and_tune(self, metrics: Dict):
        """分析指标并执行调优"""
        for rule_name, rule in self.tuning_rules.items():
            if self.evaluate_condition(rule["condition"], metrics):
                print(f"触发调优规则: {rule_name}")
                for action in rule["actions"]:
                    await self.execute_action(action, metrics)
                    
    def evaluate_condition(self, condition: str, metrics: Dict) -> bool:
        """评估调优条件"""
        # 简化的条件评估
        if "cpu_usage > 85%" in condition:
            return metrics["cpu_usage"] > 85
        elif "memory_usage > 90%" in condition:
            return metrics["memory_usage"] > 90
        elif "avg_response_time > 1000ms" in condition:
            return metrics["avg_response_time"] > 1000
        elif "error_rate > 5%" in condition:
            return metrics["error_rate"] > 5
        return False
        
    async def execute_action(self, action: str, metrics: Dict):
        """执行调优动作"""
        print(f"执行调优动作: {action}")
        
        if action == "increase_instance_count":
            await self.scale_up_instances()
        elif action == "clear_cache":
            await self.clear_system_cache()
        elif action == "restart_heavy_services":
            await self.restart_services(["xiaozhi-server-1"])
        elif action == "switch_to_local_models":
            await self.switch_to_local_models()
        # ... 其他动作实现
        
    async def scale_up_instances(self):
        """扩展实例数量"""
        # 实现实例扩展逻辑
        pass
        
    async def clear_system_cache(self):
        """清理系统缓存"""
        # 实现缓存清理逻辑
        pass
        
    async def restart_services(self, services: List[str]):
        """重启服务"""
        # 实现服务重启逻辑
        pass
        
    async def switch_to_local_models(self):
        """切换到本地模型"""
        # 实现模型切换逻辑
        pass

# 启动自动调优
if __name__ == "__main__":
    tuner = AutoTuner()
    asyncio.run(tuner.monitor_and_tune())
```

---

## ✅ **部署检查清单**

### **部署前准备**
- [ ] 确认硬件资源充足 (4核8GB)
- [ ] 备份现有配置和数据
- [ ] 准备Nginx配置文件
- [ ] 准备多实例Docker配置

### **部署步骤**
- [ ] 阶段1: 部署基础负载均衡
- [ ] 阶段2: 实施服务分离
- [ ] 阶段3: 配置智能路由
- [ ] 配置监控和告警
- [ ] 部署自动调优系统

### **部署后验证**
- [ ] 负载分发正常工作
- [ ] 故障转移机制正常
- [ ] 性能指标达到预期
- [ ] 监控系统正常运行
- [ ] 自动调优功能正常

**预期效果**: 设备支撑能力从10台提升到30-50台，改善潜力达到**300-400%**！