# LLM和TTS服务扩容部署指南

## 概述

本指南详细说明如何部署和配置增强版的LLM和TTS服务，实现基于远程API的智能路由、负载均衡和故障转移。

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   用户请求      │    │   负载均衡器    │    │   智能路由器    │
│                 │───▶│                 │───▶│                 │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
                       ┌────────────────────────────────┼────────────────────────────────┐
                       │                                │                                │
                       ▼                                ▼                                ▼
            ┌─────────────────┐              ┌─────────────────┐              ┌─────────────────┐
            │   OpenAI API    │              │   通义千问API   │              │   百川API      │
            │                 │              │                 │              │                 │
            └─────────────────┘              └─────────────────┘              └─────────────────┘
                       │                                │                                │
                       └────────────────────────────────┼────────────────────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │   Redis缓存     │
                                              │                 │
                                              └─────────────────┘
```

## 前置要求

### 系统要求
- Python 3.8+
- Redis 6.0+
- Docker (可选)
- Kubernetes (可选)

### 依赖包
```bash
pip install fastapi uvicorn redis aioredis httpx asyncio-mqtt prometheus-client
```

## 配置步骤

### 1. 环境变量配置

创建 `.env` 文件：

```bash
# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
REDIS_DB=0

# LLM API配置
OPENAI_API_KEY=your_openai_api_key
QWEN_API_KEY=your_qwen_api_key
BAICHUAN_API_KEY=your_baichuan_api_key
GLM_API_KEY=your_glm_api_key
MOONSHOT_API_KEY=your_moonshot_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# TTS API配置
AZURE_TTS_KEY=your_azure_tts_key
AZURE_TTS_REGION=your_azure_region
GOOGLE_TTS_KEY=your_google_tts_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
BAIDU_TTS_API_KEY=your_baidu_api_key
BAIDU_TTS_SECRET_KEY=your_baidu_secret_key
XUNFEI_APP_ID=your_xunfei_app_id
XUNFEI_API_KEY=your_xunfei_api_key
XUNFEI_API_SECRET=your_xunfei_api_secret

# 服务配置
LLM_SERVICE_PORT=8001
TTS_SERVICE_PORT=8002
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
CACHE_TTL=3600

# 监控配置
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ENABLE_METRICS=true

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=/var/log/xiaozhi/service.log
```

### 2. 服务配置文件

创建 `config/production.yaml`：

```yaml
# 生产环境配置
llm_service:
  providers:
    openai:
      enabled: true
      api_key: ${OPENAI_API_KEY}
      base_url: "https://api.openai.com/v1"
      models:
        - name: "gpt-4o-mini"
          max_tokens: 4096
          cost_per_1k_tokens: 0.00015
          max_concurrent: 50
        - name: "gpt-4o"
          max_tokens: 4096
          cost_per_1k_tokens: 0.005
          max_concurrent: 20
      timeout: 30
      retry_attempts: 3
      
    qwen:
      enabled: true
      api_key: ${QWEN_API_KEY}
      base_url: "https://dashscope.aliyuncs.com/api/v1"
      models:
        - name: "qwen-plus"
          max_tokens: 8192
          cost_per_1k_tokens: 0.04
          max_concurrent: 100
        - name: "qwen-turbo"
          max_tokens: 8192
          cost_per_1k_tokens: 0.008
          max_concurrent: 200
      timeout: 20
      retry_attempts: 3
      
    deepseek:
      enabled: true
      api_key: ${DEEPSEEK_API_KEY}
      base_url: "https://api.deepseek.com/v1"
      models:
        - name: "deepseek-chat"
          max_tokens: 4096
          cost_per_1k_tokens: 0.001
          max_concurrent: 100
      timeout: 25
      retry_attempts: 3

  routing:
    strategy: "intelligent"
    weights:
      cost: 0.3
      latency: 0.3
      success_rate: 0.4
    
  caching:
    enabled: true
    ttl: 3600
    max_size: 10000
    strategy: "adaptive"
    
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
    half_open_max_calls: 3

tts_service:
  providers:
    azure:
      enabled: true
      api_key: ${AZURE_TTS_KEY}
      region: ${AZURE_TTS_REGION}
      voices:
        - name: "zh-CN-XiaoxiaoNeural"
          gender: "female"
          cost_per_1k_chars: 0.016
        - name: "zh-CN-YunxiNeural"
          gender: "male"
          cost_per_1k_chars: 0.016
      max_concurrent: 50
      timeout: 30
      
    baidu:
      enabled: true
      api_key: ${BAIDU_TTS_API_KEY}
      secret_key: ${BAIDU_TTS_SECRET_KEY}
      voices:
        - name: "zh"
          gender: "female"
          cost_per_1k_chars: 0.033
      max_concurrent: 30
      timeout: 25
      
    edge_tts:
      enabled: true
      voices:
        - name: "zh-CN-XiaoxiaoNeural"
          gender: "female"
          cost_per_1k_chars: 0
      max_concurrent: 20
      timeout: 20
      
  routing:
    strategy: "intelligent"
    weights:
      cost: 0.4
      latency: 0.3
      quality: 0.3
      
  caching:
    enabled: true
    ttl: 7200
    max_size: 5000
    
monitoring:
  enabled: true
  metrics_port: 9090
  health_check_interval: 30
  log_level: "INFO"
```

### 3. Docker部署

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建日志目录
RUN mkdir -p /var/log/xiaozhi

# 暴露端口
EXPOSE 8001 8002 9090

# 启动脚本
CMD ["python", "scripts/start_services.py"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  redis:
    image: redis:6.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    
  llm-service:
    build: .
    ports:
      - "8001:8001"
    environment:
      - SERVICE_TYPE=llm
      - REDIS_HOST=redis
    depends_on:
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/var/log/xiaozhi
    restart: unless-stopped
    
  tts-service:
    build: .
    ports:
      - "8002:8002"
    environment:
      - SERVICE_TYPE=tts
      - REDIS_HOST=redis
    depends_on:
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/var/log/xiaozhi
    restart: unless-stopped
    
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
      
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning

volumes:
  redis_data:
  prometheus_data:
  grafana_data:
```

### 4. Kubernetes部署

#### namespace.yaml

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: xiaozhi-services
```

#### configmap.yaml

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: service-config
  namespace: xiaozhi-services
data:
  production.yaml: |
    # 配置内容（同上面的production.yaml）
```

#### secret.yaml

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: api-keys
  namespace: xiaozhi-services
type: Opaque
stringData:
  OPENAI_API_KEY: "your_openai_api_key"
  QWEN_API_KEY: "your_qwen_api_key"
  BAICHUAN_API_KEY: "your_baichuan_api_key"
  # 其他API密钥...
```

#### redis-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis
  namespace: xiaozhi-services
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis
  template:
    metadata:
      labels:
        app: redis
    spec:
      containers:
      - name: redis
        image: redis:6.2-alpine
        ports:
        - containerPort: 6379
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis
  namespace: xiaozhi-services
spec:
  selector:
    app: redis
  ports:
  - port: 6379
    targetPort: 6379
```

#### llm-service-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service
  namespace: xiaozhi-services
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-service
  template:
    metadata:
      labels:
        app: llm-service
    spec:
      containers:
      - name: llm-service
        image: xiaozhi/llm-service:latest
        ports:
        - containerPort: 8001
        env:
        - name: SERVICE_TYPE
          value: "llm"
        - name: REDIS_HOST
          value: "redis"
        envFrom:
        - secretRef:
            name: api-keys
        volumeMounts:
        - name: config
          mountPath: /app/config
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: service-config
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
  namespace: xiaozhi-services
spec:
  selector:
    app: llm-service
  ports:
  - port: 8001
    targetPort: 8001
  type: ClusterIP
```

#### tts-service-deployment.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tts-service
  namespace: xiaozhi-services
spec:
  replicas: 2
  selector:
    matchLabels:
      app: tts-service
  template:
    metadata:
      labels:
        app: tts-service
    spec:
      containers:
      - name: tts-service
        image: xiaozhi/tts-service:latest
        ports:
        - containerPort: 8002
        env:
        - name: SERVICE_TYPE
          value: "tts"
        - name: REDIS_HOST
          value: "redis"
        envFrom:
        - secretRef:
            name: api-keys
        volumeMounts:
        - name: config
          mountPath: /app/config
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "300m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8002
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config
        configMap:
          name: service-config
---
apiVersion: v1
kind: Service
metadata:
  name: tts-service
  namespace: xiaozhi-services
spec:
  selector:
    app: tts-service
  ports:
  - port: 8002
    targetPort: 8002
  type: ClusterIP
```

#### ingress.yaml

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: xiaozhi-ingress
  namespace: xiaozhi-services
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - host: xiaozhi.local
    http:
      paths:
      - path: /llm
        pathType: Prefix
        backend:
          service:
            name: llm-service
            port:
              number: 8001
      - path: /tts
        pathType: Prefix
        backend:
          service:
            name: tts-service
            port:
              number: 8002
```

## 启动脚本

创建 `scripts/start_services.py`：

```python
#!/usr/bin/env python3
import os
import sys
import asyncio
import uvicorn
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def start_llm_service():
    """启动LLM服务"""
    from enhanced_llm_service import app as llm_app
    config = uvicorn.Config(
        llm_app,
        host="0.0.0.0",
        port=int(os.getenv("LLM_SERVICE_PORT", 8001)),
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def start_tts_service():
    """启动TTS服务"""
    from enhanced_tts_service import app as tts_app
    config = uvicorn.Config(
        tts_app,
        host="0.0.0.0",
        port=int(os.getenv("TTS_SERVICE_PORT", 8002)),
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    service_type = os.getenv("SERVICE_TYPE", "both")
    
    if service_type == "llm":
        await start_llm_service()
    elif service_type == "tts":
        await start_tts_service()
    else:
        # 同时启动两个服务
        await asyncio.gather(
            start_llm_service(),
            start_tts_service()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

## 部署步骤

### 1. 本地开发环境

```bash
# 1. 克隆代码
git clone <repository_url>
cd xiaozhi-server

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑.env文件，填入API密钥

# 4. 启动Redis
docker run -d --name redis -p 6379:6379 redis:6.2-alpine

# 5. 启动服务
python scripts/start_services.py
```

### 2. Docker部署

```bash
# 1. 构建镜像
docker-compose build

# 2. 启动服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 健康检查
curl http://localhost:8001/health
curl http://localhost:8002/health
```

### 3. Kubernetes部署

```bash
# 1. 创建命名空间
kubectl apply -f k8s/namespace.yaml

# 2. 创建配置和密钥
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# 3. 部署Redis
kubectl apply -f k8s/redis-deployment.yaml

# 4. 部署服务
kubectl apply -f k8s/llm-service-deployment.yaml
kubectl apply -f k8s/tts-service-deployment.yaml

# 5. 创建Ingress
kubectl apply -f k8s/ingress.yaml

# 6. 检查部署状态
kubectl get pods -n xiaozhi-services
kubectl get services -n xiaozhi-services
```

## 监控和维护

### 1. 健康检查

```bash
# LLM服务健康检查
curl http://localhost:8001/health

# TTS服务健康检查
curl http://localhost:8002/health

# 服务统计信息
curl http://localhost:8001/stats
curl http://localhost:8002/stats
```

### 2. 日志监控

```bash
# 查看服务日志
tail -f /var/log/xiaozhi/service.log

# Docker环境
docker-compose logs -f llm-service
docker-compose logs -f tts-service

# Kubernetes环境
kubectl logs -f deployment/llm-service -n xiaozhi-services
kubectl logs -f deployment/tts-service -n xiaozhi-services
```

### 3. 性能监控

访问Grafana仪表板：
- URL: http://localhost:3000
- 用户名: admin
- 密码: admin

关键监控指标：
- 请求响应时间
- 请求成功率
- API调用成本
- 缓存命中率
- 服务可用性

### 4. 故障排除

#### 常见问题

1. **API密钥错误**
   ```bash
   # 检查环境变量
   echo $OPENAI_API_KEY
   
   # 更新密钥
   kubectl patch secret api-keys -n xiaozhi-services -p '{"stringData":{"OPENAI_API_KEY":"new_key"}}'
   ```

2. **Redis连接失败**
   ```bash
   # 检查Redis状态
   redis-cli ping
   
   # Docker环境
   docker-compose exec redis redis-cli ping
   
   # Kubernetes环境
   kubectl exec -it deployment/redis -n xiaozhi-services -- redis-cli ping
   ```

3. **服务响应慢**
   ```bash
   # 检查服务负载
   curl http://localhost:8001/stats
   
   # 调整并发限制
   # 编辑配置文件中的max_concurrent参数
   ```

4. **内存不足**
   ```bash
   # 检查内存使用
   docker stats
   
   # Kubernetes环境
   kubectl top pods -n xiaozhi-services
   
   # 调整资源限制
   # 编辑deployment.yaml中的resources配置
   ```

## 扩容策略

### 1. 水平扩容

```bash
# Docker Compose
docker-compose up --scale llm-service=3 --scale tts-service=2

# Kubernetes
kubectl scale deployment llm-service --replicas=5 -n xiaozhi-services
kubectl scale deployment tts-service --replicas=3 -n xiaozhi-services
```

### 2. 垂直扩容

```yaml
# 更新资源限制
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### 3. 自动扩容

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-service-hpa
  namespace: xiaozhi-services
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 安全配置

### 1. API密钥管理

```bash
# 使用Kubernetes Secrets
kubectl create secret generic api-keys \
  --from-literal=OPENAI_API_KEY=your_key \
  --from-literal=QWEN_API_KEY=your_key \
  -n xiaozhi-services

# 定期轮换密钥
kubectl patch secret api-keys -n xiaozhi-services -p '{"stringData":{"OPENAI_API_KEY":"new_key"}}'
```

### 2. 网络安全

```yaml
# NetworkPolicy示例
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: xiaozhi-network-policy
  namespace: xiaozhi-services
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

### 3. 访问控制

```yaml
# RBAC配置
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: xiaozhi-services
  name: xiaozhi-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]
```

## 备份和恢复

### 1. Redis数据备份

```bash
# 手动备份
redis-cli BGSAVE

# 定时备份脚本
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb /backup/redis_backup_$DATE.rdb
```

### 2. 配置备份

```bash
# 备份Kubernetes配置
kubectl get all -n xiaozhi-services -o yaml > backup/k8s_config_$(date +%Y%m%d).yaml

# 备份ConfigMap和Secret
kubectl get configmap service-config -n xiaozhi-services -o yaml > backup/configmap.yaml
kubectl get secret api-keys -n xiaozhi-services -o yaml > backup/secret.yaml
```

这个部署指南提供了完整的部署流程，包括本地开发、Docker和Kubernetes部署方式，以及监控、维护和故障排除的详细说明。