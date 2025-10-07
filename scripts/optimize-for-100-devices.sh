#!/bin/bash
# Xiaozhi ESP32 Server - 100台设备优化部署脚本
# 基于性能评估报告的优化建议

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装，请先安装 Kubernetes CLI"
        exit 1
    fi
    
    # 检查docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    # 检查helm
    if ! command -v helm &> /dev/null; then
        log_warning "Helm 未安装，将跳过 Helm 相关部署"
    fi
    
    log_success "依赖检查完成"
}

# 优化Docker镜像
optimize_docker_images() {
    log_info "优化Docker镜像..."
    
    # VAD服务优化镜像
    cat > Dockerfile.vad.optimized << 'EOF'
FROM python:3.9-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.vad.txt .
RUN pip install --no-cache-dir -r requirements.vad.txt

# 复制应用代码
COPY services/vad_service.py /app/
COPY config/ /app/config/

WORKDIR /app

# 优化启动参数
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV CUDA_VISIBLE_DEVICES=0

EXPOSE 8004

CMD ["python", "-m", "uvicorn", "vad_service:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "1"]
EOF

    # ASR服务优化镜像
    cat > Dockerfile.asr.optimized << 'EOF'
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements.asr.txt .
RUN pip3 install --no-cache-dir -r requirements.asr.txt

# 复制应用代码和模型
COPY services/asr_service.py /app/
COPY models/SenseVoiceSmall/ /app/models/
COPY config/ /app/config/

WORKDIR /app

# 优化启动参数
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

EXPOSE 8001

CMD ["python3", "-m", "uvicorn", "asr_service:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
EOF

    # 构建优化镜像
    log_info "构建VAD优化镜像..."
    docker build -f Dockerfile.vad.optimized -t xiaozhi/vad-service:optimized .
    
    log_info "构建ASR优化镜像..."
    docker build -f Dockerfile.asr.optimized -t xiaozhi/asr-service:optimized .
    
    log_success "Docker镜像优化完成"
}

# 创建优化的Kubernetes配置
create_optimized_k8s_configs() {
    log_info "创建优化的Kubernetes配置..."
    
    # 创建优化配置目录
    mkdir -p k8s/optimized
    
    # VAD服务优化配置
    cat > k8s/optimized/vad-service-optimized.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vad-service-optimized
  namespace: xiaozhi-system
spec:
  replicas: 8
  selector:
    matchLabels:
      app: vad-service
      version: optimized
  template:
    metadata:
      labels:
        app: vad-service
        version: optimized
    spec:
      containers:
      - name: vad-service
        image: xiaozhi/vad-service:optimized
        ports:
        - containerPort: 8004
        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
            nvidia.com/gpu: 0.5
          limits:
            cpu: 4000m
            memory: 8Gi
            nvidia.com/gpu: 0.5
        env:
        - name: BATCH_SIZE
          value: "32"
        - name: MAX_CONCURRENT
          value: "24"
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        - name: MODEL_OPTIMIZATION
          value: "true"
        livenessProbe:
          httpGet:
            path: /health
            port: 8004
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8004
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: vad-service-optimized
  namespace: xiaozhi-system
spec:
  selector:
    app: vad-service
    version: optimized
  ports:
  - port: 8004
    targetPort: 8004
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vad-service-hpa
  namespace: xiaozhi-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vad-service-optimized
  minReplicas: 4
  maxReplicas: 16
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
EOF

    # ASR服务优化配置
    cat > k8s/optimized/asr-service-optimized.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: asr-service-optimized
  namespace: xiaozhi-system
spec:
  replicas: 12
  selector:
    matchLabels:
      app: asr-service
      version: optimized
  template:
    metadata:
      labels:
        app: asr-service
        version: optimized
    spec:
      containers:
      - name: asr-service
        image: xiaozhi/asr-service:optimized
        ports:
        - containerPort: 8001
        resources:
          requests:
            cpu: 4000m
            memory: 8Gi
            nvidia.com/gpu: 1
          limits:
            cpu: 8000m
            memory: 16Gi
            nvidia.com/gpu: 1
        env:
        - name: BATCH_SIZE
          value: "8"
        - name: MAX_CONCURRENT
          value: "16"
        - name: MODEL_PATH
          value: "/app/models/SenseVoiceSmall"
        - name: STREAMING_ENABLED
          value: "true"
        - name: FP16_OPTIMIZATION
          value: "true"
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 15
        readinessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: asr-service-optimized
  namespace: xiaozhi-system
spec:
  selector:
    app: asr-service
    version: optimized
  ports:
  - port: 8001
    targetPort: 8001
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: asr-service-hpa
  namespace: xiaozhi-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: asr-service-optimized
  minReplicas: 8
  maxReplicas: 24
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 75
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 85
EOF

    log_success "Kubernetes优化配置创建完成"
}

# 部署Redis集群优化配置
deploy_redis_optimization() {
    log_info "部署Redis集群优化配置..."
    
    cat > k8s/optimized/redis-cluster-optimized.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-cluster-config
  namespace: xiaozhi-system
data:
  redis.conf: |
    # Redis优化配置 - 100台设备
    port 6379
    bind 0.0.0.0
    protected-mode no
    
    # 内存优化
    maxmemory 8gb
    maxmemory-policy allkeys-lru
    
    # 持久化优化
    save 900 1
    save 300 10
    save 60 10000
    
    # 网络优化
    tcp-keepalive 300
    timeout 0
    tcp-backlog 511
    
    # 性能优化
    databases 16
    hash-max-ziplist-entries 512
    hash-max-ziplist-value 64
    list-max-ziplist-size -2
    set-max-intset-entries 512
    zset-max-ziplist-entries 128
    zset-max-ziplist-value 64
    
    # 集群配置
    cluster-enabled yes
    cluster-config-file nodes.conf
    cluster-node-timeout 15000
    cluster-announce-ip ${POD_IP}
    cluster-announce-port 6379
    cluster-announce-bus-port 16379
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: xiaozhi-system
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        command:
        - redis-server
        - /etc/redis/redis.conf
        env:
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        volumeMounts:
        - name: conf
          mountPath: /etc/redis/
        - name: data
          mountPath: /data
        resources:
          requests:
            cpu: 1000m
            memory: 8Gi
          limits:
            cpu: 2000m
            memory: 16Gi
      volumes:
      - name: conf
        configMap:
          name: redis-cluster-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 100Gi
EOF

    kubectl apply -f k8s/optimized/redis-cluster-optimized.yaml
    log_success "Redis集群优化配置部署完成"
}

# 部署智能负载均衡器
deploy_intelligent_load_balancer() {
    log_info "部署智能负载均衡器..."
    
    cat > k8s/optimized/intelligent-load-balancer.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: intelligent-load-balancer
  namespace: xiaozhi-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: intelligent-load-balancer
  template:
    metadata:
      labels:
        app: intelligent-load-balancer
    spec:
      containers:
      - name: load-balancer
        image: xiaozhi/intelligent-load-balancer:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 2000m
            memory: 4Gi
          limits:
            cpu: 4000m
            memory: 8Gi
        env:
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        - name: STRATEGY_VAD
          value: "least_response_time"
        - name: STRATEGY_ASR
          value: "resource_based"
        - name: STRATEGY_LLM
          value: "intelligent"
        - name: STRATEGY_TTS
          value: "least_connections"
        - name: HEALTH_CHECK_INTERVAL
          value: "10"
        - name: CIRCUIT_BREAKER_THRESHOLD
          value: "5"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: intelligent-load-balancer
  namespace: xiaozhi-system
spec:
  selector:
    app: intelligent-load-balancer
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
EOF

    kubectl apply -f k8s/optimized/intelligent-load-balancer.yaml
    log_success "智能负载均衡器部署完成"
}

# 配置监控和告警
setup_monitoring() {
    log_info "配置监控和告警..."
    
    # 部署Prometheus配置
    cat > k8s/optimized/prometheus-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
      evaluation_interval: 15s
    
    rule_files:
    - "/etc/prometheus/rules/*.yml"
    
    alerting:
      alertmanagers:
      - static_configs:
        - targets:
          - alertmanager:9093
    
    scrape_configs:
    - job_name: 'xiaozhi-services'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - xiaozhi-system
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: (vad-service|asr-service|llm-service|tts-service|intelligent-load-balancer)
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
        target_label: __address__
    
    - job_name: 'redis-cluster'
      static_configs:
      - targets: ['redis-cluster:6379']
      metrics_path: /metrics
    
    - job_name: 'kubernetes-nodes'
      kubernetes_sd_configs:
      - role: node
      relabel_configs:
      - action: labelmap
        regex: __meta_kubernetes_node_label_(.+)
EOF

    # 部署告警规则
    cat > k8s/optimized/alert-rules.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: alert-rules
  namespace: monitoring
data:
  xiaozhi-alerts.yml: |
    groups:
    - name: xiaozhi-performance
      rules:
      - alert: HighCPUUsage
        expr: rate(container_cpu_usage_seconds_total[5m]) * 100 > 80
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% for more than 2 minutes"
      
      - alert: HighMemoryUsage
        expr: (container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100 > 85
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 85% for more than 2 minutes"
      
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is above 1 second"
      
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "Service {{ $labels.instance }} is down"
      
      - alert: RedisConnectionsHigh
        expr: redis_connected_clients > 1000
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High Redis connections"
          description: "Redis has more than 1000 connections"
EOF

    kubectl apply -f k8s/optimized/prometheus-config.yaml
    kubectl apply -f k8s/optimized/alert-rules.yaml
    
    log_success "监控和告警配置完成"
}

# 性能测试
run_performance_test() {
    log_info "运行性能测试..."
    
    cat > scripts/performance-test-100-devices.py << 'EOF'
#!/usr/bin/env python3
"""
100台设备性能测试脚本
"""

import asyncio
import aiohttp
import time
import json
import base64
import random
from concurrent.futures import ThreadPoolExecutor
import statistics

class PerformanceTest:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.results = {
            'vad': [],
            'asr': [],
            'llm': [],
            'tts': []
        }
    
    async def test_vad_service(self, session, device_id):
        """测试VAD服务"""
        # 生成模拟音频数据
        audio_data = base64.b64encode(b"fake_audio_data" * 100).decode()
        
        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}/vad/detect", json={
                "session_id": f"device_{device_id}",
                "audio_data": audio_data,
                "sample_rate": 16000
            }) as response:
                result = await response.json()
                end_time = time.time()
                
                self.results['vad'].append({
                    'device_id': device_id,
                    'response_time': end_time - start_time,
                    'success': response.status == 200,
                    'timestamp': start_time
                })
        except Exception as e:
            self.results['vad'].append({
                'device_id': device_id,
                'response_time': -1,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            })
    
    async def test_asr_service(self, session, device_id):
        """测试ASR服务"""
        audio_data = base64.b64encode(b"fake_speech_data" * 200).decode()
        
        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}/asr/recognize", json={
                "session_id": f"device_{device_id}",
                "audio_data": audio_data,
                "language": "zh",
                "priority": random.randint(1, 3)
            }) as response:
                result = await response.json()
                end_time = time.time()
                
                self.results['asr'].append({
                    'device_id': device_id,
                    'response_time': end_time - start_time,
                    'success': response.status == 200,
                    'timestamp': start_time
                })
        except Exception as e:
            self.results['asr'].append({
                'device_id': device_id,
                'response_time': -1,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            })
    
    async def test_llm_service(self, session, device_id):
        """测试LLM服务"""
        messages = [
            {"role": "user", "content": f"你好，我是设备{device_id}，请介绍一下你自己。"}
        ]
        
        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}/llm/chat", json={
                "session_id": f"device_{device_id}",
                "messages": messages,
                "max_tokens": 100,
                "temperature": 0.7
            }) as response:
                result = await response.json()
                end_time = time.time()
                
                self.results['llm'].append({
                    'device_id': device_id,
                    'response_time': end_time - start_time,
                    'success': response.status == 200,
                    'timestamp': start_time
                })
        except Exception as e:
            self.results['llm'].append({
                'device_id': device_id,
                'response_time': -1,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            })
    
    async def test_tts_service(self, session, device_id):
        """测试TTS服务"""
        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}/tts/synthesize", json={
                "session_id": f"device_{device_id}",
                "text": f"设备{device_id}的语音合成测试",
                "voice_id": "zh-CN-XiaoxiaoNeural",
                "speed": 1.0
            }) as response:
                result = await response.json()
                end_time = time.time()
                
                self.results['tts'].append({
                    'device_id': device_id,
                    'response_time': end_time - start_time,
                    'success': response.status == 200,
                    'timestamp': start_time
                })
        except Exception as e:
            self.results['tts'].append({
                'device_id': device_id,
                'response_time': -1,
                'success': False,
                'error': str(e),
                'timestamp': start_time
            })
    
    async def simulate_device(self, session, device_id):
        """模拟单个设备的完整流程"""
        # VAD -> ASR -> LLM -> TTS
        await self.test_vad_service(session, device_id)
        await asyncio.sleep(0.1)  # 模拟设备间隔
        
        await self.test_asr_service(session, device_id)
        await asyncio.sleep(0.1)
        
        await self.test_llm_service(session, device_id)
        await asyncio.sleep(0.1)
        
        await self.test_tts_service(session, device_id)
    
    async def run_test(self, num_devices=100, concurrent_devices=20):
        """运行性能测试"""
        print(f"开始测试 {num_devices} 台设备，并发数: {concurrent_devices}")
        
        connector = aiohttp.TCPConnector(limit=concurrent_devices * 2)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # 分批测试
            for batch_start in range(0, num_devices, concurrent_devices):
                batch_end = min(batch_start + concurrent_devices, num_devices)
                batch_devices = range(batch_start, batch_end)
                
                print(f"测试设备 {batch_start} - {batch_end-1}")
                
                tasks = [
                    self.simulate_device(session, device_id)
                    for device_id in batch_devices
                ]
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # 批次间隔
                await asyncio.sleep(1)
        
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*50)
        print("性能测试报告")
        print("="*50)
        
        for service, results in self.results.items():
            if not results:
                continue
            
            successful_results = [r for r in results if r['success']]
            response_times = [r['response_time'] for r in successful_results]
            
            if response_times:
                print(f"\n{service.upper()} 服务:")
                print(f"  总请求数: {len(results)}")
                print(f"  成功请求数: {len(successful_results)}")
                print(f"  成功率: {len(successful_results)/len(results)*100:.2f}%")
                print(f"  平均响应时间: {statistics.mean(response_times):.3f}s")
                print(f"  中位数响应时间: {statistics.median(response_times):.3f}s")
                print(f"  95%响应时间: {sorted(response_times)[int(len(response_times)*0.95)]:.3f}s")
                print(f"  最大响应时间: {max(response_times):.3f}s")
                print(f"  最小响应时间: {min(response_times):.3f}s")
            else:
                print(f"\n{service.upper()} 服务: 所有请求失败")

if __name__ == "__main__":
    test = PerformanceTest()
    asyncio.run(test.run_test(num_devices=100, concurrent_devices=20))
EOF

    chmod +x scripts/performance-test-100-devices.py
    log_success "性能测试脚本创建完成"
}

# 主函数
main() {
    echo "Xiaozhi ESP32 Server - 100台设备优化部署"
    echo "============================================"
    
    # 检查依赖
    check_dependencies
    
    # 选择要执行的操作
    echo ""
    echo "请选择要执行的操作:"
    echo "1) 完整优化部署 (推荐)"
    echo "2) 仅优化Docker镜像"
    echo "3) 仅部署Kubernetes配置"
    echo "4) 仅部署Redis优化"
    echo "5) 仅部署负载均衡器"
    echo "6) 仅配置监控"
    echo "7) 仅运行性能测试"
    echo "8) 退出"
    
    read -p "请输入选择 (1-8): " choice
    
    case $choice in
        1)
            log_info "开始完整优化部署..."
            optimize_docker_images
            create_optimized_k8s_configs
            deploy_redis_optimization
            deploy_intelligent_load_balancer
            setup_monitoring
            run_performance_test
            log_success "完整优化部署完成！"
            ;;
        2)
            optimize_docker_images
            ;;
        3)
            create_optimized_k8s_configs
            kubectl apply -f k8s/optimized/
            ;;
        4)
            deploy_redis_optimization
            ;;
        5)
            deploy_intelligent_load_balancer
            ;;
        6)
            setup_monitoring
            ;;
        7)
            run_performance_test
            ;;
        8)
            log_info "退出部署脚本"
            exit 0
            ;;
        *)
            log_error "无效选择，请重新运行脚本"
            exit 1
            ;;
    esac
    
    echo ""
    log_success "操作完成！"
    echo ""
    echo "后续步骤:"
    echo "1. 检查所有Pod状态: kubectl get pods -n xiaozhi-system"
    echo "2. 查看服务状态: kubectl get svc -n xiaozhi-system"
    echo "3. 监控性能指标: kubectl port-forward -n monitoring svc/grafana 3000:3000"
    echo "4. 运行性能测试: python3 scripts/performance-test-100-devices.py"
}

# 执行主函数
main "$@"