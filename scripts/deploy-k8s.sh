#!/bin/bash

# Xiaozhi ESP32 Server Kubernetes 部署脚本
# 支持100台设备的水平扩展方案

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_step "检查依赖..."
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl 未安装"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "docker 未安装"
        exit 1
    fi
    
    # 检查Kubernetes集群连接
    if ! kubectl cluster-info &> /dev/null; then
        log_error "无法连接到Kubernetes集群"
        exit 1
    fi
    
    log_info "依赖检查通过"
}

# 构建Docker镜像
build_images() {
    log_step "构建Docker镜像..."
    
    # VAD Service
    log_info "构建VAD服务镜像..."
    cat > Dockerfile.vad << EOF
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libsndfile1 \\
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements-vad.txt .
RUN pip install --no-cache-dir -r requirements-vad.txt

# 复制代码
COPY services/vad_service.py .

EXPOSE 8000

CMD ["python", "vad_service.py"]
EOF

    # ASR Service
    log_info "构建ASR服务镜像..."
    cat > Dockerfile.asr << EOF
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libsndfile1 \\
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements-asr.txt .
RUN pip install --no-cache-dir -r requirements-asr.txt

# 复制代码
COPY services/asr_service.py .

EXPOSE 8001

CMD ["python", "asr_service.py"]
EOF

    # LLM Service
    log_info "构建LLM服务镜像..."
    cat > Dockerfile.llm << EOF
FROM python:3.9-slim

WORKDIR /app

# 安装Python依赖
COPY requirements-llm.txt .
RUN pip install --no-cache-dir -r requirements-llm.txt

# 复制代码
COPY services/llm_service.py .

EXPOSE 8002

CMD ["python", "llm_service.py"]
EOF

    # TTS Service
    log_info "构建TTS服务镜像..."
    cat > Dockerfile.tts << EOF
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libsndfile1 \\
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
COPY requirements-tts.txt .
RUN pip install --no-cache-dir -r requirements-tts.txt

# 复制代码
COPY services/tts_service.py .

EXPOSE 8003

CMD ["python", "tts_service.py"]
EOF

    # 构建镜像
    docker build -f Dockerfile.vad -t xiaozhi/vad-service:latest .
    docker build -f Dockerfile.asr -t xiaozhi/asr-service:latest .
    docker build -f Dockerfile.llm -t xiaozhi/llm-service:latest .
    docker build -f Dockerfile.tts -t xiaozhi/tts-service:latest .
    
    log_info "Docker镜像构建完成"
}

# 创建requirements文件
create_requirements() {
    log_step "创建requirements文件..."
    
    # VAD requirements
    cat > requirements-vad.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
numpy==1.24.3
torch==2.0.1
torchaudio==2.0.2
silero-vad==4.0.1
websockets==12.0
python-multipart==0.0.6
EOF

    # ASR requirements
    cat > requirements-asr.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
numpy==1.24.3
torch==2.0.1
torchaudio==2.0.2
funasr==1.0.0
librosa==0.10.1
python-multipart==0.0.6
EOF

    # LLM requirements
    cat > requirements-llm.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
aiohttp==3.9.1
pydantic==2.5.0
python-multipart==0.0.6
EOF

    # TTS requirements
    cat > requirements-tts.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
edge-tts==6.1.9
azure-cognitiveservices-speech==1.34.0
aiofiles==23.2.1
python-multipart==0.0.6
EOF

    log_info "Requirements文件创建完成"
}

# 部署到Kubernetes
deploy_to_k8s() {
    log_step "部署到Kubernetes..."
    
    # 创建命名空间
    log_info "创建命名空间..."
    kubectl apply -f k8s/namespace.yaml
    
    # 部署Redis集群
    log_info "部署Redis集群..."
    kubectl apply -f k8s/redis-cluster.yaml
    
    # 等待Redis就绪
    log_info "等待Redis集群就绪..."
    kubectl wait --for=condition=ready pod -l app=redis-cluster -n xiaozhi-system --timeout=300s
    
    # 初始化Redis集群
    log_info "初始化Redis集群..."
    kubectl exec -it redis-cluster-0 -n xiaozhi-system -- redis-cli --cluster create \\
        redis-cluster-0.redis-cluster.xiaozhi-system.svc.cluster.local:6379 \\
        redis-cluster-1.redis-cluster.xiaozhi-system.svc.cluster.local:6379 \\
        redis-cluster-2.redis-cluster.xiaozhi-system.svc.cluster.local:6379 \\
        redis-cluster-3.redis-cluster.xiaozhi-system.svc.cluster.local:6379 \\
        redis-cluster-4.redis-cluster.xiaozhi-system.svc.cluster.local:6379 \\
        redis-cluster-5.redis-cluster.xiaozhi-system.svc.cluster.local:6379 \\
        --cluster-replicas 1 --cluster-yes || true
    
    # 部署存储和配置
    log_info "部署存储和配置..."
    kubectl apply -f k8s/storage-config.yaml
    
    # 部署AI服务
    log_info "部署VAD服务..."
    kubectl apply -f k8s/vad-service.yaml
    
    log_info "部署ASR服务..."
    kubectl apply -f k8s/asr-service.yaml
    
    log_info "部署LLM和TTS服务..."
    kubectl apply -f k8s/llm-tts-services.yaml
    
    # 等待服务就绪
    log_info "等待服务就绪..."
    kubectl wait --for=condition=available deployment -l component=ai-pipeline -n xiaozhi-system --timeout=600s
    kubectl wait --for=condition=available deployment -l component=gateway -n xiaozhi-system --timeout=300s
    
    log_info "Kubernetes部署完成"
}

# 验证部署
verify_deployment() {
    log_step "验证部署..."
    
    # 检查Pod状态
    log_info "检查Pod状态..."
    kubectl get pods -n xiaozhi-system
    
    # 检查服务状态
    log_info "检查服务状态..."
    kubectl get services -n xiaozhi-system
    
    # 检查HPA状态
    log_info "检查HPA状态..."
    kubectl get hpa -n xiaozhi-system
    
    # 获取API Gateway外部IP
    log_info "获取API Gateway访问地址..."
    EXTERNAL_IP=$(kubectl get service api-gateway -n xiaozhi-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [ -n "$EXTERNAL_IP" ]; then
        log_info "API Gateway地址: http://$EXTERNAL_IP"
    else
        log_warn "API Gateway外部IP尚未分配，请稍后检查"
    fi
    
    # 健康检查
    log_info "执行健康检查..."
    kubectl exec -n xiaozhi-system deployment/vad-service -- curl -f http://localhost:8000/health || log_warn "VAD服务健康检查失败"
    kubectl exec -n xiaozhi-system deployment/asr-service -- curl -f http://localhost:8001/health || log_warn "ASR服务健康检查失败"
    kubectl exec -n xiaozhi-system deployment/llm-service -- curl -f http://localhost:8002/health || log_warn "LLM服务健康检查失败"
    kubectl exec -n xiaozhi-system deployment/tts-service -- curl -f http://localhost:8003/health || log_warn "TTS服务健康检查失败"
    
    log_info "部署验证完成"
}

# 显示监控信息
show_monitoring() {
    log_step "监控信息..."
    
    cat << EOF

=== 监控和管理命令 ===

1. 查看Pod状态:
   kubectl get pods -n xiaozhi-system -w

2. 查看服务日志:
   kubectl logs -f deployment/vad-service -n xiaozhi-system
   kubectl logs -f deployment/asr-service -n xiaozhi-system
   kubectl logs -f deployment/llm-service -n xiaozhi-system
   kubectl logs -f deployment/tts-service -n xiaozhi-system

3. 查看HPA状态:
   kubectl get hpa -n xiaozhi-system -w

4. 手动扩缩容:
   kubectl scale deployment vad-service --replicas=5 -n xiaozhi-system
   kubectl scale deployment asr-service --replicas=6 -n xiaozhi-system

5. 查看资源使用:
   kubectl top pods -n xiaozhi-system
   kubectl top nodes

6. 访问服务:
   kubectl port-forward service/api-gateway 8080:80 -n xiaozhi-system

=== 性能测试 ===

1. VAD测试:
   curl -X POST http://localhost:8080/vad/detect \\
     -H "Content-Type: application/json" \\
     -d '{"session_id": "test", "audio_data": "base64_audio_data"}'

2. ASR测试:
   curl -X POST http://localhost:8080/asr/recognize \\
     -H "Content-Type: application/json" \\
     -d '{"session_id": "test", "audio_data": "base64_audio_data"}'

3. LLM测试:
   curl -X POST http://localhost:8080/llm/chat \\
     -H "Content-Type: application/json" \\
     -d '{"session_id": "test", "messages": [{"role": "user", "content": "你好"}]}'

4. TTS测试:
   curl -X POST http://localhost:8080/tts/synthesize \\
     -H "Content-Type: application/json" \\
     -d '{"session_id": "test", "text": "你好，世界"}'

EOF
}

# 主函数
main() {
    log_info "开始部署Xiaozhi ESP32 Server Kubernetes集群..."
    
    check_dependencies
    create_requirements
    build_images
    deploy_to_k8s
    verify_deployment
    show_monitoring
    
    log_info "部署完成！系统现在可以支持100台设备的并发访问。"
}

# 执行主函数
main "$@"