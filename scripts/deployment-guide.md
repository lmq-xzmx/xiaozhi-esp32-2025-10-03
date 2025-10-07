# Xiaozhi ESP32 Server 优化部署指南

## 📋 部署前准备

### 系统要求
- Kubernetes 集群 (v1.20+)
- Docker (v20.10+)
- Python 3.8+
- 至少 32GB 内存，16核 CPU
- GPU 支持 (推荐 NVIDIA V100/A100)

### 依赖安装
```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 Kubernetes 工具
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# 安装 Helm
curl https://get.helm.sh/helm-v3.12.0-linux-amd64.tar.gz | tar xz
sudo mv linux-amd64/helm /usr/local/bin/
```

## 🚀 部署步骤

### 1. 执行优化部署
```bash
# 进入项目目录
cd /root/xiaozhi-server

# 给脚本执行权限
chmod +x scripts/optimize-for-100-devices.sh
chmod +x scripts/monitoring-setup.sh

# 执行完整优化部署
./scripts/optimize-for-100-devices.sh
# 选择选项 7: 执行完整优化部署

# 部署监控系统
./scripts/monitoring-setup.sh
# 选择选项 5: 部署完整监控栈
```

### 2. 验证部署状态
```bash
# 运行部署验证脚本
python scripts/deployment-validator.py \
    --output-json validation-results.json \
    --output-text validation-report.txt

# 查看验证结果
cat validation-report.txt
```

### 3. 执行组件评估
```bash
# 运行组件性能评估
python scripts/component-evaluator.py \
    --url http://your-xiaozhi-server.com \
    --config optimization-configs.yaml \
    --duration 300 \
    --components vad,asr,llm,tts \
    --output evaluation-results.json \
    --charts-dir ./charts

# 查看评估结果
cat evaluation-results.json
```

### 4. 执行性能测试
```bash
# 运行负载测试
python scripts/performance-test.py \
    --url http://your-xiaozhi-server.com \
    --devices 50 \
    --duration 600 \
    --interval 2.0 \
    --output performance-results.json

# 查看性能报告
cat performance_report.txt
```

## 📊 监控和告警

### 访问监控面板
```bash
# 获取 Grafana 访问地址
kubectl get svc grafana -n monitoring

# 获取 Grafana 密码
kubectl get secret grafana-admin -n monitoring -o jsonpath="{.data.password}" | base64 -d

# 访问 Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring

# 访问 Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring
```

### 关键监控指标
- **VAD 组件**: 响应时间 < 300ms, 吞吐量 > 35 QPS
- **ASR 组件**: 响应时间 < 2000ms, 吞吐量 > 12 QPS  
- **LLM 组件**: 响应时间 < 3000ms, 吞吐量 > 6 QPS
- **TTS 组件**: 响应时间 < 1200ms, 吞吐量 > 25 QPS
- **系统资源**: CPU < 80%, 内存 < 85%, GPU < 90%

## 🔧 故障排查

### 常见问题解决

#### 1. Pod 启动失败
```bash
# 查看 Pod 状态
kubectl get pods -n xiaozhi-system

# 查看 Pod 日志
kubectl logs <pod-name> -n xiaozhi-system

# 查看 Pod 事件
kubectl describe pod <pod-name> -n xiaozhi-system
```

#### 2. 服务不可访问
```bash
# 检查服务状态
kubectl get svc -n xiaozhi-system

# 检查端点
kubectl get endpoints -n xiaozhi-system

# 测试服务连通性
kubectl run test-pod --image=busybox --rm -it -- wget -qO- http://service-name:port/health
```

#### 3. 性能问题诊断
```bash
# 查看资源使用情况
kubectl top nodes
kubectl top pods -n xiaozhi-system

# 查看 HPA 状态
kubectl get hpa -n xiaozhi-system

# 查看 PVC 状态
kubectl get pvc -n xiaozhi-system
```

#### 4. Redis 集群问题
```bash
# 检查 Redis 集群状态
kubectl exec -it redis-cluster-0 -n xiaozhi-system -- redis-cli cluster nodes

# 检查 Redis 内存使用
kubectl exec -it redis-cluster-0 -n xiaozhi-system -- redis-cli info memory
```

## 📈 性能调优

### 根据监控数据调优

#### VAD 组件调优
```bash
# 如果响应时间过高，增加副本数
kubectl scale deployment vad-service --replicas=6 -n xiaozhi-system

# 如果内存使用过高，调整内存限制
kubectl patch deployment vad-service -n xiaozhi-system -p '{"spec":{"template":{"spec":{"containers":[{"name":"vad","resources":{"limits":{"memory":"6Gi"}}}]}}}}'
```

#### ASR 组件调优
```bash
# 如果 GPU 使用率低，增加 GPU 工作进程
kubectl set env deployment/asr-service GPU_WORKERS=6 -n xiaozhi-system

# 如果批处理效率低，调整批大小
kubectl set env deployment/asr-service BATCH_SIZE=64 -n xiaozhi-system
```

#### LLM 组件调优
```bash
# 如果缓存命中率低，增加缓存大小
kubectl set env deployment/llm-service CACHE_SIZE=10000 -n xiaozhi-system

# 如果响应时间过高，启用更多本地模型实例
kubectl scale deployment llm-local --replicas=4 -n xiaozhi-system
```

#### TTS 组件调优
```bash
# 如果音频质量问题，调整编码参数
kubectl set env deployment/tts-service OPUS_BITRATE=64000 -n xiaozhi-system

# 如果缓存占用过高，调整缓存策略
kubectl set env deployment/tts-service CACHE_TTL=3600 -n xiaozhi-system
```

## 🔄 滚动更新

### 安全更新流程
```bash
# 1. 备份当前配置
kubectl get all -n xiaozhi-system -o yaml > backup-$(date +%Y%m%d).yaml

# 2. 更新镜像
kubectl set image deployment/vad-service vad=xiaozhi/vad:v2.0 -n xiaozhi-system

# 3. 监控更新状态
kubectl rollout status deployment/vad-service -n xiaozhi-system

# 4. 如需回滚
kubectl rollout undo deployment/vad-service -n xiaozhi-system
```

## 📋 定期维护

### 每日检查
```bash
# 运行健康检查
python scripts/deployment-validator.py --quick-check

# 检查关键指标
kubectl top nodes
kubectl get pods -n xiaozhi-system | grep -v Running
```

### 每周检查
```bash
# 完整性能评估
python scripts/component-evaluator.py --full-evaluation

# 清理旧日志
kubectl delete pods -l app=log-cleaner -n xiaozhi-system
```

### 每月检查
```bash
# 完整负载测试
python scripts/performance-test.py --devices 100 --duration 1800

# 更新优化配置
./scripts/optimize-for-100-devices.sh
```

## 🎯 扩容到 1000 台设备

### 边缘计算部署准备
```bash
# 1. 部署边缘节点
kubectl apply -f configs/edge-computing/

# 2. 配置边缘模型
python scripts/model-optimization.py --target edge --quantization int8

# 3. 测试边缘性能
python scripts/edge-performance-test.py --nodes 10 --devices-per-node 100
```

## 📞 技术支持

如遇到问题，请按以下顺序排查：
1. 查看本指南的故障排查部分
2. 检查监控面板的告警信息
3. 运行 `deployment-validator.py` 获取详细诊断
4. 查看相关组件的日志文件

---

**注意**: 本指南基于 Kubernetes 环境，如使用其他容器编排工具，请相应调整命令。