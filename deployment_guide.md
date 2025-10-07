# 🚀 优化配置部署指南

## 📋 **部署前检查清单**

### **1. 备份当前配置**
```bash
# 备份当前docker-compose文件
cp docker-compose_optimized.yml docker-compose_optimized_backup.yml

# 备份当前配置文件
cp -r data data_backup_$(date +%Y%m%d_%H%M%S)
```

### **2. 确认系统资源**
```bash
# 检查内存
free -h
# 确保可用内存 > 2GB

# 检查CPU
lscpu | grep "CPU(s)"
# 确认4核CPU

# 检查磁盘空间
df -h
# 确保可用空间 > 10GB
```

---

## 🔧 **部署步骤**

### **第一步：应用新的Docker配置**

```bash
# 1. 停止当前服务
docker-compose -f docker-compose_optimized.yml down

# 2. 应用新配置
docker-compose -f docker-compose-optimized-for-your-server.yml up -d

# 3. 检查服务状态
docker-compose -f docker-compose-optimized-for-your-server.yml ps
```

### **第二步：应用VAD优化配置**

```bash
# 复制VAD优化配置到容器
docker cp vad_optimized_for_your_server.yaml xiaozhi-esp32-server:/opt/xiaozhi-esp32-server/config/

# 重启主服务以应用配置
docker-compose -f docker-compose-optimized-for-your-server.yml restart xiaozhi-esp32-server
```

### **第三步：应用ASR优化配置**

```bash
# 复制ASR优化配置到容器
docker cp asr_streaming_optimized_for_your_server.yaml xiaozhi-esp32-server:/opt/xiaozhi-esp32-server/config/

# 重启主服务以应用配置
docker-compose -f docker-compose-optimized-for-your-server.yml restart xiaozhi-esp32-server
```

### **第四步：应用MySQL优化配置**

```bash
# 复制MySQL配置到容器
docker cp mysql-optimized.cnf xiaozhi-esp32-server-db:/etc/mysql/conf.d/

# 重启数据库服务
docker-compose -f docker-compose-optimized-for-your-server.yml restart xiaozhi-esp32-server-db
```

---

## 📊 **部署后验证**

### **1. 服务健康检查**

```bash
# 检查所有服务状态
docker-compose -f docker-compose-optimized-for-your-server.yml ps

# 检查服务日志
docker-compose -f docker-compose-optimized-for-your-server.yml logs xiaozhi-esp32-server | tail -50

# 检查资源使用
docker stats --no-stream
```

### **2. 功能测试**

```bash
# 测试主服务API
curl -X GET http://localhost:8003/xiaozhi/health/

# 测试OTA接口
curl -X GET http://localhost:8003/xiaozhi/ota/

# 测试WebSocket连接
curl -X GET http://localhost:8003/xiaozhi/websocket/status/
```

### **3. 性能基线测试**

```bash
# 监控资源使用（运行5分钟）
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" > performance_baseline.log &
sleep 300
kill %1

# 查看基线性能
cat performance_baseline.log
```

---

## 🔍 **监控和调优**

### **1. 关键监控指标**

```bash
# 创建监控脚本
cat > monitor.sh << 'EOF'
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    
    # 系统资源
    echo "CPU使用率:"
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
    
    echo "内存使用率:"
    free | grep Mem | awk '{printf "%.1f%%\n", $3/$2 * 100.0}'
    
    # Docker容器资源
    echo "容器资源使用:"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemPerc}}"
    
    echo "---"
    sleep 60
done
EOF

chmod +x monitor.sh
```

### **2. 性能调优建议**

```yaml
如果内存使用率 > 85%:
  - 减少VAD批处理大小: 24 → 20
  - 减少ASR批处理大小: 8 → 6
  - 减少最大并发数: 36 → 30

如果CPU使用率 > 85%:
  - 减少工作线程数: 3 → 2
  - 增加批处理等待时间: 40ms → 60ms
  - 启用更激进的缓存策略

如果响应延迟 > 500ms:
  - 检查网络连接
  - 增加工作线程数
  - 减少批处理大小
```

---

## ⚠️ **故障排除**

### **1. 常见问题**

**问题1: 容器启动失败**
```bash
# 检查错误日志
docker-compose -f docker-compose-optimized-for-your-server.yml logs

# 检查端口占用
netstat -tulpn | grep :8003

# 重新构建镜像
docker-compose -f docker-compose-optimized-for-your-server.yml build --no-cache
```

**问题2: 内存不足**
```bash
# 检查内存使用
free -h
docker stats --no-stream

# 临时解决方案
docker-compose -f docker-compose-optimized-for-your-server.yml restart
```

**问题3: 性能下降**
```bash
# 检查系统负载
uptime
top -bn1

# 检查磁盘IO
iostat -x 1 5

# 检查网络连接
ss -tuln
```

### **2. 回滚方案**

```bash
# 如果新配置有问题，快速回滚
docker-compose -f docker-compose-optimized-for-your-server.yml down
docker-compose -f docker-compose_optimized_backup.yml up -d

# 恢复备份数据
rm -rf data
mv data_backup_* data
```

---

## 📈 **性能优化路线图**

### **短期优化（1-2周）**
- [x] VAD/ASR配置优化
- [x] 资源限制调整
- [x] 监控系统部署
- [ ] 性能基线建立
- [ ] 负载测试

### **中期优化（1-2月）**
- [ ] 模型量化进一步优化
- [ ] 缓存策略优化
- [ ] 数据库查询优化
- [ ] 网络连接池优化

### **长期规划（3-6月）**
- [ ] 硬件升级评估
- [ ] 集群部署方案
- [ ] GPU加速集成
- [ ] 自动扩缩容

---

## ✅ **部署完成确认**

部署完成后，请确认以下指标：

- [ ] 所有容器正常运行
- [ ] 内存使用率 < 80%
- [ ] CPU使用率 < 80%
- [ ] API响应正常
- [ ] WebSocket连接正常
- [ ] 支持设备数达到60-80台

**恭喜！您的服务器已成功优化，预期性能提升5倍！** 🎉