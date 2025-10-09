# ASR优化部署指南

## 🎯 概述

本指南提供了在4核8GB服务器上部署优化ASR服务的完整步骤，支持20-25个设备的并发语音识别请求。

## 📋 前置要求

### 硬件要求
- **CPU**: 4核心或以上
- **内存**: 8GB或以上
- **存储**: 至少20GB可用空间
- **网络**: 稳定的网络连接

### 软件要求
- **操作系统**: Linux (推荐Ubuntu 20.04+)
- **Docker**: 20.10+
- **Docker Compose**: 1.29+
- **Python**: 3.9+

## 🚀 快速部署

### 1. 停止现有服务
```bash
# 停止当前运行的服务
docker-compose down

# 清理容器和网络
docker system prune -f
```

### 2. 应用优化配置
```bash
# 确保配置文件存在
ls -la asr_memory_optimized_4core_8gb.yaml
ls -la asr_docker_optimized_4core_8gb.yaml

# 启动优化后的服务
docker-compose up -d
```

### 3. 验证部署
```bash
# 检查容器状态
docker ps

# 验证ASR服务
python simple_asr_test.py

# 启动性能监控
python asr_performance_monitor.py --interval 30
```

## 📁 配置文件说明

### 核心配置文件

#### 1. `asr_memory_optimized_4core_8gb.yaml`
ASR内存和缓存优化配置：
- 内存限制: 1500MB
- 缓存大小: 512MB
- 并发限制: 20个请求
- 批处理大小: 8个请求

#### 2. `asr_docker_optimized_4core_8gb.yaml`
Docker容器资源优化配置：
- CPU限制: 2.5核心
- 内存限制: 4GB
- CPU绑定: 核心0-2
- 共享内存: 2GB

#### 3. `docker-compose.yml`
更新的Docker Compose配置，包含：
- ASR服务环境变量
- Redis缓存优化
- 资源限制和预留
- 健康检查配置

## 🔧 详细配置步骤

### 1. 环境变量配置

在`docker-compose.yml`中已配置以下ASR优化环境变量：

```yaml
environment:
  # ASR内存优化
  - ASR_MAX_MEMORY_MB=1500
  - ASR_ENABLE_MEMORY_POOL=true
  - ASR_CACHE_SIZE_MB=512
  
  # ASR并发优化
  - ASR_MAX_CONCURRENT=20
  - ASR_BATCH_SIZE=8
  - ASR_WORKER_THREADS=4
  
  # ASR音频处理优化
  - ASR_CHUNK_SIZE=640
  - ASR_OVERLAP_SIZE=64
  - ASR_QUEUE_SIZE=50
```

### 2. Redis缓存配置

Redis服务已优化为ASR缓存：

```yaml
xiaozhi-esp32-server-redis:
  command: >
    redis-server
    --maxmemory 512mb
    --maxmemory-policy allkeys-lru
    --hash-max-ziplist-entries 512
    --hash-max-ziplist-value 64
    --list-max-ziplist-size -2
    --set-max-intset-entries 512
```

### 3. 资源限制配置

各服务的资源分配：

```yaml
# ASR主服务
deploy:
  resources:
    limits:
      cpus: '2.5'
      memory: 4G
    reservations:
      cpus: '1.5'
      memory: 2G

# Redis缓存
deploy:
  resources:
    limits:
      cpus: '0.5'
      memory: 768M
    reservations:
      cpus: '0.2'
      memory: 512M
```

## 📊 性能监控

### 1. 启动监控服务

```bash
# 后台运行监控（无仪表板）
python asr_performance_monitor.py --interval 30 --no-dashboard &

# 前台运行监控（带仪表板）
python asr_performance_monitor.py --interval 10
```

### 2. 监控指标

监控脚本会跟踪以下关键指标：

- **响应时间**: 平均ASR处理时间
- **缓存命中率**: 缓存效果评估
- **并发连接数**: 当前活跃连接
- **系统资源**: CPU、内存、磁盘使用
- **错误率**: 请求失败比例

### 3. 告警阈值

默认告警阈值：
- 响应时间 > 1.0秒
- 缓存命中率 < 10%
- 内存使用 > 1400MB
- CPU使用率 > 90%

## 🧪 测试验证

### 1. 快速功能测试

```bash
# 基础功能测试
python quick_asr_test.py

# 完整优化效果测试
python simple_asr_test.py

# 性能压力测试
python test_asr_optimization.py
```

### 2. 健康检查

```bash
# 检查服务健康状态
curl http://localhost:8001/health

# 获取服务统计信息
curl http://localhost:8001/asr/stats

# 检查容器状态
docker ps
docker stats
```

### 3. 预期性能指标

优化后的性能基准：
- **单次请求延迟**: ~0.24秒
- **缓存命中延迟**: ~0.01秒
- **并发吞吐量**: 30+ req/s
- **内存使用**: <1500MB
- **缓存命中率**: >10%

## 🔍 故障排除

### 常见问题

#### 1. 服务启动失败
```bash
# 检查容器日志
docker-compose logs xiaozhi-esp32-server

# 检查资源使用
docker stats

# 重启服务
docker-compose restart xiaozhi-esp32-server
```

#### 2. 内存使用过高
```bash
# 检查内存配置
grep -r "ASR_MAX_MEMORY_MB" docker-compose.yml

# 调整内存限制
# 编辑docker-compose.yml中的ASR_MAX_MEMORY_MB值

# 重启服务
docker-compose restart xiaozhi-esp32-server
```

#### 3. 响应时间过长
```bash
# 检查并发设置
curl http://localhost:8001/asr/stats

# 调整并发参数
# 编辑docker-compose.yml中的ASR_MAX_CONCURRENT值

# 重启服务
docker-compose restart xiaozhi-esp32-server
```

#### 4. 缓存效果差
```bash
# 检查Redis状态
docker-compose logs xiaozhi-esp32-server-redis

# 检查缓存配置
curl http://localhost:8001/asr/stats

# 重启Redis
docker-compose restart xiaozhi-esp32-server-redis
```

### 日志查看

```bash
# ASR服务日志
docker-compose logs -f xiaozhi-esp32-server

# Redis日志
docker-compose logs -f xiaozhi-esp32-server-redis

# 监控日志
tail -f logs/asr_monitor.log

# 系统资源监控
htop
```

## 📈 性能调优

### 1. 根据负载调整

#### 高负载场景 (>15个并发设备)
```yaml
environment:
  - ASR_MAX_CONCURRENT=25
  - ASR_BATCH_SIZE=10
  - ASR_WORKER_THREADS=6
  - ASR_CACHE_SIZE_MB=768
```

#### 低负载场景 (<10个并发设备)
```yaml
environment:
  - ASR_MAX_CONCURRENT=15
  - ASR_BATCH_SIZE=6
  - ASR_WORKER_THREADS=3
  - ASR_CACHE_SIZE_MB=256
```

### 2. 内存优化

```yaml
# 内存紧张时
environment:
  - ASR_MAX_MEMORY_MB=1200
  - ASR_CACHE_SIZE_MB=384
  - ASR_ENABLE_AGGRESSIVE_GC=true
```

### 3. 网络优化

```yaml
# 网络延迟敏感场景
environment:
  - ASR_CHUNK_SIZE=320
  - ASR_OVERLAP_SIZE=32
  - ASR_NETWORK_TIMEOUT=5
```

## 🔄 升级和维护

### 1. 定期维护

```bash
# 每周清理Docker缓存
docker system prune -f

# 每月重启服务
docker-compose restart

# 检查磁盘空间
df -h

# 清理日志文件
find logs/ -name "*.log" -mtime +30 -delete
```

### 2. 配置备份

```bash
# 备份配置文件
tar -czf asr_config_backup_$(date +%Y%m%d).tar.gz \
  asr_memory_optimized_4core_8gb.yaml \
  asr_docker_optimized_4core_8gb.yaml \
  docker-compose.yml

# 备份到远程
scp asr_config_backup_*.tar.gz user@backup-server:/backups/
```

### 3. 监控数据分析

```bash
# 分析监控日志
grep "告警" logs/asr_monitor.log | tail -20

# 性能趋势分析
python -c "
import json
with open('logs/asr_monitor.log', 'r') as f:
    lines = f.readlines()
    # 分析性能趋势
"
```

## 📞 支持和联系

如果在部署过程中遇到问题：

1. 查看本指南的故障排除部分
2. 检查日志文件获取详细错误信息
3. 参考性能监控数据分析问题
4. 根据实际负载调整配置参数

## 📝 部署检查清单

- [ ] 硬件要求满足（4核8GB+）
- [ ] Docker和Docker Compose已安装
- [ ] 配置文件已准备就绪
- [ ] 停止现有服务
- [ ] 应用新配置启动服务
- [ ] 验证服务健康状态
- [ ] 运行性能测试
- [ ] 启动监控服务
- [ ] 配置告警阈值
- [ ] 备份配置文件
- [ ] 文档化部署信息

---

**部署指南版本**: v1.0  
**适用环境**: 4核8GB服务器  
**支持设备数**: 20-25个设备  
**更新时间**: 2024年12月