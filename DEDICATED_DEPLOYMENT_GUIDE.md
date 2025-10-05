# 小智ESP32专用服务器部署指南

## 概述

本指南适用于**4核3GHz + 7.5GB内存**的专用小智ESP32服务器，通过激进优化配置实现**支持3-4台设备稳定并发，彻底消除卡顿现象**。

## 系统要求

- **硬件**: 4核CPU + 7.5GB内存 + 100GB存储
- **操作系统**: Linux (推荐Ubuntu 20.04+)
- **Docker**: 20.10+
- **权限**: root用户权限

## 优化文件说明

### 1. 核心配置文件

| 文件名 | 用途 | 优化重点 |
|--------|------|----------|
| `docker-compose_dedicated.yml` | Docker容器配置 | 3.5核CPU + 6GB内存 + CPU亲和性 |
| `.config_dedicated.yaml` | SenseVoice配置 | 3并发会话 + 模型池 + 内存管理 |
| `sensevoice_dedicated.py` | ASR代码优化 | 模型池 + 内存池 + 并发控制 |
| `optimize_dedicated_server.sh` | 系统级优化 | 内核参数 + CPU调度 + 内存管理 |

### 2. 性能提升预期

| 指标 | 优化前 | 优化后 | 提升幅度 |
|------|--------|--------|----------|
| 并发设备数 | 1台(卡顿) | 3-4台(稳定) | **300-400%** |
| 首次响应时间 | 3-5秒 | 0.5-1秒 | **80-90%** |
| 系统负载 | 6.87 | <3.0 | **>50%** |
| 内存使用率 | 88% | <75% | **15%+** |
| CPU利用率 | 不均衡 | 均衡分配 | **显著改善** |

## 部署步骤

### 第一步: 系统级优化

```bash
# 1. 进入项目目录
cd /root/xiaozhi-server

# 2. 执行系统级优化（需要root权限）
sudo ./optimize_dedicated_server.sh

# 3. 重启系统以应用所有优化
sudo reboot
```

**系统优化包含:**
- 内核参数调优 (内存管理、CPU调度、网络优化)
- 系统限制提升 (文件描述符、进程数、内存锁定)
- CPU调度器优化 (性能模式、CPU亲和性)
- 内存管理优化 (交换分区、缓存策略)
- 磁盘I/O优化 (调度器、预读、队列深度)
- Docker配置优化 (日志、存储、限制)

### 第二步: 停止现有服务

```bash
# 停止当前运行的服务
cd /root/xiaozhi-server
docker-compose down

# 清理Docker资源
docker system prune -f
```

### 第三步: 部署优化配置

```bash
# 1. 备份原始配置
cp docker-compose.yml docker-compose.yml.backup
cp data/.config.yaml data/.config.yaml.backup

# 2. 使用专用服务器配置
cp docker-compose_dedicated.yml docker-compose.yml
cp data/.config_dedicated.yaml data/.config.yaml

# 3. 替换SenseVoice实现（如果需要）
# 注意：这需要根据实际代码结构进行调整
# cp sensevoice_dedicated.py path/to/sensevoice/implementation.py
```

### 第四步: 启动优化服务

```bash
# 启动专用服务器配置
docker-compose up -d

# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f xiaozhi-esp32-server
```

### 第五步: 验证优化效果

#### 5.1 系统监控

```bash
# 运行系统监控
/usr/local/bin/xiaozhi-monitor.sh

# 查看CPU亲和性设置
/usr/local/bin/xiaozhi-cpu-affinity.sh

# 检查内存优化
/usr/local/bin/xiaozhi-memory-optimize.sh
```

#### 5.2 性能测试

```bash
# 运行性能监控脚本
python3 monitor_performance.py --mode test --duration 300 --connections 3

# 查看实时性能指标
python3 monitor_performance.py --mode monitor --duration 60
```

#### 5.3 并发测试

```bash
# 测试3台设备并发
python3 monitor_performance.py --mode both --duration 600 --connections 3

# 测试4台设备并发（压力测试）
python3 monitor_performance.py --mode both --duration 300 --connections 4
```

## 监控和维护

### 自动监控

系统已配置自动监控任务:

```bash
# 查看监控日志
tail -f /var/log/xiaozhi-monitor.log

# 查看内存优化日志
tail -f /var/log/xiaozhi-memory.log

# 查看启动优化日志
tail -f /var/log/xiaozhi-startup.log
```

### 手动检查

```bash
# 检查Docker容器资源使用
docker stats

# 检查系统负载
htop

# 检查内存使用
free -h

# 检查磁盘I/O
iotop

# 检查网络连接
netstat -tulpn | grep :8000
```

### 性能调优

如果需要进一步调优:

```bash
# 调整并发会话数（在.config.yaml中）
max_concurrent_sessions: 2  # 降低到2
max_concurrent_sessions: 4  # 提高到4

# 调整Docker资源限制（在docker-compose.yml中）
cpus: '3.0'     # 降低CPU限制
cpus: '4.0'     # 提高CPU限制
memory: 5G      # 降低内存限制
memory: 7G      # 提高内存限制
```

## 故障排除

### 常见问题

#### 1. 服务启动失败

```bash
# 检查Docker日志
docker-compose logs xiaozhi-esp32-server

# 检查系统资源
free -h
df -h

# 检查端口占用
netstat -tulpn | grep :8000
```

#### 2. 性能不达预期

```bash
# 检查CPU亲和性是否生效
taskset -cp $(docker inspect xiaozhi-esp32-server --format '{{.State.Pid}}')

# 检查内核参数是否生效
sysctl vm.swappiness
sysctl net.ipv4.tcp_congestion_control

# 检查Docker资源限制
docker inspect xiaozhi-esp32-server | grep -A 10 "Resources"
```

#### 3. 内存使用过高

```bash
# 手动执行内存优化
/usr/local/bin/xiaozhi-memory-optimize.sh

# 检查内存泄漏
docker exec xiaozhi-esp32-server ps aux --sort=-%mem | head -10

# 重启服务释放内存
docker-compose restart xiaozhi-esp32-server
```

#### 4. 并发连接失败

```bash
# 检查连接数限制
ulimit -n

# 检查WebSocket连接
netstat -an | grep :8003 | wc -l

# 检查SenseVoice模型状态
docker exec xiaozhi-esp32-server ps aux | grep python
```

### 回滚方案

如果优化后出现问题，可以快速回滚:

```bash
# 停止服务
docker-compose down

# 恢复原始配置
cp docker-compose.yml.backup docker-compose.yml
cp data/.config.yaml.backup data/.config.yaml

# 恢复系统配置（从备份目录）
BACKUP_DIR=$(ls -1d /root/xiaozhi-server/backup/* | tail -1)
sudo cp $BACKUP_DIR/sysctl.conf.bak /etc/sysctl.conf
sudo cp $BACKUP_DIR/limits.conf.bak /etc/security/limits.conf

# 重启服务
docker-compose up -d

# 重启系统（如果需要）
sudo reboot
```

## 预期效果

### 性能指标

部署完成后，系统应达到以下性能指标:

- **并发设备**: 3台设备稳定运行，4台设备压力测试通过
- **响应延迟**: 首次识别 < 1秒，后续识别 < 0.5秒
- **系统负载**: 平均负载 < 3.0，峰值负载 < 4.0
- **内存使用**: 稳定在70-75%，峰值不超过80%
- **CPU使用**: 均衡分配，单核使用率 < 85%

### 稳定性指标

- **服务可用性**: 99.9%+
- **连接成功率**: 99%+
- **识别准确率**: 保持原有水平
- **系统稳定性**: 24小时连续运行无重启

## 维护建议

### 日常维护

1. **每日检查**: 运行监控脚本，查看系统状态
2. **每周清理**: 清理Docker日志和临时文件
3. **每月优化**: 执行内存整理和性能调优

### 定期更新

1. **配置调优**: 根据实际使用情况调整并发参数
2. **系统更新**: 定期更新系统和Docker版本
3. **性能测试**: 定期进行压力测试验证性能

### 监控告警

建议设置以下告警阈值:

- CPU使用率 > 80%
- 内存使用率 > 85%
- 系统负载 > 4.0
- 磁盘使用率 > 85%
- 服务响应时间 > 2秒

## 技术支持

如遇到问题，请提供以下信息:

1. 系统监控输出: `/usr/local/bin/xiaozhi-monitor.sh`
2. Docker容器状态: `docker-compose ps`
3. 系统资源使用: `htop` 截图
4. 错误日志: `docker-compose logs`
5. 性能测试结果: `monitor_performance.py` 输出

---

**注意**: 本配置为专用服务器激进优化版本，请确保服务器专门用于小智ESP32项目，避免运行其他重要服务。