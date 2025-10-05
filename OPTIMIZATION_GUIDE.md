# 小智ESP32服务器多设备并发优化指南

## 问题分析

### 现象描述
- **单设备**: 初始响应慢，声音结结巴巴，然后逐渐改善
- **多设备**: 持续结结巴巴，性能不稳定

### 根本原因
1. **SenseVoice模型冷启动延迟** - 首次加载模型需要时间
2. **CPU资源竞争** - 当前负载6.79，接近饱和
3. **内存压力** - 可用内存仅975MB，多实例运行时不足
4. **并发控制不足** - 缺乏有效的连接和资源管理

## 优化方案

### 1. Docker资源配置优化

#### 当前问题
- 无资源限制，容器可能占用过多系统资源
- 缺乏内存和CPU的合理分配

#### 解决方案
```bash
# 停止当前服务
docker-compose down

# 使用优化配置启动
docker-compose -f docker-compose_optimized.yml up -d
```

#### 优化要点
- **CPU限制**: 主容器限制3核，避免占满所有CPU
- **内存分配**: 主容器6GB，为SenseVoice预留充足内存
- **共享内存**: 增加2GB共享内存，提升模型加载性能
- **环境变量**: 设置OMP_NUM_THREADS=2，控制并行计算线程

### 2. SenseVoice配置优化

#### 当前问题
- chunk_size=1024可能导致延迟
- 缺乏并发会话控制
- VAD配置不够优化

#### 解决方案
```bash
# 备份当前配置
cp data/.config.yaml data/.config.yaml.backup

# 应用优化配置
cp data/.config_optimized.yaml data/.config.yaml

# 重启服务
docker-compose restart xiaozhi-esp32-server
```

#### 优化要点
- **chunk_size**: 512 → 降低延迟
- **max_concurrent_sessions**: 2 → 限制并发数
- **VAD优化**: 减少单段时间到15秒
- **线程池配置**: ASR/LLM/TTS分别限制工作线程

### 3. 代码层面优化

#### SenseVoice模型池化
```python
# 当前问题：每次请求都可能重新加载模型
# 优化方案：使用模型池复用实例

# 部署优化版本
cp sensevoice_optimized.py asr_providers/sensevoice_stream.py
```

#### 优化特性
- **模型池**: 最多2个模型实例，支持并发复用
- **连接级锁**: 防止同一连接的并发冲突
- **内存管理**: 及时清理临时文件和垃圾回收
- **超时控制**: 10秒获取模型超时，避免无限等待

### 4. 系统监控

#### 性能监控脚本
```bash
# 安装依赖
pip install psutil websockets

# 运行监控（监控5分钟）
python monitor_performance.py --mode monitor --duration 300

# 运行并发测试
python monitor_performance.py --mode test --connections 2

# 同时运行监控和测试
python monitor_performance.py --mode both --duration 300 --connections 2
```

#### 监控指标
- **CPU使用率**: 目标<70%
- **内存使用率**: 目标<70%
- **并发连接数**: 建议≤2
- **响应延迟**: 目标<20秒

## 部署步骤

### 第一步：应用Docker优化
```bash
# 1. 停止当前服务
docker-compose down

# 2. 使用优化配置
docker-compose -f docker-compose_optimized.yml up -d

# 3. 检查容器状态
docker ps
```

### 第二步：更新配置文件
```bash
# 1. 备份原配置
cp data/.config.yaml data/.config.yaml.$(date +%Y%m%d_%H%M%S)

# 2. 应用优化配置
cp data/.config_optimized.yaml data/.config.yaml

# 3. 重启主服务
docker-compose -f docker-compose_optimized.yml restart xiaozhi-esp32-server
```

### 第三步：部署代码优化（可选）
```bash
# 1. 备份原文件
cp sensevoice_stream.py sensevoice_stream.py.backup

# 2. 部署优化版本
cp sensevoice_optimized.py sensevoice_stream.py

# 3. 重启服务
docker-compose -f docker-compose_optimized.yml restart xiaozhi-esp32-server
```

### 第四步：性能验证
```bash
# 1. 启动监控
python monitor_performance.py --mode monitor --duration 180 &

# 2. 进行真机测试
# - 连接1台设备，测试响应速度
# - 连接2台设备，测试并发性能

# 3. 查看监控报告
# 报告文件：performance_report_YYYYMMDD_HHMMSS.json
```

## 预期效果

### 单设备优化效果
- ✅ **消除冷启动延迟**: 模型预加载，首次响应<5秒
- ✅ **稳定音频输出**: 消除结结巴巴现象
- ✅ **响应时间**: 端到端延迟<15秒

### 多设备优化效果
- ✅ **并发稳定性**: 2台设备同时使用无卡顿
- ✅ **资源控制**: CPU使用率<70%，内存使用率<70%
- ✅ **公平调度**: 设备间响应时间相近

## 故障排除

### 常见问题

#### 1. 内存不足错误
```bash
# 检查内存使用
docker stats xiaozhi-esp32-server

# 解决方案：增加Docker内存限制
# 在docker-compose_optimized.yml中调整memory: 8G
```

#### 2. 模型加载失败
```bash
# 检查模型文件
ls -la models/SenseVoiceSmall/

# 检查容器日志
docker logs xiaozhi-esp32-server

# 确保模型文件正确挂载
```

#### 3. 并发连接超时
```bash
# 检查当前连接数
netstat -an | grep :8000

# 调整并发限制
# 在.config_optimized.yaml中减少max_concurrent_sessions
```

### 性能调优

#### 根据硬件配置调整

**4核8GB内存**:
```yaml
max_concurrent_sessions: 2
asr_max_workers: 2
cpus: '3.0'
memory: 6G
```

**8核16GB内存**:
```yaml
max_concurrent_sessions: 4
asr_max_workers: 4
cpus: '6.0'
memory: 12G
```

**2核4GB内存**:
```yaml
max_concurrent_sessions: 1
asr_max_workers: 1
cpus: '1.5'
memory: 3G
```

## 监控和维护

### 日常监控
```bash
# 每日性能检查
python monitor_performance.py --mode monitor --duration 60

# 检查Docker资源使用
docker stats --no-stream

# 检查系统负载
uptime
```

### 定期维护
```bash
# 清理临时文件
docker exec xiaozhi-esp32-server find /tmp -name "*.wav" -mtime +1 -delete

# 重启服务（每周）
docker-compose -f docker-compose_optimized.yml restart

# 检查日志大小
docker logs xiaozhi-esp32-server --tail 100
```

## 总结

通过以上优化方案，可以有效解决多设备并发时的性能问题：

1. **Docker资源配置** - 合理分配CPU和内存资源
2. **SenseVoice优化** - 模型池化和并发控制
3. **配置调优** - 降低延迟和提升稳定性
4. **监控体系** - 实时掌握系统性能状态

建议按步骤逐一应用，并通过监控脚本验证效果。如遇问题，可参考故障排除部分或回滚到备份配置。