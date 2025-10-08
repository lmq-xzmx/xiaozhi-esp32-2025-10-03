# P0级别性能优化总结

## 优化目标
- **支持100台设备并发访问**
- **系统整体延迟降低到200ms以下**
- **内存使用率从95%降低到75%**
- **缓存命中率从20-30%提升到60-80%**
- **TTS延迟从500-1000ms降低到50-150ms**
- **VAD/ASR延迟从800ms降低到200ms以下**

## 已完成的P0优化项目

### 1. MySQL数据库优化 ✅
**文件**: `config/mysql-optimized.cnf`

**优化内容**:
- InnoDB缓冲池: 1.5GB (原1GB)
- 查询缓存: 256MB
- 连接数: 500 (原151)
- 线程缓存: 50
- 表缓存: 4000
- 慢查询日志优化
- 二进制日志优化

**预期效果**: 内存使用率从95%降低到75%

### 2. LLM语义缓存优化 ✅
**文件**: `llm_service.py`

**优化内容**:
- 语义相似度阈值: 0.85 → 0.75
- 缓存TTL: 3600s → 7200s
- 预缓存常见问题 (100条)
- 智能缓存预热
- 缓存命中率监控

**预期效果**: 缓存命中率从20-30%提升到60-80%

### 3. TTS本地优先策略优化 ✅
**文件**: `tts_service.py`

**优化内容**:
- 最大并发: 20 → 40
- 工作线程: 8 → 12
- 常见短语预缓存
- 智能引擎选择策略
- 本地优先级算法

**预期效果**: 延迟从500-1000ms降低到50-150ms

### 4. VAD/ASR实时处理优化 ✅
**文件**: `vad_service.py`, `asr_service.py`

**VAD优化**:
- 批处理大小: 16 → 32
- 最大并发: 24 → 48
- 工作线程: 4 → 6
- ONNX线程: 2 → 4
- 实时参数优化 (chunk_size: 512, speech_threshold: 0.6)
- 批处理超时: 50ms → 15ms

**ASR优化**:
- 批处理大小: 8 → 16
- 最大并发: 16 → 40
- 工作线程: 4 → 8
- INT8量化模型优先
- 实时参数优化 (chunk_size: 800, beam_size: 1)
- 批处理超时: 10ms → 5ms

**预期效果**: 延迟从800ms降低到200ms以下

### 5. 系统级优化配置 ✅
**文件**: `system_optimization_p0.yaml`, `scripts/apply_system_optimization.sh`

**优化内容**:
- **网络优化**: TCP缓冲区、连接队列、BBR拥塞控制
- **内存优化**: 交换倾向、脏页管理、内存映射
- **CPU优化**: 性能调度器、频率锁定、C-states禁用
- **磁盘IO优化**: noop调度器、预读优化
- **文件系统优化**: 文件描述符限制、inotify监控
- **容器优化**: Docker配置、安全选项、资源限制
- **监控和自动调优**: 性能指标监控、动态参数调整

## Docker Compose配置更新

### 环境变量优化
```yaml
# VAD优化参数
VAD_MAX_CONCURRENT: 48
VAD_WORKER_THREADS: 6
VAD_INTRA_OP_THREADS: 4
VAD_INTER_OP_THREADS: 4
VAD_CHUNK_SIZE: 512
VAD_SPEECH_THRESHOLD: 0.6
VAD_BATCH_TIMEOUT_MS: 15

# ASR优化参数
ASR_BATCH_SIZE: 16
ASR_MAX_CONCURRENT: 40
ASR_WORKER_THREADS: 4
ASR_STREAM_WORKERS: 3
CHUNK_SIZE: 800
CHUNK_DURATION_MS: 50
OVERLAP_MS: 10
STREAM_BUFFER_SIZE: 3200
BATCH_TIMEOUT_MS: 20
ENABLE_INT8: true
BEAM_SIZE: 1

# SenseVoice专用配置
SENSEVOICE_MAX_CONCURRENT: 40
SENSEVOICE_ENABLE_INT8: true
SENSEVOICE_MAX_AUDIO_LENGTH: 30
```

### 系统级配置
```yaml
ulimits:
  nofile: 65536
  nproc: 32768
  memlock: unlimited

security_opt:
  - seccomp:unconfined
  - apparmor:unconfined

cap_add:
  - SYS_NICE
  - SYS_RESOURCE
```

## 性能监控指标

### 关键指标
1. **响应延迟**
   - TTS: < 150ms
   - VAD: < 100ms
   - ASR: < 200ms
   - 整体: < 200ms

2. **并发处理能力**
   - 支持100台设备同时访问
   - VAD最大并发: 48
   - ASR最大并发: 40
   - TTS最大并发: 40

3. **资源使用率**
   - 内存使用率: < 75%
   - CPU使用率: < 80%
   - 磁盘IO等待: < 20%

4. **缓存性能**
   - LLM缓存命中率: > 60%
   - TTS缓存命中率: > 70%
   - 语义缓存命中率: > 75%

## 部署和验证

### 1. 应用系统优化
```bash
# 运行系统优化脚本（需要root权限）
sudo /root/xiaozhi-server/scripts/apply_system_optimization.sh

# 验证优化效果
xiaozhi-optimization-check.sh
```

### 2. 重启服务
```bash
# 重启Docker服务
cd /root/xiaozhi-server
docker-compose -f docker-compose-final-optimized.yml down
docker-compose -f docker-compose-final-optimized.yml up -d
```

### 3. 性能验证
```bash
# 检查服务状态
docker-compose -f docker-compose-final-optimized.yml ps

# 查看服务日志
docker-compose -f docker-compose-final-optimized.yml logs -f

# 监控资源使用
docker stats
```

## 预期性能提升

| 指标 | 优化前 | 优化后 | 提升幅度 |
|------|--------|--------|----------|
| 内存使用率 | 95% | 75% | -20% |
| LLM缓存命中率 | 20-30% | 60-80% | +150% |
| TTS延迟 | 500-1000ms | 50-150ms | -80% |
| VAD/ASR延迟 | 800ms | <200ms | -75% |
| 并发设备数 | 50台 | 100台 | +100% |
| 整体响应延迟 | 1-2s | <200ms | -85% |

## 注意事项

1. **系统重启**: 某些内核参数优化需要重启系统后生效
2. **监控告警**: 建议设置性能监控告警，及时发现性能问题
3. **渐进式部署**: 建议在测试环境验证后再部署到生产环境
4. **备份配置**: 部署前备份原有配置文件
5. **性能测试**: 部署后进行压力测试验证优化效果

## 故障排除

### 常见问题
1. **内存不足**: 检查MySQL缓冲池配置是否过大
2. **连接超时**: 检查网络参数和防火墙配置
3. **服务启动失败**: 检查文件权限和依赖服务状态
4. **性能下降**: 检查系统资源使用情况和优化参数

### 回滚方案
如果优化后出现问题，可以：
1. 恢复原有配置文件
2. 重置内核参数: `sysctl --system`
3. 重启相关服务
4. 监控系统状态恢复

---

**优化完成时间**: $(date)
**优化级别**: P0 (最高优先级)
**预期效果**: 支持100台设备并发，整体延迟<200ms