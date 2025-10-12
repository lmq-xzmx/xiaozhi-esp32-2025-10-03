#!/bin/bash
# ASR延迟优化启动脚本
# 优化策略: extreme

echo '🚀 启动ASR延迟优化配置'
echo '=' * 50

# 设置环境变量
export ASR_MAX_CONCURRENT=200
export ASR_WORKER_THREADS=16
export ASR_IO_THREADS=8
export ASR_TIMEOUT=3
export ASR_MAX_RETRIES=1
export ASR_ENABLE_STREAMING=true
export ASR_CHUNK_SIZE=80
export ASR_OVERLAP_SIZE=40
export ASR_BATCH_SIZE=1
export ASR_ENABLE_FP16=true
export ASR_ZERO_COPY=true
export ASR_MEMORY_POOL=true
export ASR_MODEL_WARMUP=true
export ASR_ENABLE_CACHE=true
export ASR_CACHE_SIZE_MB=8192
export ASR_CACHE_TTL=300
export ASR_VAD_THRESHOLD=0.2
export ASR_SILENCE_TIMEOUT=0.3
export ASR_HIGH_PRIORITY=true
export ASR_CPU_AFFINITY=0,1,2,3
export ASR_TENSORRT=true

# 系统优化
echo '⚡ 应用系统优化...'
# 设置高优先级
sudo renice -10 $$

# 设置CPU亲和性
taskset -cp 0,1,2,3 $$

# 禁用CPU频率调节
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

echo '📊 当前配置:'
echo '   策略: extreme'
echo '   最大并发: $ASR_MAX_CONCURRENT'
echo '   音频块大小: $ASR_CHUNK_SIZE'
echo '   工作线程: $ASR_WORKER_THREADS'
echo '   缓存大小: $ASR_CACHE_SIZE_MB MB'

echo '🎯 预期延迟减少: 100-150ms'
echo '⚠️  请监控系统资源使用情况'

# 启动ASR服务
echo '🚀 启动优化后的ASR服务...'
python3 services/asr_service.py