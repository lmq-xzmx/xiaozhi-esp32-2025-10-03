#!/bin/bash
# ASR超级优化启动脚本
echo '🚀 启动ASR超级优化模式...'

export ASR_BATCH_SIZE=16
export ASR_BATCH_TIMEOUT=20
export ASR_MICRO_BATCH=True
export ASR_CHUNK_SIZE=256
export ASR_OVERLAP_SIZE=64
export ASR_STREAMING_CHUNK=128
export ASR_MAX_CONCURRENT=200
export ASR_WORKER_THREADS=16
export ASR_IO_THREADS=8
export ASR_QUEUE_SIZE=500
export ASR_ZERO_COPY=True
export ASR_MEMORY_POOL=True
export ASR_CACHE_SIZE_MB=8192
export ASR_PREALLOC_BUFFERS=True
export ASR_ENABLE_INT8=True
export ASR_ENABLE_FP16=True
export ASR_ENABLE_TURBO=True
export ASR_BEAM_SIZE=1
export ASR_MAX_LENGTH=512
export ASR_WEBSOCKET_BUFFER=4096
export ASR_TCP_NODELAY=True
export ASR_KEEPALIVE=True
export ASR_CPU_AFFINITY=0-15
export ASR_PRIORITY=high
export ASR_NUMA_POLICY=local

echo '⚡ ASR超级优化配置已加载'
echo '🎯 目标延迟减少: 100-150ms'
echo '🚀 启动ASR服务...'

python3 services/asr_service.py